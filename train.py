import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import RefDataset
# from utils.dataset_2_branches import RefDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default=None,
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    # wandb
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    else:
        writer = None
    dist.barrier()

    # build model
    model, param_list = build_segmenter(args)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()

    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int(
        (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                            mask_dir=args.mask_root,
                            dataset=args.dataset,
                            split=args.train_split,
                            mode='train',
                            input_size=args.input_size,
                            word_length=args.word_len,
                            word_en_length = args.word_en_length,
                            new_lmdb_dir=args.new_train_lmdb,
                            args=args,
                            is_decouple = args.is_decouple)
    # train_data = RefDataset(lmdb_dir=args.train_lmdb,
    #                         mask_dir=args.mask_root,
    #                         dataset=args.dataset,
    #                         split=args.train_split,
    #                         mode='train',
    #                         input_size=args.input_size,
    #                         word_length=args.word_len,
    #                         word_en_length = args.word_en_length,
    #                         new_lmdb_dir=args.new_train_lmdb,
    #                         is_decouple = args.is_decouple)
    val_data = RefDataset(lmdb_dir=args.val_lmdb,
                          mask_dir=args.mask_root,
                          dataset=args.dataset,
                          split=args.val_split,
                          mode='val',
                          args=args, 
                          input_size=args.input_size,
                          word_length=args.word_len,
                          word_en_length = args.word_en_length,
                            new_lmdb_dir = args.new_val_lmdb,
                            is_decouple = args.is_decouple)

    # val_data = RefDataset(lmdb_dir=args.val_lmdb,
    #                       mask_dir=args.mask_root,
    #                       dataset=args.dataset,
    #                       split=args.val_split,
    #                       mode='val',
    #                       input_size=args.input_size,
    #                       word_length=args.word_len,
    #                       word_en_length = args.word_en_length,
    #                         new_lmdb_dir = args.new_val_lmdb,
    #                         is_decouple = args.is_decouple)


    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False)


    best_IoU_raw = 0.0
    best_IoU_new = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint['epoch']
            best_IoU_raw = checkpoint.get("best_iou_raw", 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            if args.if_two_branches:
                best_IoU_new = checkpoint.get("best_iou_new", 0.0)

            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']
                )
            )
        else:
            raise ValueError(
                f"=> resume failed! no checkpoint found at '{args.resume}'. Please check args.resume again!"
            )

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)
        train_data.set_epoch(epoch_log)

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log,
            args, writer if args.rank == 0 else None)

        # evaluation
        if args.if_two_branches:
            iou_raw, prec_dict_raw, iou_new, prec_dict_new, cons = validate(
                val_loader, model, epoch_log, args, writer if args.rank == 0 else None
            )
            logger.info(f"Epoch {epoch_log}/{args.epochs} - raw_IoU: {iou_raw:.4f}")
            logger.info(f"Epoch {epoch_log}/{args.epochs} - new_IoU: {iou_new:.4f}")
            logger.info(f"Val Consistency Score: {cons:.4f}")
        else:
            iou_raw, prec_dict_raw, __, ___ = validate(
                val_loader, model, epoch_log, args, writer if args.rank == 0 else None
            )
            logger.info(f"Epoch {epoch_log}/{args.epochs} - IoU: {iou_raw:.4f}")

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            state = {
                'epoch': epoch_log,
                'cur_raw_iou': iou_raw,
                'best_iou_raw': best_IoU_raw,
                'prec_raw': prec_dict_raw,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            if args.if_two_branches:
                state.update({
                    'cur_new_iou': iou_new,
                    'best_iou_new': best_IoU_new,
                    'prec_new': prec_dict_new,
                })
            torch.save(state, lastname)

            # best model check
            if iou_raw >= best_IoU_raw:
                best_IoU_raw = iou_raw
                bestname = os.path.join(args.output_dir, "raw_best_model.pth")
                shutil.copyfile(lastname, bestname)
            if args.if_two_branches and iou_new >= best_IoU_new:
                best_IoU_new = iou_new
                bestname = os.path.join(args.output_dir, "new_best_model.pth")
                shutil.copyfile(lastname, bestname)

        if args.rank == 0:
            writer.add_scalar('Val/raw_IoU', iou_raw, epoch_log)
            if args.if_two_branches:
                writer.add_scalar('Val/new_IoU', iou_new, epoch_log)
            for key, value in prec_dict_raw.items():
                writer.add_scalar(f'Prec_raw/{key}', value, epoch_log)
            if args.if_two_branches:
                for key, value in prec_dict_new.items():
                    writer.add_scalar(f'Prec_new/{key}', value, epoch_log)

        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)
    if dist.get_rank() == 0:
        writer.close()

    logger.info("* Best raw_IoU={} * ".format(best_IoU_raw))
    if args.if_two_branches:
        logger.info("* Best new_IoU={} * ".format(best_IoU_new))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
