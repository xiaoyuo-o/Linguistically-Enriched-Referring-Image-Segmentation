import argparse
import os
import warnings
import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference
from model import build_segmenter
from utils.dataset import RefDataset
# from utils.dataset_2_branches import RefDataset
from utils.misc import setup_logger
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    parser.add_argument("--checkpoint", required=True, type=str, help="checkpoint to evaluate")
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build dataset & dataloader
    test_data = RefDataset(lmdb_dir=args.test_lmdb,
                          mask_dir=args.mask_root,
                          dataset=args.dataset,
                          split=args.test_split,
                          args=args,
                          mode='test',
                          input_size=args.input_size,
                          word_length=args.word_len,
                          word_en_length = args.word_en_length,
                            new_lmdb_dir = args.test_lmdb,
                            is_decouple = args.is_decouple)

    # test_data = RefDataset(lmdb_dir=args.test_lmdb,
    #                       mask_dir=args.mask_root,
    #                       dataset=args.dataset,
    #                       split=args.test_split,
    #                       mode='test',
    #                       input_size=args.input_size,
    #                       word_length=args.word_len,
    #                       word_en_length = args.word_en_length,
    #                         new_lmdb_dir = args.test_lmdb,
    #                         is_decouple = args.is_decouple)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'test'))
    else:
        writer = None


    torch.cuda.set_device(0)

    model, _ = build_segmenter(args)

    model = model.cuda()
    model.eval()

    # 🔥 强制检查 + 修复所有buffer
    for name, p in model.named_parameters():
        if not p.is_cuda:
            p.data = p.data.cuda()

    for name, b in model.named_buffers():
        if not b.is_cuda:
            b.data = b.data.cuda()

    model = torch.nn.DataParallel(model)
    logger.info(model)
    args.model_dir = args.checkpoint
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # inference
    inference(test_loader, model, args, writer if args.rank==0 else None)


if __name__ == '__main__':
    main()
