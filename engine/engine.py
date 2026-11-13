import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from utils.dataset import *
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args, writer):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')

    loss_meters = {}

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]
    for i, batch in enumerate(train_loader):
        if len(batch) == 5:
            image, text, target, en_labels, en_labels_raw = batch
        elif len(batch) == 4:
            image, text, target, en_labels_raw =batch
            en_labels = None
        elif len(batch) == 3:
            image, text, target = batch
            en_labels = None
            en_labels_raw = None
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with amp.autocast():
            pred, target, loss_dict = model(image, text,  en_labels=en_labels, en_labels_raw=en_labels_raw, mask=target)

            # 计算总损失
            total_loss = 0
            for loss_name, loss_value in loss_dict.items():
                if loss_name == 'loss_nce':
                    loss_value = 0.05 * loss_value
                if loss_name == 'loss_mask_align':
                    loss_value = 0.2 * loss_value
                # if loss_name == 'loss_en':
                #     loss_value = 0.2* loss_value
                # 对于loss_mask_align，应用0.1的权重
                # if loss_name == 'loss_style':
                #     loss_value = 0.1*loss_value
                total_loss += loss_value
                
                # 确保每个损失项都有对应的度量器
                if loss_name not in loss_meters:
                    loss_meters[loss_name] = AverageMeter(loss_name, ':2.4f')
                    # 动态更新进度显示列表
                    progress.meters.append(loss_meters[loss_name])            

        # backward
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(total_loss.detach())

        sub_losses = {}
        for loss_name, loss_value in loss_dict.items():
            detached_loss = loss_value.detach()
            dist.all_reduce(detached_loss)
            sub_losses[loss_name] = detached_loss / dist.get_world_size()


        dist.all_reduce(iou)
        dist.all_reduce(pr5)

        total_loss = total_loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(total_loss.item(), image.size(0))
    # 更新每个子损失的度量
        for loss_name, loss_value in sub_losses.items():
            loss_meters[loss_name].update(loss_value.item(), image.size(0))

        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                # 使用 TensorBoard 记录
                writer.add_scalar("time/batch", batch_time.val, epoch * len(train_loader) + (i + 1))
                writer.add_scalar("time/data", data_time.val, epoch * len(train_loader) + (i + 1))
                writer.add_scalar("training/lr", lr.val, epoch * len(train_loader) + (i + 1))
                writer.add_scalar("training/loss", loss_meter.val, epoch * len(train_loader) + (i + 1))
                #记录每个子损失
                for loss_name, meter in loss_meters.items():
                    writer.add_scalar(f"training/{loss_name}", meter.val, epoch * len(train_loader) + (i + 1))
                

                writer.add_scalar("training/iou", iou_meter.val, epoch * len(train_loader) + (i + 1))
                writer.add_scalar("training/prec@50", pr_meter.val, epoch * len(train_loader) + (i + 1))

def compute_iou(p1, p2):
    p1 = (p1 > 0.5).float()
    p2 = (p2 > 0.5).float()

    inter = (p1 * p2).sum(dim=(1,2,3))
    union = ((p1 + p2) > 0).float().sum(dim=(1,2,3))

    return (inter / (union + 1e-6)).mean().item()

@torch.no_grad()
def validate(val_loader, model, epoch, args, writer):

    torch.cuda.empty_cache() 
    consistency_list = []
    iou_raw_list = []
    iou_en_list = []
    model.eval()
    time.sleep(2)

    for batch in val_loader:
        if len(batch) == 6:
            imgs, texts, masks, param, en_labels, en_labels_raw = batch
        elif len(batch) == 5:
            imgs, texts, masks, param, en_labels_raw =batch
            en_labels = None
        elif len(batch) == 4:
            imgs, texts, masks, param = batch
            en_labels = None
            en_labels_raw = None

        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)

        # inference
        with torch.no_grad():
            outputs = model(imgs, texts, en_labels=en_labels, en_labels_raw=en_labels_raw)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                preds_raw, preds_en = outputs
                consistency = compute_iou(preds_raw, preds_en)
                consistency_list.append(consistency)
            else:
                preds_raw, preds_en = outputs, None

        preds_raw = torch.sigmoid(preds_raw)
        if preds_raw.shape[-2:] != imgs.shape[-2:]:
            preds_raw = F.interpolate(preds_raw,
                                      size=imgs.shape[-2:],
                                      mode='bicubic',
                                      align_corners=True).squeeze(1)

        if preds_en is not None:
            preds_en = torch.sigmoid(preds_en)
            if preds_en.shape[-2:] != imgs.shape[-2:]:
                preds_en = F.interpolate(preds_en,
                                         size=imgs.shape[-2:],
                                         mode='bicubic',
                                         align_corners=True).squeeze(1)

        # process one batch
        for idx, (pred_raw, mask, mat, ori_size) in enumerate(
                zip(preds_raw, masks, param['inverse'], param['ori_size'])):
            h, w = np.array(ori_size)
            mat = np.array(mat)

            # raw branch
            pred_raw = pred_raw.cpu().numpy()
            # pred_raw = cv2.warpAffine(pred_raw, mat, (w, h),
            #                           flags=cv2.INTER_CUBIC,
            #                           borderValue=0.)
            pred_raw = np.array(pred_raw > 0.35)
            if isinstance(mask, np.ndarray):
                mask = mask
            else:
                mask = mask.cpu().numpy()    
            # mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            # mask = mask / 255.

            inter_raw = np.logical_and(pred_raw, mask)
            union_raw = np.logical_or(pred_raw, mask)
            iou_raw = np.sum(inter_raw) / (np.sum(union_raw) + 1e-6)
            iou_raw_list.append(iou_raw)

            # en branch (only if preds_en exists)
            if preds_en is not None:
                pred_en = preds_en[idx].cpu().numpy()
                # pred_en = cv2.warpAffine(pred_en, mat, (w, h),
                #                          flags=cv2.INTER_CUBIC,
                #                          borderValue=0.)
                pred_en = np.array(pred_en > 0.35)

                inter_en = np.logical_and(pred_en, mask)
                union_en = np.logical_or(pred_en, mask)
                iou_en = np.sum(inter_en) / (np.sum(union_en) + 1e-6)
                iou_en_list.append(iou_en)

    torch.cuda.empty_cache() 

    # gather raw
    iou_raw_list = np.stack(iou_raw_list)
    iou_raw_list = torch.from_numpy(iou_raw_list).to(imgs.device)
    iou_raw_list = concat_all_gather(iou_raw_list)
    prec_raw_list = []

    # gather en (if available)
    if len(iou_en_list) > 0:
        iou_en_list = np.stack(iou_en_list)
        iou_en_list = torch.from_numpy(iou_en_list).to(imgs.device)
        iou_en_list = concat_all_gather(iou_en_list)
        prec_en_list = []
    else:
        iou_en_list = None
        prec_en_list = None

    # precision@k
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp_raw = (iou_raw_list > thres).float().mean()
        prec_raw_list.append(tmp_raw)

        if iou_en_list is not None:
            tmp_en = (iou_en_list > thres).float().mean()
            prec_en_list.append(tmp_en)

    iou_raw = iou_raw_list.mean()
    prec_raw = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        prec_raw[key] = prec_raw_list[i].item()

    head_raw = 'Evaluation: Epoch=[{}/{}]  IoU_raw={:.2f}'.format(
        epoch, args.epochs, 100. * iou_raw.item())
    temp_raw = '  '.join([f"{k}: {100*v:.2f}" for k, v in prec_raw.items()])
    logger.info(head_raw + "  " + temp_raw)

    if iou_en_list is not None:
        iou_en = iou_en_list.mean()
        prec_en = {}
        for i, thres in enumerate(range(5, 10)):
            key = 'Pr@{}'.format(thres * 10)
            prec_en[key] = prec_en_list[i].item()

        head_en = 'Evaluation: Epoch=[{}/{}]  IoU_en={:.2f}'.format(
            epoch, args.epochs, 100. * iou_en.item())
        temp_en = '  '.join([f"{k}: {100*v:.2f}" for k, v in prec_en.items()])
        logger.info(head_en + "  " + temp_en)
    else:
        iou_en, prec_en = None, None

    if args.rank == 0:
        writer.add_scalar('validation/iou_raw', iou_raw.item(), epoch)
        for k, v in prec_raw.items():
            writer.add_scalar(f'validation/{k}_raw', v, epoch)

        if iou_en is not None:
            writer.add_scalar('validation/iou_en', iou_en.item(), epoch)
            for k, v in prec_en.items():
                writer.add_scalar(f'validation/{k}_en', v, epoch)

    avg_consistency = sum(consistency_list) / len(consistency_list)
    return iou_raw.item(), prec_raw, (iou_en.item() if iou_en else None), prec_en, avg_consistency

@torch.no_grad()
def inference(test_loader, model, args, writer, if_infer=True):
    logger.info("=> Profiling model FLOPs / Params / Inference speed using real batch...")

    import time
    from thop import profile

    model.eval()

    save_dir = os.path.join(args.output_dir, "attn_fixed")
    os.makedirs(save_dir, exist_ok=True)

    iou_raw_list = []
    cum_I, cum_U = 0, 0

    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    time.sleep(1)

    for img, mask, param, is_decouple in tbar:

        img = img.cuda(non_blocking=True)
        mask_raw = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)

        seg_id = int(param['seg_id'][0].cpu().numpy())

        # =========================
        # sentence loop
        # =========================
        for sent in param['sents']:

            caption_raw = ' '.join(sent[0].replace(',', ' ,').split())

            text_raw, char_to_token_map_list = tokenize_en(
                caption_raw, args.word_len, True
            )
            text_raw = text_raw.cuda(non_blocking=True)

            # =========================
            # MODEL FORWARD (完全不动你原来的逻辑)
            # =========================
            pred_raw = model(
                img,
                text_raw,
                en_labels=None,
                en_labels_raw=None,
                if_infer=if_infer
            )

            pred_raw = torch.sigmoid(pred_raw)

            # =========================
            # ATTENTION SAVE (ONLY CHANGE HERE)
            # =========================
            attn_maps = []

            for layer in model.module.decoder.layers:
                if hasattr(layer, "last_attn_map"):
                    attn_maps.append(layer.last_attn_map.detach().cpu())
            # print(len(attn_maps))
            save_path = os.path.join(save_dir, f"{seg_id}.pt")

            torch.save({
                "seg_id": seg_id,
                "caption": caption_raw,
                "attn": attn_maps
            }, save_path)

            print(f"[SAVE] ref={seg_id} saved -> {save_path}")

            # =========================
            # IOU PART (UNCHANGED)
            # =========================
            if pred_raw.shape[-2:] != img.shape[-2:]:
                pred_raw = F.interpolate(
                    pred_raw,
                    size=img.shape[-2:],
                    mode='bicubic',
                    align_corners=True
                ).squeeze()

            pred_raw = pred_raw.cpu().numpy()
            pred_bin = np.array(pred_raw > 0.35, dtype=bool)

            if not isinstance(mask, np.ndarray):
                mask = mask.cpu().numpy()

            inter = np.logical_and(pred_bin, mask)
            union = np.logical_or(pred_bin, mask)

            I = np.sum(inter)
            U = np.sum(union)

            cum_I += I
            cum_U += U

            iou_raw = I / (U + 1e-6)
            iou_raw_list.append(iou_raw)

    # =========================
    # FINAL METRICS (UNCHANGED)
    # =========================
    iou_raw_list = np.stack(iou_raw_list)
    iou_raw = np.mean(iou_raw_list)
    oIoU = cum_I / (cum_U + 1e-6)

    logger.info('=> Metric Calculation <=')
    logger.info('mIoU={:.2f}, OIoU={:.2f}'.format(100.*iou_raw, 100.*oIoU))

    return iou_raw.item(), oIoU

    
# @torch.no_grad()
# def inference(test_loader, model, args, writer, if_infer=True):
    
#     iou_raw_list = []
#     # iou_en_list = []
    
#     tbar = tqdm(test_loader, desc='Inference:', ncols=100)
#     model.eval()
#     time.sleep(2)
#     # for img, param, is_decouple in tbar:
#     for img, mask, param, is_decouple in tbar:
#         # print(type(mask))
#         # data
#         img = img.cuda(non_blocking=True)
#         mask_raw = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
#         # dump image & mask
#         if args.visualize:
#             seg_id = param['seg_id'][0].cpu().numpy()
#             img_name = '{}-img.jpg'.format(seg_id)
#             mask_name = '{}-mask.png'.format(seg_id)
#             cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
#                         img=param['ori_img'][0].cpu().numpy())
#             cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
#                         img=mask_raw)
#         # multiple sentences
#         for sent in param['sents']:
#             # mask = mask_raw
#             # mask = mask / 255.

#             # text_raw = tokenize(sent, args.word_len, True)
#             # text_raw = text_raw.cuda(non_blocking=True)
#             caption_raw = ' '.join(sent[0].replace(',', ' ,').split())
#             text_raw, char_to_token_map_list = tokenize_en(caption_raw, args.word_len, True)
#             text_raw = text_raw.cuda(non_blocking=True)
#             if is_decouple:
#                 anno = generate_anno(caption_raw)
#                 tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
#                 other_entity_map, rel_positive_map = get_token_positive_map_by_parse(anno, args.word_len, char_to_token_map_list)
#                 positive_map = torch.from_numpy(positive_map).unsqueeze(0)
#                 modify_positive_map = torch.from_numpy(modify_positive_map).unsqueeze(0)
#                 pron_positive_map = torch.from_numpy(pron_positive_map).unsqueeze(0)
#                 other_entity_map = torch.from_numpy(other_entity_map).unsqueeze(0)
#                 rel_positive_map = torch.from_numpy(rel_positive_map).unsqueeze(0)
#                 en_labels = {
#                 'tokens_positive': tokens_positive.astype(np.int64),
#                 'positive_map': positive_map,
#                 'modify_positive_map': modify_positive_map,
#                 'pron_positive_map': pron_positive_map,
#                 'other_entity_map': other_entity_map,
#                 'rel_positive_map': rel_positive_map
#                 }
#             else:
#                 en_labels = None
#             # text_en = text_en.cuda(non_blocking=True)
#             # text = torch.cat([text_raw,text_en], dim=1)

#             # inference
#             pred_raw = model(img, text_raw, en_labels=en_labels, en_labels_raw=None, if_infer=if_infer)

#             pred_raw = torch.sigmoid(pred_raw)
            


#             if pred_raw.shape[-2:] != img.shape[-2:]:
#                 pred_raw = F.interpolate(pred_raw,
#                                      size=img.shape[-2:],
#                                      mode='bicubic',
#                                      align_corners=True).squeeze()


#             # process one sentence
#             h, w = param['ori_size'].numpy()[0]
#             mat = param['inverse'].numpy()[0]



#             pred_raw = pred_raw.cpu().numpy()
            
#             # pred_raw = cv2.warpAffine(pred_raw, mat, (w,h),
#             #                       flags=cv2.INTER_CUBIC,
#             #                       borderValue=0.)
#             # resizer = A.Compose([A.Resize(h, w, always_apply=True)])
#             # resized = resizer(image=pred_raw)
#             # pred_raw = resized['image']

#             # pred_raw = cv2.resize(pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
#             pred_raw_ = np.array(pred_raw > 0.35)

#             if isinstance(mask, np.ndarray):
#                 mask = mask
#             else:
#                 mask = mask.cpu().numpy()          
#             # mask = mask.cpu().numpy()
#             inter_raw = np.logical_and(pred_raw_, mask)


#             union_raw = np.logical_or(pred_raw_, mask)


#             iou_raw = np.sum(inter_raw) / (np.sum(union_raw) + 1e-6)


#             iou_raw_list.append(iou_raw)

#             # dump prediction
#             if args.visualize:
#                 pred_raw = cv2.resize(pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
#                 pred_raw = np.array(pred_raw > 0.35)
#                 pred_raw = np.array(pred_raw_*255, dtype=np.uint8)
#                 sent = "_".join(sent[0].split(" "))
#                 pred_raw_name = '{}-iou_raw={:.2f}-{}.png'.format(seg_id, iou_raw*100, sent)

#                 # pred_en = np.array(pred_en*255, dtype=np.uint8)
#                 # sent_en = "_".join(sent_en[0].split(" "))
#                 # pred_en_name = '{}-iou_en={:.2f}-{}.png'.format(seg_id, iou_en*100, sent_en)
                
#                 cv2.imwrite(filename=os.path.join(args.vis_dir, pred_raw_name),
#                             img=pred_raw)

#     logger.info('=> Metric Calculation <=')

#     iou_raw_list = np.stack(iou_raw_list)
#     iou_raw_list = torch.from_numpy(iou_raw_list).to(img.device)
#     prec_raw_list = []


#     for thres in torch.arange(0.5, 1.0, 0.1):

#         tmp_raw = (iou_raw_list > thres).float().mean()
#         prec_raw_list.append(tmp_raw)


    

#     iou_raw = iou_raw_list.mean()


#     prec_raw = {}

#     for i, thres in enumerate(range(5, 10)):
#         key = 'Pr@{}'.format(thres*10)

#         value_raw = prec_raw_list[i].item()


#         prec_raw[key] = value_raw


#     logger.info('IoU_raw={:.2f}'.format(100.*iou_raw.item()))



#     for k, v in prec_raw.items():
#         logger.info('{}: {:.2f}.'.format(k, 100.*v))
#     writer.add_scalar('inference_raw/iou', iou_raw.item())
#     for k, v in prec_raw.items():
#         writer.add_scalar(f'inference_raw/{k}', v)



#     return iou_raw.item(), prec_raw

@torch.no_grad()
def inference_(test_loader, model, args, writer, if_infer=False):
    
    iou_raw_list = []
    iou_en_list = []
    
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param, is_decouple in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        # dump image & mask
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask)
        # multiple sentences
        for sent, sent_en in zip(param['sents_raw'], param['sents_en']):
            mask = mask / 255.
            caption_raw = ' '.join(sent[0].replace(',', ' ,').split())
            text_raw, char_to_token_map_list_raw = tokenize_en(caption_raw, args.word_len, True)
            
            caption_new = ' '.join(sent_en[0].replace(',', ' ,').split())
            text_en, char_to_token_map_list = tokenize_en(caption_new, args.word_en_length, True)
            
            if is_decouple:
                anno = generate_anno(caption_new)
                tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, rel_positive_map = get_token_positive_map_by_parse(anno, args.word_en_length, char_to_token_map_list)
                positive_map = torch.from_numpy(positive_map).unsqueeze(0)
                modify_positive_map = torch.from_numpy(modify_positive_map).unsqueeze(0)
                pron_positive_map = torch.from_numpy(pron_positive_map).unsqueeze(0)
                other_entity_map = torch.from_numpy(other_entity_map).unsqueeze(0)
                rel_positive_map = torch.from_numpy(rel_positive_map).unsqueeze(0)
                en_labels = {
                'tokens_positive': tokens_positive.astype(np.int64),
                'positive_map': positive_map,
                'modify_positive_map': modify_positive_map,
                'pron_positive_map': pron_positive_map,
                'other_entity_map': other_entity_map,
                'rel_positive_map': rel_positive_map
                }


                anno_raw = generate_anno(caption_raw)
                tokens_positive_raw, positive_map_raw, modify_positive_map_raw, pron_positive_map_raw, \
                other_entity_map_raw, rel_positive_map_raw = get_token_positive_map_by_parse(anno_raw, args.word_len, char_to_token_map_list_raw)
                positive_map_raw = torch.from_numpy(positive_map_raw).unsqueeze(0)
                modify_positive_map_raw = torch.from_numpy(modify_positive_map_raw).unsqueeze(0)
                pron_positive_map_raw = torch.from_numpy(pron_positive_map_raw).unsqueeze(0)
                other_entity_map_raw = torch.from_numpy(other_entity_map_raw).unsqueeze(0)
                rel_positive_map_raw = torch.from_numpy(rel_positive_map_raw).unsqueeze(0)
                en_labels_raw = {
                'tokens_positive': tokens_positive_raw.astype(np.int64),
                'positive_map': positive_map_raw,
                'modify_positive_map': modify_positive_map_raw,
                'pron_positive_map': pron_positive_map_raw,
                'other_entity_map': other_entity_map_raw,
                'rel_positive_map': rel_positive_map_raw
                }



                
            else:
                en_labels = None
                en_labels = None
            text_raw = text_raw.cuda(non_blocking=True)
            text_en = text_en.cuda(non_blocking=True)
            text = torch.cat([text_raw,text_en], dim=1)

            # inference
            pred_raw, pred_en = model(img, text, en_labels=en_labels,en_labels_raw=en_labels_raw, if_infer=if_infer)

            pred_raw = torch.sigmoid(pred_raw)
            pred_en = torch.sigmoid(pred_en)


            if pred_raw.shape[-2:] != img.shape[-2:]:
                pred_raw = F.interpolate(pred_raw,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()

            if pred_en.shape[-2:] != img.shape[-2:]:
                pred_en = F.interpolate(pred_en,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]



            pred_raw = pred_raw.cpu().numpy()
            pred_raw = cv2.warpAffine(pred_raw, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred_raw = np.array(pred_raw > 0.35)

            pred_en = pred_en.cpu().numpy()
            pred_en = cv2.warpAffine(pred_en, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred_en = np.array(pred_en > 0.35)
            # iou

            inter_raw = np.logical_and(pred_raw, mask)
            inter_en = np.logical_and(pred_en, mask)

            union_raw = np.logical_or(pred_raw, mask)
            union_en = np.logical_or(pred_en, mask)

            iou_raw = np.sum(inter_raw) / (np.sum(union_raw) + 1e-6)
            iou_en = np.sum(inter_en) / (np.sum(union_en) + 1e-6)

            iou_raw_list.append(iou_raw)
            iou_en_list.append(iou_en)
            # dump prediction
            if args.visualize:
                pred_raw = np.array(pred_raw*255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_raw_name = '{}-iou_raw={:.2f}-{}.png'.format(seg_id, iou_raw*100, sent)

                pred_en = np.array(pred_en*255, dtype=np.uint8)
                sent_en = "_".join(sent_en[0].split(" "))
                pred_en_name = '{}-iou_en={:.2f}-{}.png'.format(seg_id, iou_en*100, sent_en)
                
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_raw_name),
                            img=pred_raw)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_en_name),
                            img=pred_en)
    logger.info('=> Metric Calculation <=')

    iou_raw_list = np.stack(iou_raw_list)
    iou_raw_list = torch.from_numpy(iou_raw_list).to(img.device)
    prec_raw_list = []

    iou_en_list = np.stack(iou_en_list)
    iou_en_list = torch.from_numpy(iou_en_list).to(img.device)
    prec_en_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):

        tmp_raw = (iou_raw_list > thres).float().mean()
        prec_raw_list.append(tmp_raw)

        tmp_en = (iou_en_list > thres).float().mean()
        prec_en_list.append(tmp_en)
    

    iou_raw = iou_raw_list.mean()
    iou_en = iou_en_list.mean()

    prec_raw = {}
    prec_en = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)

        value_raw = prec_raw_list[i].item()
        value_en = prec_en_list[i].item()

        prec_raw[key] = value_raw
        prec_en[key] = value_en

    logger.info('IoU_raw={:.2f}'.format(100.*iou_raw.item()))
    logger.info('IoU_en={:.2f}'.format(100.*iou_en.item()))


    for k, v in prec_raw.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))
    writer.add_scalar('inference_raw/iou', iou_raw.item())
    for k, v in prec_raw.items():
        writer.add_scalar(f'inference_raw/{k}', v)

    for k, v in prec_en.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))
    writer.add_scalar('inference_en/iou', iou_en.item())
    for k, v in prec_en.items():
        writer.add_scalar(f'inference_en/{k}', v)

    return iou_raw.item(), prec_raw, iou_en.item(), prec_en
