import argparse
import os
import warnings
import cv2
import torch
import torch.nn.functional as F
from loguru import logger
from torchvision import transforms
from PIL import Image
import numpy as np
import sng_parser
import utils.config as config
from model import build_segmenter
from utils.dataset import get_token_positive_map_by_parse, tokenize_en
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def get_parser():
    parser = argparse.ArgumentParser(description='Single-image RIS Demo')
    parser.add_argument('--config', type=str, required=True,
                        help='config yaml, 与训练一致')
    parser.add_argument('--model_ckpt', type=str, required=True,
                        help='训练好的权重路径')
    parser.add_argument('--image', type=str, required=True,
                        help='待推理的图片路径')
    parser.add_argument('--sentence', type=str, required=True,
                        help='自然语言描述')
    parser.add_argument('--outdir', type=str, default='outputs/demo',
                        help='directory for masks and overlays')
    parser.add_argument('--gt-mask', type=str, default=None,
                        help='optional ground-truth mask used to report IoU')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='掩码阈值')
    parser.add_argument('--input_size', type=int, default=416,
                        help='模型输入尺寸')
    parser.add_argument('--if_parse', action='store_true', help='caption if need parse')
    return parser
def generate_anno(caption):
    anno ={}
    graph_node, graph_edge = sng_parser.parse(caption)
    anno["graph_node"] = graph_node
    anno["graph_edge"] = graph_edge
    auxi_entity = None
    for node in graph_node:
        if (node["node_id"] != 0) and (node["node_type"] == "Object"):
            auxi_entity = node
            break
    anno["auxi_entity"] = auxi_entity            
        #caption_en decoupling
    anno["utterance"] = caption
    return anno


@logger.catch
def main():
    args = get_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # === 1. 载入配置 ===
    cfg = config.load_cfg_from_cfg_file(args.config)

    # === 2. 构建模型 ===
    model, _ = build_segmenter(cfg)
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Loading checkpoint from {}".format(args.model_ckpt))
    checkpoint = torch.load(args.model_ckpt, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # === 3. 图像预处理 ===
    img_raw = Image.open(args.image).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],
                             std=[0.26862954,0.26130258,0.27577711])
    ])
    img_tensor = tfm(img_raw).unsqueeze(0).cuda()

    # === 4. 文本编码 ===
    sentence = ' '.join(args.sentence.replace(',', ' ,').split())
    text_tensor, char_to_token_map_list = tokenize_en(sentence, 60, True)
    text_tensor = text_tensor.cuda()
    if args.if_parse:
        anno = generate_anno(sentence)
        tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
        other_entity_map, rel_positive_map = get_token_positive_map_by_parse(anno, 60, char_to_token_map_list)
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
    else:
        en_labels = None
    # === 5. 推理 ===
    with torch.no_grad():
        pred = model(img_tensor, text_tensor,
                     en_labels=en_labels, en_labels_raw=None, if_infer=True)
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=(args.input_size, args.input_size),
                             mode='bilinear', align_corners=True)
        pred = pred.squeeze().cpu().numpy()

    # === 6. 生成掩码与可视化 ===
    mask = (pred > args.threshold).astype(np.uint8)
    if args.gt_mask:
        gt_mask = np.array(
            Image.open(args.gt_mask).convert('L').resize(
                (args.input_size, args.input_size)
            )
        )
        gt_mask = (gt_mask > 127).astype(np.uint8)
        intersection = np.logical_and(mask == 1, gt_mask == 1).sum()
        union = np.logical_or(mask == 1, gt_mask == 1).sum()
        iou = intersection / union if union > 0 else 0
        logger.info(f'IoU: {iou:.4f}')

    mask_img = Image.fromarray((mask*255).astype(np.uint8))
    mask_path = os.path.join(args.outdir, 'result_mask.png')
    mask_img.save(mask_path)
    img_np = np.array(img_raw.resize((args.input_size, args.input_size)))
    overlay = img_np.copy()

    # α=0.7 表示绿色更明显，可根据视觉效果调节（0.5~0.8）
    alpha = 0.5 
    green = np.array([255, 0, 0])  # 亮绿色

    overlay[mask == 1] = ((1 - alpha) * overlay[mask == 1] + alpha * green).astype(np.uint8)
    # 半透明叠加
    # img_np = np.array(img_raw.resize((args.input_size, args.input_size)))
    # overlay = img_np.copy()
    # overlay[mask==1] = (0.5*overlay[mask==1] + 0.5*np.array([0,0,255])).astype(np.uint8)
    edges = cv2.Canny(mask*255, 100, 200)  # 边缘检测
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))  # 粗线，5pt
    overlay[edges > 0] = [220, 0, 0]  # 红色轮廓

    overlay_path = os.path.join(args.outdir, 'result_overlay.png')
    Image.fromarray(overlay).save(overlay_path)

    logger.info(f"Saved mask to {mask_path}")
    logger.info(f"Saved overlay to {overlay_path}")

if __name__ == '__main__':
    main()
