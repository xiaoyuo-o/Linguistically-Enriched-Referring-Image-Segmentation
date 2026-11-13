# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# Minimal REFER loader
# 只用于 ref_id / seg_id -> COCO image file name
# ============================================================
class MiniREFER:
    def __init__(self, data_root, dataset="refcoco", splitBy="unc", dataset_type="raw"):
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.splitBy = splitBy
        self.dataset_type = dataset_type

        self.data_dir = self.data_root / dataset

        if dataset_type == "en":
            ref_file = self.data_dir / f"refs({splitBy}_en).p"
        else:
            ref_file = self.data_dir / f"refs({splitBy}).p"

        instances_file = self.data_dir / "instances.json"

        if not ref_file.exists():
            raise FileNotFoundError(f"Cannot find ref file: {ref_file}")
        if not instances_file.exists():
            raise FileNotFoundError(f"Cannot find instances file: {instances_file}")

        print(f"[INFO] loading refs from: {ref_file}")
        with open(ref_file, "rb") as f:
            refs = pickle.load(f, fix_imports=True, encoding="latin1")

        print(f"[INFO] loading instances from: {instances_file}")
        with open(instances_file, "r", encoding="utf-8") as f:
            instances = json.load(f)

        self.Refs = {int(ref["ref_id"]): ref for ref in refs}
        self.Imgs = {int(img["id"]): img for img in instances["images"]}

        print(f"[INFO] refs: {len(self.Refs)}, images: {len(self.Imgs)}")

    def get_img_name_by_ref_id(self, ref_id):
        ref_id = int(ref_id)
        ref = self.Refs[ref_id]
        image_id = int(ref["image_id"])
        return self.Imgs[image_id]["file_name"]


# ============================================================
# Basic utils
# ============================================================
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_int(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def parse_int_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [int(v.strip()) for v in str(s).split(",") if v.strip() != ""]


def is_square_number(n):
    n = int(n)
    r = int(round(np.sqrt(n)))
    return r * r == n


def find_image_path(img_dir, image_name):
    img_dir = Path(img_dir)

    p = img_dir / image_name
    if p.exists():
        return p

    matches = list(img_dir.rglob(image_name))
    if len(matches) > 0:
        return matches[0]

    return None


def get_seg_id(data, fallback_stem):
    if isinstance(data, dict):
        for key in ["seg_id", "ref_id"]:
            if key in data:
                return to_int(data[key])

    # 如果 .pt 文件名本身就是 ref_id，也可以兜底
    return int(fallback_stem)


def get_caption(data):
    if not isinstance(data, dict):
        return ""

    for key in ["caption", "sentence", "sent", "text", "expression"]:
        if key in data:
            cap = data[key]
            if isinstance(cap, torch.Tensor):
                return str(cap.detach().cpu().numpy().tolist())
            if isinstance(cap, (list, tuple)):
                if len(cap) > 0:
                    return str(cap[0])
                return ""
            return str(cap)

    return ""


def get_attn_from_data(data, layer_idx=-1):
    """
    支持以下 .pt 结构：
    1. dict: data["attn"]
    2. dict: data["attn_maps"]
    3. 直接保存的 list/tensor
    """
    if isinstance(data, dict):
        attn_obj = None
        for key in ["attn", "attn_maps", "attn_list", "attention", "attention_maps"]:
            if key in data:
                attn_obj = data[key]
                break

        if attn_obj is None:
            raise KeyError(f"No attention key found. Existing keys: {list(data.keys())}")
    else:
        attn_obj = data

    if isinstance(attn_obj, (list, tuple)):
        if len(attn_obj) == 0:
            raise ValueError("Empty attention list.")

        idx = layer_idx if layer_idx >= 0 else len(attn_obj) + layer_idx
        if idx < 0 or idx >= len(attn_obj):
            raise IndexError(f"layer_idx={layer_idx} out of range, len={len(attn_obj)}")

        attn = attn_obj[idx]
    else:
        attn = attn_obj

    return attn


def prepare_attn_matrix(attn):
    """
    把 attention 统一成 [N, L]
    N: image tokens, 例如 676 = 26×26
    L: text tokens, 例如 60

    支持：
    [1, 676, 60]
    [676, 60]
    [1, heads, 676, 60]
    [heads, 676, 60]
    """
    if isinstance(attn, torch.Tensor):
        t = attn.detach().float().cpu()
    elif isinstance(attn, np.ndarray):
        t = torch.from_numpy(attn).float()
    else:
        t = torch.as_tensor(attn).float()

    original_shape = tuple(t.shape)

    if t.ndim == 4:
        # [B, heads, N, L]
        if t.shape[0] == 1:
            t = t[0]          # [heads, N, L]
        else:
            t = t.mean(dim=0)
        t = t.mean(dim=0)     # [N, L]

    elif t.ndim == 3:
        # [B, N, L] 或 [heads, N, L]
        if t.shape[0] == 1:
            t = t[0]
        else:
            t = t.mean(dim=0)

    elif t.ndim == 2:
        pass

    else:
        raise ValueError(f"Unsupported attention shape: {original_shape}")

    if t.ndim != 2:
        raise ValueError(f"Attention cannot be converted to [N,L], got {tuple(t.shape)}")

    # 确保第一维是空间 token
    if not is_square_number(t.shape[0]) and is_square_number(t.shape[1]):
        t = t.transpose(0, 1)

    if not is_square_number(t.shape[0]):
        raise ValueError(
            f"Cannot infer spatial grid from attention shape {tuple(t.shape)}. "
            f"Expected first dim to be square, e.g. 676=26x26."
        )

    return t, original_shape


def normalize_01(x, low=2, high=98):
    """
    百分位归一化，避免极端值影响热力图。
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(x, [low, high])
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32), False, float(lo), float(hi)

    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)
    return x.astype(np.float32), True, float(lo), float(hi)


# ============================================================
# Attention aggregation
# ============================================================
def choose_attention_heat(attn_nl, mode="auto", token_ids=None, topk=3, exclude_tokens=None):
    """
    attn_nl: [N, L]

    mode:
    - auto: 自动选择空间变化最明显的 topk 个 token
    - token: 使用你手动指定的 token_ids
    - max: 对所有有效 token 取 max
    - mean: 对所有有效 token 取 mean，不推荐，只用于对比
    """
    if token_ids is None:
        token_ids = []
    if exclude_tokens is None:
        exclude_tokens = []

    N, L = attn_nl.shape
    device = attn_nl.device

    # 过滤掉完全没有 attention 的 padding token
    col_sum = attn_nl.sum(dim=0)
    valid_ids = torch.where(col_sum > 1e-8)[0].cpu().numpy().tolist()
    if len(valid_ids) == 0:
        valid_ids = list(range(L))

    exclude_set = set(exclude_tokens)
    valid_ids = [i for i in valid_ids if i not in exclude_set]
    if len(valid_ids) == 0:
        valid_ids = list(range(L))

    valid_ids_tensor = torch.tensor(valid_ids, dtype=torch.long, device=device)

    if mode == "token":
        if len(token_ids) == 0:
            raise ValueError("mode='token' requires --token_ids, e.g. --token_ids 5,6")

        selected_ids = [i for i in token_ids if 0 <= i < L]
        if len(selected_ids) == 0:
            raise ValueError(f"Invalid token_ids={token_ids}, attention token length is {L}.")

        ids_tensor = torch.tensor(selected_ids, dtype=torch.long, device=device)
        heat = attn_nl[:, ids_tensor].mean(dim=-1)

    elif mode == "max":
        selected_ids = valid_ids
        heat = attn_nl[:, valid_ids_tensor].max(dim=-1).values

    elif mode == "mean":
        selected_ids = valid_ids
        heat = attn_nl[:, valid_ids_tensor].mean(dim=-1)

    elif mode == "auto":
        # 自动找空间变化最大的 token
        sub = attn_nl[:, valid_ids_tensor]  # [N, valid_L]

        spatial_score = sub.std(dim=0) + 0.5 * (
            sub.max(dim=0).values - sub.min(dim=0).values
        )

        k = min(int(topk), spatial_score.numel())
        if k <= 0:
            k = 1

        top_pos = torch.topk(spatial_score, k=k, largest=True).indices
        ids_tensor = valid_ids_tensor[top_pos]
        selected_ids = ids_tensor.cpu().numpy().tolist()

        # 用 score 加权平均，而不是简单平均
        weights = spatial_score[top_pos]
        weights = weights / (weights.sum() + 1e-8)
        heat = (attn_nl[:, ids_tensor] * weights.unsqueeze(0)).sum(dim=-1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    grid = int(round(np.sqrt(N)))
    heat_2d = heat.reshape(grid, grid).detach().cpu().numpy()

    return heat_2d, selected_ids, valid_ids


# ============================================================
# Visualization
# ============================================================
def make_overlay_bgr(img_bgr, heat_2d, out_size=224, alpha_max=0.60, heat_thr=0.08):
    """
    只在 heat 高的地方叠加颜色，低响应区域不蒙膜。
    """
    if out_size and out_size > 0:
        img_vis = cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    else:
        img_vis = img_bgr.copy()

    H, W = img_vis.shape[:2]

    heat_resized = cv2.resize(heat_2d, (W, H), interpolation=cv2.INTER_CUBIC)
    heat_resized = np.clip(heat_resized, 0, 1).astype(np.float32)

    heat_alpha = heat_resized.copy()
    heat_alpha[heat_alpha < heat_thr] = 0.0

    heat_uint8 = np.uint8(np.clip(heat_resized * 255, 0, 255))
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    alpha = (alpha_max * heat_alpha)[..., None].astype(np.float32)

    overlay = (
        img_vis.astype(np.float32) * (1.0 - alpha)
        + heat_color.astype(np.float32) * alpha
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay, heat_color, heat_uint8


def save_single_token_maps(
    img_bgr,
    attn_nl,
    token_ids,
    save_token_dir,
    out_size,
    alpha_max,
    heat_thr,
    save_aux=False,
):
    """
    保存单独 token 的可视化。
    """
    save_token_dir = Path(save_token_dir)
    save_token_dir.mkdir(parents=True, exist_ok=True)

    N, L = attn_nl.shape
    grid = int(round(np.sqrt(N)))

    saved = []

    for tid in token_ids:
        if tid < 0 or tid >= L:
            continue

        token_heat = attn_nl[:, tid].reshape(grid, grid).detach().cpu().numpy()
        token_heat, _, _, _ = normalize_01(token_heat)

        token_overlay, token_color, token_gray = make_overlay_bgr(
            img_bgr=img_bgr,
            heat_2d=token_heat,
            out_size=out_size,
            alpha_max=alpha_max,
            heat_thr=heat_thr,
        )

        overlay_name = f"token_{tid:02d}_overlay.png"
        cv2.imwrite(str(save_token_dir / overlay_name), token_overlay)
        saved.append(overlay_name)

        if save_aux:
            cv2.imwrite(str(save_token_dir / f"token_{tid:02d}_heat.png"), token_gray)
            cv2.imwrite(str(save_token_dir / f"token_{tid:02d}_heat_color.png"), token_color)

    return saved


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser("Clean attention visualization for CRIS/RIS")

    parser.add_argument(
        "--pt_dir",
        type=str,
        default="/data1/gxy/project/RIS/CRIS.pytorch-master_text1_2cross_att_both_parse/exp_test/vis/refcoco-R/CRIS/ours/val/attn_fixed",
        help="保存 attention .pt 文件的文件夹",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/data1/gxy/dataset/dataset/images/mscoco/images/train2014",
        help="COCO 原图路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data1/gxy/project/RIS/CRIS.pytorch-master_text1_2cross_att_both_parse/exp_test/vis/refcoco-R/CRIS/ours/val/output_attn_clean",
        help="可视化结果保存路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data1/gxy/dataset/dataset",
        help="REFER 数据根目录，里面应该有 refcoco/refs(unc).p 和 refcoco/instances.json",
    )
    parser.add_argument("--dataset", type=str, default="refcoco")
    parser.add_argument("--splitBy", type=str, default="unc")
    parser.add_argument("--dataset_type", type=str, default="raw", choices=["raw", "en"])

    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="选择第几层 attention，默认 -1 表示最后一层",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "token", "max", "mean"],
        help="attention 聚合方式。推荐 auto 或 token。",
    )
    parser.add_argument(
        "--token_ids",
        type=str,
        default="",
        help="mode=token 时使用，例如 --token_ids 5,6",
    )
    parser.add_argument(
        "--exclude_tokens",
        type=str,
        default="",
        help="排除某些 token，例如 --exclude_tokens 0,59",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="mode=auto 时选择空间变化最大的 topk 个 token",
    )

    parser.add_argument(
        "--out_size",
        type=int,
        default=224,
        help="输出图片大小。设为 0 则使用原图大小。",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.60,
        help="最大透明度",
    )
    parser.add_argument(
        "--heat_thr",
        type=float,
        default=0.1,
        help="低于该值的 heatmap 区域不叠加颜色，避免整图蒙膜",
    )

    parser.add_argument(
        "--save_aux",
        action="store_true",
        help="额外保存 heat 灰度图和纯 color heatmap。默认不保存，只保存 overlay。",
    )
    parser.add_argument(
        "--save_token_maps",
        action="store_true",
        help="额外保存所选 token 的单独 overlay。",
    )
    parser.add_argument(
        "--all_token_maps",
        action="store_true",
        help="保存所有有效 token 的单独 overlay。数量会很多。",
    )

    parser.add_argument(
        "--only_seg_ids",
        type=str,
        default="",
        help="只处理某些 seg_id/ref_id，例如 --only_seg_ids 123,456",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多处理多少张。0 表示不限制。",
    )

    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    img_dir = Path(args.img_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    token_ids = parse_int_list(args.token_ids)
    exclude_tokens = parse_int_list(args.exclude_tokens)
    only_seg_ids = set(parse_int_list(args.only_seg_ids))

    refer = MiniREFER(
        data_root=args.data_root,
        dataset=args.dataset,
        splitBy=args.splitBy,
        dataset_type=args.dataset_type,
    )

    pt_files = sorted([p for p in pt_dir.iterdir() if p.suffix == ".pt"])
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in: {pt_dir}")

    rows = []
    processed = 0

    for pt_path in tqdm(pt_files, desc="Visualizing attention"):
        if args.max_samples > 0 and processed >= args.max_samples:
            break

        try:
            data = safe_torch_load(pt_path)

            seg_id = get_seg_id(data, pt_path.stem)
            if len(only_seg_ids) > 0 and seg_id not in only_seg_ids:
                continue

            caption = get_caption(data)

            image_name = refer.get_img_name_by_ref_id(seg_id)
            img_path = find_image_path(img_dir, image_name)

            if img_path is None:
                print(f"[WARN] missing image for seg_id={seg_id}: {image_name}")
                continue

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[WARN] cv2 failed to read image: {img_path}")
                continue

            attn = get_attn_from_data(data, layer_idx=args.layer)
            attn_nl, original_shape = prepare_attn_matrix(attn)

            heat_2d_raw, selected_ids, valid_ids = choose_attention_heat(
                attn_nl,
                mode=args.mode,
                token_ids=token_ids,
                topk=args.topk,
                exclude_tokens=exclude_tokens,
            )

            heat_2d, ok_norm, norm_lo, norm_hi = normalize_01(heat_2d_raw)

            overlay, heat_color, heat_gray = make_overlay_bgr(
                img_bgr=img_bgr,
                heat_2d=heat_2d,
                out_size=args.out_size,
                alpha_max=args.alpha_max,
                heat_thr=args.heat_thr,
            )

            base = f"{seg_id}_layer{args.layer}_{args.mode}"

            # 默认只保存这一张：干净图，没有任何字
            overlay_name = f"{base}_overlay.png"
            overlay_path = save_dir / overlay_name
            cv2.imwrite(str(overlay_path), overlay)

            heat_name = ""
            heat_color_name = ""

            if args.save_aux:
                heat_name = f"{base}_heat.png"
                heat_color_name = f"{base}_heat_color.png"

                cv2.imwrite(str(save_dir / heat_name), heat_gray)
                cv2.imwrite(str(save_dir / heat_color_name), heat_color)

            saved_token_files = []

            if args.save_token_maps:
                if args.all_token_maps:
                    token_save_ids = valid_ids
                else:
                    token_save_ids = selected_ids

                token_dir = save_dir / f"{base}_tokens"

                saved_token_files = save_single_token_maps(
                    img_bgr=img_bgr,
                    attn_nl=attn_nl,
                    token_ids=token_save_ids,
                    save_token_dir=token_dir,
                    out_size=args.out_size,
                    alpha_max=args.alpha_max,
                    heat_thr=args.heat_thr,
                    save_aux=args.save_aux,
                )

            rows.append(
                {
                    "output_image": overlay_name,
                    "pt_file": pt_path.name,
                    "seg_id": seg_id,
                    "image_name": image_name,
                    "caption": caption,
                    "mode": args.mode,
                    "layer": args.layer,
                    "selected_token_ids": ",".join(map(str, selected_ids)),
                    "valid_token_count": len(valid_ids),
                    "attn_original_shape": str(original_shape),
                    "attn_matrix_shape": str(tuple(attn_nl.shape)),
                    "norm_ok": ok_norm,
                    "norm_low": norm_lo,
                    "norm_high": norm_hi,
                    "aux_heat": heat_name,
                    "aux_heat_color": heat_color_name,
                    "token_maps_dir": f"{base}_tokens" if args.save_token_maps else "",
                    "saved_token_files": ";".join(saved_token_files),
                }
            )

            processed += 1
            print(f"[SAVE] {overlay_path}")

        except Exception as e:
            print(f"[ERROR] {pt_path.name}: {repr(e)}")
            rows.append(
                {
                    "output_image": "",
                    "pt_file": pt_path.name,
                    "seg_id": "",
                    "image_name": "",
                    "caption": "",
                    "mode": args.mode,
                    "layer": args.layer,
                    "selected_token_ids": "",
                    "valid_token_count": "",
                    "attn_original_shape": "",
                    "attn_matrix_shape": "",
                    "norm_ok": "",
                    "norm_low": "",
                    "norm_high": "",
                    "aux_heat": "",
                    "aux_heat_color": "",
                    "token_maps_dir": "",
                    "saved_token_files": "",
                    "error": repr(e),
                }
            )

    # ========================================================
    # 另存“文档”：告诉你每张图对应哪句话
    # ========================================================
    csv_path = save_dir / "attn_caption_index.csv"
    txt_path = save_dir / "attn_caption_index.txt"

    if len(rows) > 0:
        fieldnames = [
            "output_image",
            "pt_file",
            "seg_id",
            "image_name",
            "caption",
            "mode",
            "layer",
            "selected_token_ids",
            "valid_token_count",
            "attn_original_shape",
            "attn_matrix_shape",
            "norm_ok",
            "norm_low",
            "norm_high",
            "aux_heat",
            "aux_heat_color",
            "token_maps_dir",
            "saved_token_files",
            "error",
        ]

        # CSV：Excel 可以直接打开
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        # TXT：方便快速查看
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                f.write(f"[{i}]\n")
                f.write(f"output_image: {row.get('output_image', '')}\n")
                f.write(f"seg_id: {row.get('seg_id', '')}\n")
                f.write(f"image_name: {row.get('image_name', '')}\n")
                f.write(f"caption: {row.get('caption', '')}\n")
                f.write(f"mode: {row.get('mode', '')}\n")
                f.write(f"layer: {row.get('layer', '')}\n")
                f.write(f"selected_token_ids: {row.get('selected_token_ids', '')}\n")
                f.write(f"token_maps_dir: {row.get('token_maps_dir', '')}\n")
                f.write("\n")

    print(f"[DONE] clean overlays saved to: {save_dir}")
    print(f"[DONE] caption index csv saved to: {csv_path}")
    print(f"[DONE] caption index txt saved to: {txt_path}")


if __name__ == "__main__":
    main()