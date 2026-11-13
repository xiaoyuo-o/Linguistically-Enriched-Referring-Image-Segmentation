import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.config as config
from model import build_segmenter
from utils.dataset import (
    RefDataset,
    generate_anno,
    get_token_positive_map_by_parse,
    tokenize_en,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CRIS checkpoint on one caption branch."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--branch", choices=("raw", "reconstructed"), default="raw"
    )
    parser.add_argument("--lmdb", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_parser_labels(caption, token_length):
    annotation = generate_anno(caption)
    tokens, char_to_token = tokenize_en(caption, token_length, True)
    parsed = get_token_positive_map_by_parse(
        annotation, token_length, char_to_token
    )
    (
        tokens_positive,
        positive_map,
        modify_positive_map,
        pron_positive_map,
        other_entity_map,
        rel_positive_map,
    ) = parsed
    labels = {
        "tokens_positive": tokens_positive.astype(np.int64),
        "positive_map": torch.from_numpy(positive_map).unsqueeze(0),
        "modify_positive_map": torch.from_numpy(modify_positive_map).unsqueeze(0),
        "pron_positive_map": torch.from_numpy(pron_positive_map).unsqueeze(0),
        "other_entity_map": torch.from_numpy(other_entity_map).unsqueeze(0),
        "rel_positive_map": torch.from_numpy(rel_positive_map).unsqueeze(0),
    }
    return tokens, labels


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_has_module = next(iter(state_dict)).startswith("module.")
    model_has_module = next(iter(model.state_dict())).startswith("module.")
    if state_has_module and not model_has_module:
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    elif model_has_module and not state_has_module:
        state_dict = {
            f"module.{key}": value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    return checkpoint

@torch.no_grad()
def evaluate(loader, model, cfg, device, threshold):
    sample_ious = []
    intersections = 0
    unions = 0
    model.eval()

    for image, mask, params, is_decouple in tqdm(loader, desc="Evaluation"):
        image = image.to(device, non_blocking=True)
        for sentence in params["sents"]:
            caption = " ".join(sentence[0].replace(",", " ,").split())
            tokens, labels = build_parser_labels(caption, cfg.word_len)
            tokens = tokens.to(device, non_blocking=True)
            use_dtvi = (
                bool(is_decouple[0])
                if torch.is_tensor(is_decouple)
                else bool(is_decouple)
            )
            prediction = model(
                image,
                tokens,
                en_labels=labels if use_dtvi else None,
                en_labels_raw=None,
                if_infer=True,
            )
            prediction = torch.sigmoid(prediction)
            if prediction.shape[-2:] != image.shape[-2:]:
                prediction = F.interpolate(
                    prediction,
                    size=image.shape[-2:],
                    mode="bicubic",
                    align_corners=True,
                )

            pred_mask = prediction.squeeze().cpu().numpy() > threshold
            target_mask = mask.squeeze().cpu().numpy().astype(bool)
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            sample_ious.append(intersection / (union + 1e-6))
            intersections += int(intersection)
            unions += int(union)

    sample_ious = np.asarray(sample_ious)
    metrics = {
        "mIoU": float(sample_ious.mean()),
        "oIoU": float(intersections / (unions + 1e-6)),
    }
    for value in np.arange(0.5, 1.0, 0.1):
        metrics[f"Pr@{int(round(value * 100))}"] = float(
            (sample_ious > value).mean()
        )
    return metrics

def main():
    cli = parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    lmdb_path = cli.lmdb
    if lmdb_path is None:
        lmdb_path = (
            cfg.val_lmdb if cli.branch == "raw" else cfg.new_val_lmdb
        )
    split = cli.split or cfg.val_split

    device = torch.device(cli.device)
    if device.type != "cuda":
        raise ValueError("This CRIS implementation currently requires CUDA.")
    torch.cuda.set_device(device)

    dataset = RefDataset(
        lmdb_dir=lmdb_path,
        new_lmdb_dir=lmdb_path,
        mask_dir=cfg.mask_root,
        dataset=cfg.dataset,
        split=split,
        mode="test",
        args=cfg,
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        word_en_length=cfg.word_en_length,
        is_decouple=cfg.is_decouple,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.workers,
        pin_memory=True,
    )

    model, _ = build_segmenter(cfg)
    model = torch.nn.DataParallel(model.to(device))
    checkpoint = load_checkpoint(model, Path(cli.checkpoint))
    metrics = evaluate(loader, model, cfg, device, cli.threshold)

    print(f"checkpoint_epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"branch: {cli.branch}")
    print(f"lmdb: {lmdb_path}")
    for name, value in metrics.items():
        print(f"{name}: {100.0 * value:.2f}")


if __name__ == "__main__":
    main()
