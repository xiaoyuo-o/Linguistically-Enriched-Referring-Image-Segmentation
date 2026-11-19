import argparse
import os
import pickle

import clip
import lmdb
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from refer import REFER


class ImageDataset(Dataset):
    def __init__(self, image_ids, refer, image_dir, preprocess):
        self.image_ids = image_ids
        self.refer = refer
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        name = self.refer.Imgs[image_id]["file_name"]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        return image_id, self.preprocess(image)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()
    refer = REFER(args.data_root, args.dataset, args.split_by,
                  dataset_type=args.expression_type)
    ref_ids = refer.getRefIds(split=args.split)
    image_ids = refer.getImgIds(ref_ids)
    loader = DataLoader(ImageDataset(image_ids, refer, args.image_dir, preprocess),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device == "cuda")
    image_features, image_keys = [], []
    with torch.no_grad():
        for ids, images in tqdm(loader, desc="Encoding images"):
            features = model.encode_image(images.to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            image_features.append(features.cpu())
            image_keys.extend(ids.tolist())
    image_features = torch.cat(image_features).to(device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    env = lmdb.open(args.output, map_size=args.map_size, subdir=True)
    keys = []
    with env.begin(write=True) as txn, torch.no_grad():
        for ref_id in tqdm(ref_ids, desc="Building candidate pools"):
            ref = refer.loadRefs(ref_id)[0]
            pools = {}
            for sentence in ref["sentences"]:
                tokens = clip.tokenize([sentence["sent"]]).to(device)
                text = model.encode_text(tokens)
                text = text / text.norm(dim=-1, keepdim=True)
                scores = (text @ image_features.T).squeeze(0).cpu().tolist()
                ranked = sorted(zip(image_keys, scores),
                                key=lambda item: item[1], reverse=True)
                pools[sentence["sent_id"]] = ranked[:args.top_k]
            txn.put(str(ref_id).encode("utf-8"), pickle.dumps(pools))
            keys.append(ref_id)
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))
    env.sync()
    env.close()
    print("Wrote", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--dataset", default="refcoco", choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split-by", default="unc")
    parser.add_argument("--split", default="train")
    parser.add_argument("--expression-type", default="raw", choices=["raw", "en"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--map-size", type=int, default=1 << 40)
    main(parser.parse_args())
