# DRERIS-CRIS

Official CRIS-based implementation for **Rethinking Language Bias in Referring Image Segmentation via Linguistically Enriched Data Reconstruction and Semantic Alignment** (IEEE Transactions on Image Processing, DOI: [10.1109/TIP.2026.3748959](https://doi.org/10.1109/TIP.2026.3748959)).

This release contains the CRIS experiments used to study original and linguistically enriched referring expressions. It includes the SAMA sampling pipeline, DTVI interaction module, dual-branch training, and independent evaluation of either branch.

## Code status

This repository is being prepared for public release. The RefCOCO checkpoint and reconstructed datasets are not committed to Git. Download links will be added before publication.

## Main components

- `utils/dataset.py`: paired original/enhanced data loading and SAMA candidate retrieval.
- `model/segmenter.py`: CRIS model, dual-branch forward path, parser-guided features, and DTVI vision-guided text interaction.
- `engine/engine.py`: dual-branch optimization and validation.
- `train.py`: distributed training entry point.
- `evaluate.py`: reproducible single-branch evaluation for original or reconstructed expressions.
- `config/release/refcoco_r101.yaml`: public RefCOCO R101 configuration.

The implementation is derived from the official [CRIS](https://github.com/DerrickWang005/CRIS.pytorch) codebase. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Installation

A CUDA-enabled PyTorch environment is required. The experiments were developed with Python 3.8.

```bash
conda create -n dreris python=3.8 -y
conda activate dreris
pip install torch torchvision
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set `NLTK_DATA` only when the NLTK resources are stored outside its standard search locations.

## Data layout

Update `config/release/refcoco_r101.yaml` if your files use a different layout.

```text
data/
├── refer/
└── refcoco/
    ├── raw/
    │   ├── train.lmdb/
    │   └── val.lmdb/
    ├── enhanced/
    │   ├── train.lmdb/
    │   └── val.lmdb/
    ├── masks/
    └── similarity/
        ├── t2i_similarity.lmdb/
        ├── t2i_similarity_en.lmdb/
        └── i2i_similarity.lmdb/
weights/
├── RN101.pt
└── cris_r101_refcoco.pth
```

The original and enhanced LMDBs must preserve sample correspondence. The three similarity databases contain the SAMA retrieval candidates used during training.

## Training

`train.py` launches one worker per visible GPU. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --config config/release/refcoco_r101.yaml
```

Override existing configuration fields with `--opts`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config config/release/refcoco_r101.yaml \
  --opts TRAIN.batch_size 8 TRAIN.batch_size_val 8
```

## Evaluation

Place the released checkpoint at `weights/cris_r101_refcoco.pth`.

Original-expression branch:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config config/release/refcoco_r101.yaml \
  --checkpoint weights/cris_r101_refcoco.pth \
  --branch raw
```

Reconstructed-expression branch:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config config/release/refcoco_r101.yaml \
  --checkpoint weights/cris_r101_refcoco.pth \
  --branch reconstructed
```

The evaluator reports mIoU, oIoU, and precision at IoU thresholds 0.5 through 0.9. The verified RefCOCO validation mIoU values for the release checkpoint are 72.45 (original) and 58.95 (reconstructed). Its SHA-256 is:

```text
cfd7edf8f9bd1fb8d2988f743a184e603af9b02d6be2caa88400b05f3aa61947
```

See [weights/README.md](weights/README.md) for checkpoint details.

## Citation

```bibtex
@article{zhang2026rethinking,
  title={Rethinking Language Bias in Referring Image Segmentation via Linguistically Enriched Data Reconstruction and Semantic Alignment},
  author={Zhang, Lu and Guo, Xiaoyu and Zhuge, Yunzhi and Zhang, Wenbo and Hu, Ping and Lu, Huchuan and He, You},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  doi={10.1109/TIP.2026.3748959}
}
```

Please also cite CRIS when using this implementation.

## License

The code is released under the MIT License. Third-party components remain subject to their original notices and licenses.
