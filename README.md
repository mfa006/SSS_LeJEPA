# SSS-LeJEPA: Side-Scan Sonar Linear Probing with LeJEPA Backbones

Evaluation suite for benchmarking frozen vision backbone representations on
**Side-Scan Sonar (SSS) semantic segmentation** using dense linear probing.
Built on top of [LeJEPA](https://arxiv.org/abs/2511.08544) pre-trained models
and standard baselines (ResNet-34, ConvNeXt-Tiny).

> [!NOTE]
> The linear probe is a **single 1×1 convolution** — zero hidden layers, zero
> fine-tuning — trained on only **10% of the dataset (~180 tiles)**. This
> measures the intrinsic quality of the frozen representations.

---

## Project Structure

```
SSS_LeJEPA/
├── sss_probe/              # Core Python package
│   ├── config.py           # Data configs, backbone specs, probe hyperparams
│   ├── dataset.py          # SSS dataset loader & few-shot partitioner
│   ├── backbones.py        # Frozen feature extractors (LeJEPA / ResNet / ConvNeXt)
│   ├── probe.py            # DenseLinearProbe & GlobalLinearProbe
│   ├── trainer.py          # Training, checkpointing, evaluation engine
│   └── metrics.py          # mIoU, mDice, PCA maps, class separation
├── scripts/
│   ├── run_dense_linear_probe.py    # Main training + eval script
│   └── run_feature_separation.py   # Unsupervised PCA / separation benchmark
├── tests/
│   └── test_sss_probe_blueprint.py # Unit tests
├── eval/
│   ├── cache/              # Pre-extracted feature tensors (.pt)
│   ├── checkpoints/        # Trained probe weights (.pt)
│   └── results/            # Metrics JSON/CSV + plots
└── test/                   # SSS dataset (1,800 tiles)
    ├── images/             # .tiff sonar waterfall images
    ├── masks/              # .tiff semantic masks (classes 0–3)
    └── class_labels/       # .csv per-tile label presence
```

---

## Acoustic Semantic Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Seabed Ripples | Periodic bedform texture |
| 1 | Target Anomaly | Bright highlight (object on seabed) |
| 2 | Smooth Sediment | Flat, low-backscatter regions |
| 3 | Acoustic Shadow / Nadir | Dark regions behind targets or nadir |

---

## Supported Backbones

| `<model_name>` | Family | Feature Dim | Grid | Params |
|---|---|---|---|---|
| `lejepa_vits16` | ViT-S/16 (LeJEPA) | 384 | 14×14 | 21.7 M |
| `lejepa_vitb16` | ViT-B/16 (LeJEPA) | 768 | 14×14 | 86.6 M |
| `resnet34` | ResNet-34 | 512 | 7×7 | 21.3 M |
| `convnext_tiny` | ConvNeXt-Tiny | 768 | 7×7 | 27.8 M |

---

## Installation

```bash
cd /path/to/SSS_LeJEPA
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install timm transformers huggingface_hub tqdm matplotlib scikit-learn pillow pandas
```

---

## Verify Installation

```bash
pytest tests/test_sss_probe_blueprint.py -v
```

---

## Feature Separation (Unsupervised, Zero Training)

Visualise PCA feature maps and measure class separation without any probe training:

```bash
python scripts/run_feature_separation.py \
    --image   test/images/<your_tile>.tiff \
    --mask    test/masks/<your_tile>.tiff \
    --models  <model_name> \
    --device  <cuda|cpu> \
    --output  eval/results/feature_separation_comparison.png
```

Compare multiple backbones in one run:

```bash
python scripts/run_feature_separation.py \
    --image   test/images/<your_tile>.tiff \
    --mask    test/masks/<your_tile>.tiff \
    --models  lejepa_vits16 resnet34 convnext_tiny \
    --device  cuda \
    --output  eval/results/feature_separation_comparison.png
```

---

## Dense Linear Probe Training & Evaluation

All settings — data paths, backbone, and hyperparameters — are read from a single YAML file.
Edit [`configs/linear_probe.yaml`](configs/linear_probe.yaml) before running.

```bash
python scripts/run_dense_linear_probe.py --config configs/linear_probe.yaml
```

To use a custom config file:

```bash
python scripts/run_dense_linear_probe.py --config <path/to/your_config.yaml>
```

**Key fields in the YAML:**

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `train_dir` | — | Root of the **training** split (must contain `images/`, `masks/`, `class_labels/`) |
| `data` | `val_dir` | — | Root of the **validation** split |
| `data` | `test_dir` | — | Root of the **test** split |
| `model` | `name` | `lejepa_vits16` | Frozen backbone — see Supported Backbones table |
| `model` | `device` | `cuda` | `cuda` or `cpu` |
| `probe` | `norm_type` | `batchnorm` | `batchnorm` \| `layernorm` \| `l2` \| `none` |
| `training` | `epochs` | `30` | Training epochs |
| `training` | `learning_rate` | `0.001` | AdamW learning rate |
| `training` | `batch_size` | `32` | Batch size |
| `training` | `early_stopping_patience` | `10` | Epochs without val mIoU improvement before stopping |
| `output` | `cache_dir` | `eval/cache` | Pre-extracted feature tensors |
| `output` | `checkpoint_dir` | `eval/checkpoints` | Trained probe `.pt` |
| `output` | `results_dir` | `eval/results` | Metrics JSON/CSV + plots |

> [!TIP]
> Features are cached automatically on first run. Re-running the same backbone
> skips extraction and goes straight to probe training.

**Outputs written to `eval/`:**

```
eval/cache/<model_name>_{train,val,test}_feats.pt   ← feature cache (auto, one-time)
eval/checkpoints/<model_name>_linear_probe.pt       ← best probe weights
eval/results/<model_name>_metrics.json              ← full per-class metrics
eval/results/<model_name>_training_curves.png       ← loss & mIoU curves
eval/results/<model_name>_test_predictions.png      ← qualitative tile comparison
eval/results/benchmark_summary.json                 ← cumulative benchmark table
eval/results/benchmark_summary.csv                  ← same, as CSV
```

---


## Data Layout

Data must be pre-split into three directories. Each must follow the same structure:

```
<split_dir>/
├── images/        ← sonar .tiff tiles
├── masks/         ← semantic .tiff masks
└── class_labels/  ← .csv per-tile label presence  (optional)
```

Set `data.train_dir`, `data.val_dir`, and `data.test_dir` in `configs/linear_probe.yaml` accordingly.

---

## Citation

```bibtex
@misc{balestriero2025lejepaprovablescalableselfsupervised,
      title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
      author={Randall Balestriero and Yann LeCun},
      year={2025},
      eprint={2511.08544},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.08544},
}
```



## Data Layout

Data must be pre-split into three directories. Each must follow the same structure:

```
<split_dir>/
├── images/        ← sonar .tiff tiles
├── masks/         ← semantic .tiff masks
└── class_labels/  ← .csv per-tile label presence  (optional)
```

Set `data.train_dir`, `data.val_dir`, and `data.test_dir` in `configs/linear_probe.yaml` accordingly.

---
 