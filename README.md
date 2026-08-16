# Crop Disease Detection from Leaf Images

Seven CNN architectures compared on the full **PlantVillage** dataset — 54,282 labelled
leaf images, 14 crop species, 38 disease classes — trained under identical settings on a
leakage-free split.

## Results

All models: 20 epochs, Adam lr=1e-4, batch 32, 224×224, class-weighted loss, seed 42, on
the identical split. Precision/Recall/F1 are macro-averaged over the 38 classes.

| Model | Parameters | Test accuracy | Test macro-F1 | Train time |
|---|---:|---:|---:|---:|
| **DenseNet121** | 6,992,806 | **99.79 %** | **99.62 %** | 84 min |
| ResNet18 | 11,196,006 | 99.66 % | 99.53 % | 59 min |
| GoogLeNet (Inception v1) | 5,638,854 | 99.64 % | 99.50 % | 57 min |
| VGG16 | 134,416,230 | 99.16 % | 98.94 % | 137 min |
| SqueezeNet 1.1 | 741,990 | 98.68 % | 98.27 % | 54 min |
| AlexNet | 57,159,526 | 98.32 % | 98.06 % | 63 min |
| LeNet-5 *(from scratch)* | 64,386 | 73.17 % | 67.14 % | 44 min |

Trained on an NVIDIA RTX 3060. Parameter counts are measured with the **38-class** head,
not the published 1000-class ImageNet figures.

### What the numbers show

**Model size does not predict accuracy.** Ranked by parameters the order is VGG16 (134 M)
→ AlexNet (57 M) → ResNet18 (11 M) → DenseNet (7 M) → GoogLeNet (5.6 M) → SqueezeNet
(0.74 M); ranked by accuracy it is close to inverted. The two largest models finish 4th
and 6th, while DenseNet beats VGG16 by 0.68 macro-F1 points with **19× fewer parameters**.
Dense connections, residual connections and inception modules reuse features; VGG16 and
AlexNet mostly stack capacity.

**Pretraining is worth about 32 points.** LeNet-5 is the only model here without ImageNet
weights — it trains from scratch and lands at 67.14 % macro-F1 against 98–99.6 % for the
rest. Its gap measures the absence of transfer learning as much as the 1998 architecture,
so it is not a like-for-like architectural comparison.

**The dataset is close to saturated.** Six of seven models sit within 1.6 macro-F1 points
of each other. PlantVillage is uniform studio photography of single detached leaves, so
the interesting differences are in *cost*, not accuracy — SqueezeNet reaches within 1.4
points of the best at 1/180th of VGG16's size.

## The split, and why it is rebuilt

Splitting PlantVillage naively leaks: the same leaf re-ingested under a new UUID, or
photographed again seconds later, lands in both train and test. Measured on this dataset,
a naive stratified split contaminates **1,228 of 16,293 evaluation images (7.5 %)**, worth
roughly a point of free, fake accuracy.

Part A of the pipeline therefore:

1. hashes every image (SHA-256 for exact duplicates, dHash for near-duplicates),
2. groups related images with union-find — near-identical shots, the same physical leaf
   across days, and the same capture re-ingested under a new name,
3. assigns whole **groups** to one split, keeping each class near 70/15/15,
4. re-runs the leakage scan as a pass/fail gate.

After the repair all four leakage counts are **0**. Result: 37,997 train / 8,149 val /
8,136 test.

Use the `split` column of `manifest_clean.csv` as-is. If you re-split, split on
`group_key`, or you reintroduce exactly the leakage this removes.

## Repository layout

| Path | What it is |
|---|---|
| `crop_disease_full_pipeline.ipynb` | Part A (dataset repair + split) then trains `MODEL_KEY` |
| `crop_disease_<model>.ipynb` | one notebook per architecture, same pipeline |
| `lite_disease_net_v4_38.ipynb` | custom dual-branch CBAM architecture |
| `crop-disease-38/` | results JSON, per-class CSVs, curves, confusion matrices |
| `metrics_tables_38.docx` | Training / Validation / Testing tables for the report |
| `my_photos/` | inference on arbitrary photos |

Model weights (`*.pth`), the manifests and the datasets are **not** in the repo — they
exceed GitHub's file limits. Everything needed to regenerate them is here.

## Reproducing

```bash
# 1. get the dataset (14 crops, 38 classes, 54,305 images)
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/spMohanty/PlantVillage-Dataset.git plantvillage_full
cd plantvillage_full && git sparse-checkout set raw/color && cd ..

# 2. open any crop_disease_*.ipynb and Run All
#    DATA_PATH already points at plantvillage_full/raw/color
```

Needs a GPU (about 1 hour per model) plus `torch`, `torchvision`, `scikit-learn`,
`pandas`, `matplotlib`, `seaborn`, `torchmetrics`, `tqdm` — the config cell pip-installs
anything missing.

To train a different architecture, change `MODEL_KEY` in the config cell. Valid keys are
listed in the Part C registry (`resnet50`, `mobilenet_v2`, `efficientnet_b0`, …). Results
files are named per model, so runs never overwrite each other.

## Notes on the metrics

- **Headline is macro-F1, not accuracy.** Class sizes range from 152 to 5,507 images
  (36× imbalance), so accuracy alone flatters a model that neglects rare classes.
- **Per-class recall uses Wilson score intervals.** The usual normal approximation
  collapses to ±0 when recall is exactly 1.0, claiming certainty that 33/33 does not
  support; Wilson reports `[0.896, 1.000]` instead.
- **Small classes carry wide intervals.** `Potato___healthy` has 26 test images, so its
  recall interval spans roughly 20 points. Differences smaller than that are noise.
- **Lab data, not field data.** Expect a substantial drop on real field photography;
  a model trained here can be confidently wrong on a crop it has never seen.
