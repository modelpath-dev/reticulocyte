# AI-Assisted Reticulocyte Counting Pipeline

Automated detection and classification of **reticulocytes** (immature red blood cells) from blood smear microscopy images using classical computer vision and deep learning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Data Flow](#architecture--data-flow)
- [Project Structure](#project-structure)
- [Module Descriptions](#module-descriptions)
- [Datasets](#datasets)
- [Trained Models](#trained-models)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Limitations](#limitations)
- [Recommended Next Steps](#recommended-next-steps)

---

## Overview

Reticulocyte counting is a critical hematology test used to assess bone marrow function. Manual counting under a microscope is slow, subjective, and has poor reproducibility. This project builds an end-to-end pipeline that:

1. **Detects** all red blood cells (RBCs) in a blood smear image
2. **Classifies** each detected cell as a mature RBC or a reticulocyte
3. **Reports** the reticulocyte percentage with visualizations

Two inference modes are supported:
- **Classical**: Hough circle detection + SVM classifier (no GPU required)
- **Deep Learning**: YOLOv8s detector + EfficientNet-B0 classifier (GPU recommended)

---

## Architecture & Data Flow

### End-to-End Pipeline Flowchart

```
┌──────────────────────────┐
│   Input: Blood Smear     │
│    Microscopy Image      │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│      PREPROCESSING       │
│  - Macenko stain norm    │
│  - Background removal    │
│  - Tiling (if >1024px)   │
└────────────┬─────────────┘
             ▼
┌──────────────────────────────────────────────┐
│          STAGE 1: RBC DETECTION              │
│                                              │
│  ┌──────────────────┐  ┌──────────────────┐  │
│  │    Classical      │  │      Deep        │  │
│  │  Hough Circles    │  │   YOLOv8s        │  │
│  │  + Watershed      │  │  (BCCD trained)  │  │
│  │  + NMS            │  │  conf=0.35       │  │
│  └────────┬─────────┘  └────────┬─────────┘  │
│           └──────────┬──────────┘             │
└──────────────────────┬───────────────────────┘
                       ▼
              RBC Bounding Boxes
                       ▼
┌──────────────────────────────────────────────┐
│        CROP EXTRACTION (64x64 px)            │
│  - Pad & resize each detected cell           │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│    STAGE 2: RETICULOCYTE CLASSIFICATION      │
│                                              │
│  ┌──────────────────┐  ┌──────────────────┐  │
│  │    Classical      │  │      Deep        │  │
│  │  Color Histograms │  │  EfficientNet-B0 │  │
│  │  (HSV/LAB/LUV)   │  │  (ImageNet       │  │
│  │  + Gabor Texture  │  │   fine-tuned)    │  │
│  │  → SVM (RBF)     │  │                  │  │
│  └────────┬─────────┘  └────────┬─────────┘  │
│           └──────────┬──────────┘             │
└──────────────────────┬───────────────────────┘
                       ▼
          Per-cell: Mature vs Reticulocyte
                       ▼
┌──────────────────────────────────────────────┐
│            POSTPROCESSING & OUTPUT           │
│  - Aggregate counts → reticulocyte %         │
│  - Color-coded overlay (red/green/orange)    │
│  - Cell grid visualization                   │
│  - JSON report with per-cell predictions     │
└──────────────────────────────────────────────┘
```

### Training Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                                                             │
│  ┌─────────────────────────────────────────┐                │
│  │   DETECTOR TRAINING (train_detector.py) │                │
│  │                                         │                │
│  │   BCCD Raw (364 images, Pascal VOC)     │                │
│  │          │                              │                │
│  │          ▼                              │                │
│  │   preprocess.py: convert_bccd_to_yolo() │                │
│  │          │                              │                │
│  │          ▼                              │                │
│  │   BCCD YOLO (train/val/test 70/15/15)   │                │
│  │          │                              │                │
│  │          ▼                              │                │
│  │   YOLOv8s fine-tune (50 epochs)         │                │
│  │          │                              │                │
│  │          ▼                              │                │
│  │   models/rbc_detector/weights/best.pt   │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  ┌──────────────────────────────────────────────┐           │
│  │   CLASSIFIER TRAINING (train_classifier.py)  │           │
│  │                                              │           │
│  │   BCCD Crops (mature RBC images)             │           │
│  │          │                                   │           │
│  │          ▼                                   │           │
│  │   preprocess.py: generate_synthetic_retics() │           │
│  │   (add simulated blue granules)              │           │
│  │          │                                   │           │
│  │          ▼                                   │           │
│  │   Synthetic Dataset                          │           │
│  │   (2000 retic + 2000 mature)                 │           │
│  │          │                                   │           │
│  │          ▼                                   │           │
│  │   EfficientNet-B0 fine-tune (30 epochs)      │           │
│  │          │                                   │           │
│  │          ▼                                   │           │
│  │   models/retic_classifier/                   │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
reticulocyte/
├── 01_literature_review.md            # Literature analysis (15 papers)
├── 02_dataset_inventory.csv           # 11 datasets cataloged with licensing
├── 03_reproducibility_notes.md        # Setup instructions & data acquisition
├── 04_model_baseline/                 # Core pipeline code
│   ├── config.py                      # Central configuration
│   ├── preprocess.py                  # Stain normalization, tiling, format conversion
│   ├── detect_rbc.py                  # Hough circles + YOLOv8 detectors
│   ├── classify_reticulocyte.py       # SVM + EfficientNet classifiers
│   ├── pipeline.py                    # End-to-end orchestration
│   ├── overlay.py                     # Visualization (bounding boxes, grids)
│   ├── train_detector.py              # YOLOv8 training script
│   ├── train_classifier.py            # EfficientNet-B0 training script
│   ├── classical_baseline.py          # Standalone Hough+SVM baseline
│   └── evaluate.py                    # Metrics & reporting
├── data/
│   ├── bccd_raw/                      # 364 Wright-Giemsa stained images
│   ├── bccd_yolo/                     # YOLO-format BCCD (70/15/15 split)
│   ├── bccd_crops/                    # Extracted cell crops from BCCD
│   ├── synthetic_retic/               # 2000 synthetic reticulocytes + 2000 mature
│   ├── feline_crops/                  # 800 feline reticulocyte images
│   ├── feline_raw/                    # Raw feline dataset with annotations
│   ├── cenparmi/                      # (Empty - requires consent form)
│   └── elsafty_rbc/                   # (Empty - available via Figshare)
├── models/
│   ├── rbc_detector/                  # YOLOv8s fine-tuned weights
│   │   ├── weights/best.pt
│   │   ├── results.csv
│   │   └── args.yaml
│   └── retic_classifier/             # EfficientNet-B0 training history
│       └── training_history.json
├── output/                            # Pipeline output directory
├── notebooks/                         # Jupyter notebooks
├── extract_bccd_crops.py              # Extract cell crops from BCCD annotations
├── extract_feline_crops.py            # Extract feline reticulocyte crops
├── run_pipeline.sh                    # Bash helper script
├── requirements.txt                   # Python dependencies
└── 06_demo.ipynb                      # Interactive demo notebook
```

---

## Module Descriptions

### `pipeline.py` — Orchestration Engine

The central module that ties everything together. The `ReticulocytePipeline` class:

1. Loads the chosen detector and classifier based on mode (`deep` or `classical`)
2. Preprocesses input images (stain normalization, tiling)
3. Runs Stage 1 detection to get RBC bounding boxes
4. Crops each detected cell to 64x64
5. Runs Stage 2 classification on all crops (batch inference)
6. Aggregates results and produces outputs (overlay, grid, JSON report)

```
process_image() flowchart:

  load image → preprocess → detect RBCs → crop cells
       → classify batch → aggregate counts → save outputs
```

### `detect_rbc.py` — RBC Detection

Two detector implementations:

| Feature | ClassicalDetector | DeepDetector |
|---------|-------------------|--------------|
| Method | Hough circles + watershed | YOLOv8s |
| Preprocessing | CLAHE contrast enhancement | Built-in |
| Post-processing | NMS (custom) | NMS (built-in) |
| GPU required | No | Recommended |
| Speed | ~1-3s/image | ~0.3s/image (GPU) |

**ClassicalDetector flowchart:**
```
Input image
    ▼
Convert to grayscale
    ▼
Apply CLAHE contrast enhancement
    ▼
Gaussian blur (reduce noise)
    ▼
Hough circle transform
    ▼
Watershed segmentation (split touching cells)
    ▼
Non-maximum suppression
    ▼
Bounding box list
```

**DeepDetector flowchart:**
```
Input image
    ▼
YOLOv8s inference (conf=0.35, iou=0.4)
    ▼
Filter for RBC class only
    ▼
Bounding box list with confidence scores
```

### `classify_reticulocyte.py` — Cell Classification

Two classifier implementations:

**Classical classifier flowchart:**
```
64x64 cell crop
    ▼
┌───────────────────────────────────┐
│  Feature Extraction               │
│  ├─ HSV histogram (32 bins × 3)  │
│  ├─ LAB histogram (32 bins × 3)  │
│  ├─ LUV histogram (32 bins × 3)  │
│  └─ Gabor filters (on Cr channel)│
│     (5 frequencies × 8 angles)   │
└───────────────┬───────────────────┘
                ▼
        Concatenated feature vector
                ▼
        SVM (RBF kernel, C=10)
                ▼
        mature / reticulocyte
```

**Deep classifier flowchart:**
```
64x64 cell crop
    ▼
Resize + ImageNet normalization
    ▼
EfficientNet-B0 backbone
    ▼
Dropout (0.3)
    ▼
FC layer → 2 classes
    ▼
Softmax → probability
    ▼
mature / reticulocyte (threshold=0.5)
```

### `preprocess.py` — Data Preparation

Key functions:

| Function | Purpose |
|----------|---------|
| `macenko_normalize()` | Stain normalization via optical density SVD |
| `tile_image()` | Splits large images into 1024x1024 tiles with 64px overlap |
| `reassemble_tiles()` | Merges tile predictions with NMS |
| `convert_bccd_to_yolo()` | Pascal VOC XML → YOLO format, auto-splits train/val/test |
| `add_synthetic_reticulum()` | Adds simulated blue granules to RBC crops |
| `generate_synthetic_reticulocytes()` | Creates balanced synthetic training dataset |

**Synthetic reticulocyte generation flowchart:**
```
Mature RBC crop (from BCCD)
    ▼
Random number of blue granules (3-15)
    ▼
Random positions within cell mask
    ▼
Draw semi-transparent blue dots
    ▼
Apply Gaussian blur for realism
    ▼
Synthetic reticulocyte image
```

### `overlay.py` — Visualization

Generates three types of output:
- **Detection overlay**: Bounding boxes on original image
  - Green = mature RBC
  - Red = reticulocyte
  - Orange = uncertain (score 0.4-0.6)
- **Summary panel**: Text overlay with total RBCs, reticulocyte count, and retic %
- **Cell grid**: All detected cells arranged in a grid for manual review

### `evaluate.py` — Metrics & Reporting

```
Ground truth + Predictions
    ▼
┌──────────────────────────────┐
│  Detection Metrics           │
│  ├─ IoU matching             │
│  ├─ TP / FP / FN counts     │
│  ├─ Precision & Recall       │
│  └─ F1 Score                 │
├──────────────────────────────┤
│  Counting Metrics            │
│  ├─ MAE (Mean Absolute Err)  │
│  ├─ RMSE                     │
│  └─ Pearson Correlation      │
├──────────────────────────────┤
│  Failure Analysis            │
│  ├─ Bias detection           │
│  └─ Error distribution       │
├──────────────────────────────┤
│  Output                      │
│  ├─ Scatter & histogram plots│
│  └─ Markdown report          │
└──────────────────────────────┘
```

### `config.py` — Configuration

Single source of truth for all parameters:

| Section | Key Parameters |
|---------|---------------|
| `DETECTOR` | model=`yolov8s`, conf_threshold=`0.35`, iou_threshold=`0.4` |
| `CLASSIFIER` | model=`efficientnet_b0`, input_size=`64x64`, lr=`1e-4`, epochs=`30` |
| `CLASSICAL` | SVM kernel=`rbf`, C=`10`, histogram_bins=`32` |
| `PREPROCESSING` | stain_norm=`macenko`, bg_threshold=`230`, tile_size=`1024`, overlap=`64` |
| `AUGMENTATION` | color_jitter, rotation, flips, CLAHE, Gaussian noise |

---

## Datasets

| Dataset | Size | Status | License | Purpose |
|---------|------|--------|---------|---------|
| BCCD | 364 images | Downloaded | MIT | RBC detection training |
| BCCD Crops | 61 MB | Extracted | MIT | Cell crop reference |
| Synthetic Retic | 32 MB (4000 crops) | Generated | N/A | Classifier training |
| Feline Crops | 38 MB (800 images) | Downloaded | Research | Cross-domain baseline |
| CENPARMI | 2461 images | Not available | Consent Form | Real reticulocyte data |
| Elsafty RBCs | 1M+ crops | Not available | CC BY | RBC morphology reference |

---

## Trained Models

### RBC Detector (YOLOv8s)

- **Location**: `models/rbc_detector/weights/best.pt`
- **Dataset**: BCCD (364 images, YOLO format)
- **Training**: 50 epochs, batch size 16
- **Final metrics (epoch 50)**:
  - Precision: **80.1%**
  - Recall: **77.6%**
  - mAP@0.5: **87.5%**
  - mAP@0.5:0.95: **59.7%**

### Reticulocyte Classifier (EfficientNet-B0)

- **Location**: `models/retic_classifier/`
- **Dataset**: Synthetic (2000 reticulocytes + 2000 mature RBCs)
- **Training**: 30 epochs, batch size 64
- **Final metrics (epoch 30)**:
  - Val Accuracy: **93.9%**
  - Val Precision: **96.2%**
  - Val Recall: **91.3%**
  - Val F1: **93.7%**

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd reticulocyte

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.1
- Ultralytics >= 8.0.200 (YOLOv8)
- timm >= 0.9.12 (EfficientNet)
- OpenCV >= 4.8
- scikit-learn >= 1.3
- albumentations >= 1.3

---

## Usage

### Run Inference (Single Image)

```bash
# Deep learning mode (GPU recommended)
python 04_model_baseline/pipeline.py --input path/to/smear.jpg --mode deep

# Classical mode (CPU only)
python 04_model_baseline/pipeline.py --input path/to/smear.jpg --mode classical
```

### Run Inference (Batch)

```bash
python 04_model_baseline/pipeline.py --input data/test_images/ --output output/ --mode deep
```

### Train From Scratch

```bash
# 1. Convert BCCD to YOLO format
python 04_model_baseline/preprocess.py --task bccd_to_yolo \
    --input data/bccd_raw --output data/bccd_yolo

# 2. Generate synthetic reticulocytes
python 04_model_baseline/preprocess.py --task synthesize_reticulocytes \
    --input data/bccd_crops --output data/synthetic_retic --n_samples 2000

# 3. Train RBC detector
python 04_model_baseline/train_detector.py --data data/bccd_yolo --epochs 50

# 4. Train reticulocyte classifier
python 04_model_baseline/train_classifier.py --data data/synthetic_retic --epochs 30 --synthetic
```

### Using the Shell Helper

```bash
bash run_pipeline.sh          # See available commands
bash run_pipeline.sh test     # Run smoke test
bash run_pipeline.sh train    # Train both models
bash run_pipeline.sh eval     # Evaluate pipeline
bash run_pipeline.sh demo     # Run demo
```

### Interactive Demo

```bash
jupyter notebook 06_demo.ipynb
```

---

## Evaluation Results

### YOLOv8 Detector (BCCD Test Set)

| Metric | Value |
|--------|-------|
| Precision | 80.1% |
| Recall | 77.6% |
| mAP@0.5 | 87.5% |
| mAP@0.5:0.95 | 59.7% |
| Inference (GPU) | ~0.3s/image |
| Inference (CPU) | ~3s/image |

### EfficientNet-B0 Classifier (Synthetic Validation Set)

| Metric | Value |
|--------|-------|
| Accuracy | 93.9% |
| Precision | 96.2% |
| Recall | 91.3% |
| F1 Score | 93.7% |

> **Note**: Classifier metrics are on synthetic data only. Real-world performance is unvalidated.

---

## Limitations

1. **Synthetic Training Data**: The reticulocyte classifier is trained only on simulated blue granules. Real supravital stains (new methylene blue, brilliant cresyl blue) have different color profiles and staining artifacts.

2. **No Real Labeled Data**: The CENPARMI dataset (2461 real images) requires a consent form from Concordia University. Without it, real-world accuracy is unknown.

3. **Detector Domain Shift**: YOLOv8 was trained on Giemsa-stained BCCD images. Performance on supravital stains used for reticulocyte counting is untested.

4. **Borderline Classification**: Faintly stained reticulocytes (nearly mature) are likely misclassified as mature RBCs — the most clinically important error mode.

5. **Staining Artifacts**: Over-staining or precipitation can create blue regions mistaken for reticulum, causing false positives.

---

## Recommended Next Steps

### Immediate (0-3 months)
1. Submit CENPARMI consent form to Concordia University
2. Fine-tune EfficientNet-B0 on 2,461 real reticulocyte images
3. Partner with a clinical lab for 500+ labeled supravitally stained smears
4. Validate against a Sysmex XN analyzer

### Short-term (3-6 months)
5. Replace two-stage pipeline with end-to-end YOLOv8 reticulocyte detector
6. Domain adaptation across 3+ labs/stain batches
7. Add uncertainty quantification for borderline cases

### Medium-term (6-12 months)
8. Commission expert annotation study (inter-rater agreement kappa > 0.90)
9. Pursue Class II medical device approval (510(k) pathway)
10. Deploy as AI-assisted review tool (pathologist-in-the-loop)

---

## Deployment Recommendation

> **Deploy as an AI-assisted tool, NOT a fully automated counter.**
>
> - No validated public reticulocyte dataset exists
> - Model trained on synthetic data cannot be trusted for autonomous clinical decisions
> - Inter-observer agreement on borderline reticulocytes is < 80% among humans
> - Regulatory pathway for assist tool is faster (Class II vs. Class III)
> - Matches the deployment strategy of Wang et al. (2024), the current state-of-the-art

---

## References

See `01_literature_review.md` for a comprehensive review of 15 papers covering reticulocyte counting methods, including:
- Wang et al. (2024) — 97%+ accuracy with real supravital stain data
- Traditional Hough + SVM approaches
- Deep learning (YOLO, EfficientNet) methods
- Clinical validation protocols

See `02_dataset_inventory.csv` for a catalog of 11 relevant datasets with licensing information.
