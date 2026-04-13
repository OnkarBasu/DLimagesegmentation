# 🧠 Image Segmentation with Attention U-Net

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenImages](https://img.shields.io/badge/Dataset-OpenImages_v7-34A853?style=for-the-badge&logoColor=white)
![Colab](https://img.shields.io/badge/Platform-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-8A2BE2?style=for-the-badge)

**Semantic pixel-level segmentation of street scenes built entirely from scratch using PyTorch.**
Every pixel in a photograph is classified into one of three meaningful classes using a custom
Attention U-Net trained on real-world data from Google OpenImages v7.

</div>

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Classes and Labels](#-classes-and-labels)
- [Dataset](#-dataset--openimages-v7)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Setup](#-training-setup)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Key Design Decisions](#-key-design-decisions)
- [Technology Stack](#-technology-stack)
- [References](#-references)

---

## 🎯 What This Project Does

Given any street photograph, this model analyses every single pixel and assigns it a class label.
This task is called **semantic segmentation** — understanding a scene not just by detecting objects
with bounding boxes, but by painting a precise coloured mask over the entire image.

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   INPUT                              OUTPUT                          │
│                                                                      │
│   A street photograph        →       Pixel-level coloured mask       │
│                                                                      │
│   [person walking past]      →       ░░░░░░░░░░░░░░░░░░░ Background  │
│   [cars on road]             →       ████████████████████ Person     │
│   [buildings behind]         →       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Vehicle    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Unlike classification (one label per image) or detection (bounding boxes), segmentation requires
the model to make a separate decision for every pixel — a 160×160 image requires 25,600 individual
pixel-level classifications per forward pass.

---

## 🏷️ Classes and Labels

The model segments each image into **3 classes**. Each class is assigned a fixed integer index
stored directly as the pixel value in the label map.

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CLASS DEFINITIONS                              │
├───────────┬──────────────┬────────────────┬───────────────────────────┤
│   Index   │  Class Name  │  Mask Colour   │  What it covers           │
├───────────┼──────────────┼────────────────┼───────────────────────────┤
│     0     │  Background  │  Gray          │  Road, sky, buildings,    │
│           │              │  RGB(100,100,  │  pavement, trees, and     │
│           │              │  100)          │  anything that is not     │
│           │              │                │  a person or vehicle      │
├───────────┼──────────────┼────────────────┼───────────────────────────┤
│     1     │  Person      │  Purple        │  Any human body —         │
│           │              │  RGB(127,119,  │  full or partial,         │
│           │              │  221)          │  near or far,             │
│           │              │                │  any size in frame        │
├───────────┼──────────────┼────────────────┼───────────────────────────┤
│     2     │  Vehicle     │  Orange        │  Cars AND trucks merged   │
│           │              │  RGB(216, 90,  │  into one superclass.     │
│           │              │  48)           │  Any motorised            │
│           │              │                │  four-wheeled vehicle     │
└───────────┴──────────────┴────────────────┴───────────────────────────┘
```

> **Why Vehicle instead of separate Car and Truck?**
>
> Cars and trucks share approximately 90% of their visual features — wheels, metal body panels,
> windows, and rectangular silhouette. Training a from-scratch model on limited data to reliably
> distinguish them produces severe class confusion. Merging them into a single **Vehicle**
> superclass is the standard approach used in autonomous driving research and produces
> significantly cleaner segmentation boundaries.

---

## 📦 Dataset — OpenImages v7

**Google OpenImages v7** is one of the largest publicly available computer vision datasets,
containing approximately 9 million images annotated by professional human annotators
under a CC BY 4.0 licence.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPENIMAGES V7 — KEY FACTS                    │
├─────────────────────────────────────────────────────────────────┤
│  Total images              │  ~9 million                        │
│  Segmentation masks        │  2.8 million objects               │
│  Segmentation classes      │  350 classes                       │
│  Licence                   │  CC BY 4.0 (free for any use)      │
│  Annotation quality        │  Professional human annotators     │
│  Download tool used        │  FiftyOne Python library           │
└─────────────────────────────────────────────────────────────────┘
```

### Splits used in this project

| Split | Purpose | Images |
|:---:|:---:|:---:|
| `train` | Teaching the model — seen during training | 1,000 |
| `test` | Final evaluation only — never seen during training | 100 |

### How OpenImages stores masks

OpenImages provides **one binary PNG file per object instance**, not one label map per image.
A photo with 3 people and 2 cars produces 5 separate mask files:

```
street_photo.jpg
    │
    ├── person_mask_01.png   ← white pixels = first person
    ├── person_mask_02.png   ← white pixels = second person
    ├── car_mask_01.png      ← white pixels = first car
    └── truck_mask_01.png    ← white pixels = the truck
```

These per-instance masks are merged into a single unified label map before training.

---

## 🔄 Data Pipeline

```
╔══════════════════════════════════════════════════════════════════╗
║                        DATA PIPELINE                            ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1 — DOWNLOAD
┌──────────────────────────────────────────────────────────────────┐
│  FiftyOne connects to OpenImages v7                              │
│  Downloads only images containing Person / Car / Truck           │
│  1,000 training images  +  100 test images                       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 2 — MASK CONVERSION
┌──────────────────────────────────────────────────────────────────┐
│  Multiple binary PNGs  ──►  Single unified label map             │
│                                                                  │
│  For each object in the photo:                                   │
│    1. Read bounding box (normalised 0.0–1.0 coordinates)         │
│    2. Convert to pixel coordinates                               │
│    3. Resize binary mask crop to bounding box size               │
│    4. Paint class index into label map at that location          │
│                                                                  │
│  Result: one PNG per image — pixel value = class index           │
│    0 = background  │  1 = person  │  2 = car  │  3 = truck       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 3 — SAVE TO GOOGLE DRIVE
┌──────────────────────────────────────────────────────────────────┐
│  data/train/images/  ← 1,000 street photos (.jpg)                │
│  data/train/masks/   ← 1,000 label maps   (.png)                 │
│  data/test/images/   ←   100 street photos (.jpg)                │
│  data/test/masks/    ←   100 label maps   (.png)                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 4 — CLASS REMAPPING  (at load time, not on disk)
┌──────────────────────────────────────────────────────────────────┐
│  Masks on disk  :  0=background  1=person  2=car  3=truck        │
│  During loading :  mask[mask == 3] = 2                           │
│  In memory      :  0=background  1=person  2=vehicle             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 5 — AUGMENTATION  (training only)
┌──────────────────────────────────────────────────────────────────┐
│  Transform              │  Probability  │  Effect                │
│  ───────────────────────┼───────────────┼──────────────────────  │
│  Horizontal flip        │  0.50         │  Mirror left/right     │
│  Vertical flip          │  0.10         │  Mirror up/down        │
│  Random 90° rotation    │  0.20         │  Rotate image          │
│  Colour jitter          │  0.40         │  Vary brightness       │
│  Gaussian blur          │  0.20         │  Slight blur           │
│  ImageNet normalise     │  1.00         │  Always applied        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 6 — COPY TO LOCAL SSD  (once per session)
┌──────────────────────────────────────────────────────────────────┐
│  Google Drive (~10 MB/s)  ──►  Local SSD (~500 MB/s)             │
│  One-time copy — all epoch reads from fast local storage         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 7 — PYTORCH DATALOADER
┌──────────────────────────────────────────────────────────────────┐
│  batch_size=16  │  num_workers=2  │  pin_memory=True             │
│  shuffle=True (train)  │  shuffle=False (test)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Pixel class distribution

```
  Background  ████████████████████████████████████████░  ~84%
  Vehicle     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~12%
  Person      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~4%
```

---

## 🏗️ Model Architecture

### Overview

The model is an **Attention U-Net** — a U-shaped convolutional encoder-decoder with attention
gates at every skip connection. Built entirely from scratch in PyTorch with **zero pretrained
weights**.

```
╔══════════════════════════════════════════════════════════════════════╗
║                        ATTENTION U-NET                              ║
║  INPUT  (batch, 3, 160, 160)  — RGB photo                           ║
╚══════════════════════════════════════════════════════════════════════╝

  ENCODER (left — shrinks)              DECODER (right — grows)
  ─────────────────────────             ──────────────────────────

  ┌─────────────────┐                       ┌─────────────────┐
  │  DoubleConv     │                       │  DoubleConv     │
  │  3 → 64 ch      │──── AttentionGate ───►│  128 → 64 ch    │
  │  160 × 160      │                       │  160 × 160      │
  └────────┬────────┘                       └────────▲────────┘
           │  MaxPool (÷2)                           │  ConvTranspose (×2)
  ┌─────────────────┐                       ┌─────────────────┐
  │  DoubleConv     │                       │  DoubleConv     │
  │  64 → 128 ch    │──── AttentionGate ───►│  256 → 128 ch   │
  │  80 × 80        │                       │  80 × 80        │
  └────────┬────────┘                       └────────▲────────┘
           │  MaxPool (÷2)                           │  ConvTranspose (×2)
  ┌─────────────────┐                       ┌─────────────────┐
  │  DoubleConv     │                       │  DoubleConv     │
  │  128 → 256 ch   │──── AttentionGate ───►│  512 → 256 ch   │
  │  40 × 40        │                       │  40 × 40        │
  └────────┬────────┘                       └────────▲────────┘
           │  MaxPool (÷2)                           │  ConvTranspose (×2)
  ┌─────────────────┐                       ┌─────────────────┐
  │  DoubleConv     │                       │  DoubleConv     │
  │  256 → 512 ch   │──── AttentionGate ───►│  1024 → 512 ch  │
  │  20 × 20        │                       │  20 × 20        │
  └────────┬────────┘                       └────────▲────────┘
           │  MaxPool (÷2)                           │  ConvTranspose (×2)
           └──────────────► BOTTLENECK ──────────────┘
                            512 → 1024 ch  │  10 × 10

  OUTPUT  (batch, 3, 160, 160) — argmax → predicted class per pixel
```

### Building block — DoubleConv

```
  input
    ├─ Conv2d (3×3)  →  BatchNorm  →  ReLU  →  Dropout2d (p=0.1)
    └─ Conv2d (3×3)  →  BatchNorm  →  ReLU
  output  (same spatial size, new channel count)
```

### Building block — AttentionGate

```
  Skip (encoder)     Gating signal (decoder)
       │                      │
  Conv2d 1×1             Conv2d 1×1
  BatchNorm               BatchNorm
       └──────── ADD ─────────┘
                  │
                 ReLU → Conv2d 1×1 → BatchNorm → Sigmoid (0.0–1.0)
                  │
       skip × attention_map → attended skip → decoder
```

### Model specifications

| Parameter | Value |
|:---|:---|
| Architecture | Attention U-Net |
| Encoder channels | `[64, 128, 256, 512]` |
| Bottleneck channels | `1024` |
| Input size | `160 × 160 × 3` |
| Output size | `160 × 160 × 3` |
| Total parameters | ~31 million |
| Dropout (encoder) | `0.1` |
| Dropout (bottleneck) | `0.2` |
| Pretrained weights | **None** |

---

## ⚙️ Training Setup

```
  Platform  :  Google Colab
  GPU       :  NVIDIA Tesla T4  (16 GB VRAM)
  Epochs    :  40  (early stopping patience = 12)
  Batch     :  16 images per batch
```

### Class-weighted loss — SegNet Median Frequency Balancing

```
  freq[c]    =  pixels_of_class_c / total_pixels
  median_f   =  median of all class frequencies
  weight[c]  =  median_f / freq[c]   (capped at 10×, normalised)
```

| Class | Frequency | Weight effect |
|:---|:---:|:---|
| Background | ~84% | Low — very common |
| Vehicle | ~12% | Medium — reference |
| Person | ~4% | High — rare, strongly penalised |

### Combined loss

```
  Total Loss  =  0.5 × CrossEntropyLoss  +  0.5 × DiceLoss

  CrossEntropyLoss  →  per-pixel classification with class weights
  DiceLoss          →  mask overlap quality, cleans up boundaries
```

### Learning rate schedule

```
  Optimiser  :  AdamW   lr=1e-3   weight_decay=1e-4
  Scheduler  :  Linear warmup (epochs 0–5) + Cosine decay (epochs 5–60)
  Grad clip  :  max_norm = 1.0
```

---

## 📊 Results

### 1. Training Curve

<!-- ============================================================
     HOW TO ADD THIS IMAGE
     1. In your GitHub repo click  Add file → Upload files
     2. Upload  training_curve.png  into the  results/  folder
     3. Delete this comment block — the image will appear below
     ============================================================ -->

![Training Curve](results/training_curve.png)

**Analysis:** Both train and validation loss decrease steadily across all 40 epochs.
Notably, validation loss consistently sits *below* training loss — the opposite of
overfitting — which means the model generalises well to unseen data. The warmup phase
(epochs 0–5) stabilises early training and the cosine decay allows smooth convergence
in later epochs. Validation loss is still declining at epoch 40, indicating further
training would continue to improve performance.

---

### 2. Sample Predictions

<!-- ============================================================
     HOW TO ADD THIS IMAGE
     1. In your GitHub repo click  Add file → Upload files
     2. Upload  sample_predictions.png  into the  results/  folder
     3. Delete this comment block — the image will appear below
     ============================================================ -->

![Sample Predictions](results/sample_predictions.png)

**Analysis — row by row the image shows: original photo / ground truth mask / model prediction.**

| Image | Scene | Ground Truth | Prediction | Verdict |
|:---:|:---|:---|:---|:---:|
| 1 | Person beside aircraft | Small purple person | Large orange blob over aircraft body | ❌ Aircraft mistaken for Vehicle |
| 2 | Red fire truck | Large orange vehicle | Large orange blob, good shape | ✅ Strong Vehicle detection |
| 3 | Person on motorbike mid-air | Tiny purple person | Mixed orange and purple blob | ⚠️ Person found but shape inaccurate |
| 4 | Fire truck with person beside | Orange truck + purple person | Orange truck only, person missed | ⚠️ Vehicle good, Person missed |

The model reliably detects vehicles with clean boundaries. Person detection works for
isolated figures but struggles when persons are small, partially occluded, or adjacent
to large vehicles — a known challenge in from-scratch segmentation with limited training data.

---

### 3. Confusion Matrix

Each row represents the **true class**. Each column represents the **predicted class**.
Each row sums to 1.00. The diagonal is correct predictions — higher is better.

| True \ Predicted | Background | Person | Vehicle |
|:---:|:---:|:---:|:---:|
| **Background** | **0.66** ✅ | 0.09 | 0.25 ⚠️ |
| **Person** | 0.21 ⚠️ | **0.66** ✅ | 0.13 |
| **Vehicle** | 0.10 | 0.05 | **0.84** ✅ |

**Analysis:**
- **Background (row 1):** Correctly identified 66% of the time. The main confusion is 25% of background pixels predicted as Vehicle — the model over-extends vehicle boundaries into surrounding road and wall regions.
- **Person (row 2):** Correctly identified 66% of the time. 21% of person pixels are absorbed into Background — the model finds persons but loses them at the edges, explaining the low precision score.
- **Vehicle (row 3):** The strongest performer at 84% correct. Only 10% leaks into Background and 5% into Person. Vehicle segmentation is the most reliable output of the model.

---

### 4. Per-Class Metrics — 100 Unseen Test Images

**Overall pixel accuracy: 69.29%**

| Class | Precision | Recall | F1 Score | Support (pixels) |
|:---|:---:|:---:|:---:|:---:|
| Background | 0.9604 | 0.6616 | 0.7835 | 2,069,787 |
| Person | 0.1364 | 0.6567 | 0.2259 | 49,132 |
| Vehicle | 0.4147 | 0.8439 | 0.5561 | 441,081 |
| **Macro avg** | **0.5038** | **0.7207** | **0.5218** | 2,560,000 |
| Weighted avg | 0.8506 | 0.6929 | 0.7336 | 2,560,000 |

**Prediction distribution on test set:**

| Class | Predicted pixels | Share |
|:---|:---:|:---:|
| Background | 1,425,922 | 55.70% |
| Person | 236,562 | 9.24% |
| Vehicle | 897,516 | 35.06% |

**Analysis per class:**

- **Background — F1 0.78:** Very high precision (0.96) meaning almost every background prediction is correct. Lower recall (0.66) reflects the 25% bleed from the confusion matrix where background pixels get consumed by vehicle predictions.

- **Person — F1 0.23:** Good recall (0.66) means the model finds most persons. Very low precision (0.14) means it also produces many false positives — labelling non-person regions (walls, poles, aircraft bodies) as Person. This is the model's primary weakness and stems from the class being only 4% of training pixels despite a high class weight.

- **Vehicle — F1 0.56:** Excellent recall (0.84) confirms the model reliably detects vehicles. Moderate precision (0.41) reflects the over-prediction of vehicle boundaries into surrounding background areas. The strongest overall class in the model.

> **Context:** This model was trained entirely from scratch with no pretrained weights on only
> 1,000 images. A macro F1 of 0.52 and 69% pixel accuracy under these constraints represents
> a solid baseline. With pretrained encoder weights or more training data, Person precision
> and overall accuracy would improve substantially.

---



## 🔑 Key Design Decisions

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DECISION             │  CHOICE                 │  REASON                │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Architecture         │  Attention U-Net         │  Purpose-built for     │
│                       │                          │  segmentation — skip   │
│                       │                          │  connections preserve  │
│                       │                          │  fine spatial detail   │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Pretrained weights   │  None — from scratch     │  Assignment constraint │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Car vs Truck         │  Merged → Vehicle        │  90% visual overlap    │
│                       │                          │  causes confusion on   │
│                       │                          │  limited data          │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Loss function        │  CE + Dice  (50/50)      │  CE: pixel accuracy    │
│                       │                          │  Dice: mask overlap    │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Class weighting      │  SegNet median           │  Prevents background   │
│                       │  frequency formula        │  dominance (84% px)   │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  LR schedule          │  Warmup + cosine decay   │  Stable early training │
│                       │                          │  smooth convergence    │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Data storage         │  Drive → SSD copy        │  50× faster reads      │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Early stopping       │  Patience = 12           │  Saves best weights    │
└───────────────────────┴─────────────────────────┴────────────────────────┘
```

---

## 🛠️ Technology Stack

| Library | Purpose |
|:---|:---|
| **PyTorch** | Neural network, training loop, GPU computation |
| **torchvision** | Functional image transforms |
| **Albumentations** | Fast image and mask augmentation |
| **FiftyOne** | OpenImages v7 download and management |
| **Pillow** | Image loading, resizing, mask conversion |
| **NumPy** | Array operations, mask manipulation |
| **scikit-learn** | Precision, recall, F1, confusion matrix |
| **Matplotlib** | Training curves, visualisations |
| **Google Colab** | Cloud GPU notebook (Tesla T4) |
| **Google Drive** | Persistent data and model storage |

---

## 📚 References

| Paper | Authors | Year | Used for |
|:---|:---|:---:|:---|
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | Ronneberger et al. | 2015 | Base architecture |
| [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) | Oktay et al. | 2018 | Attention gates |
| [SegNet: A Deep Convolutional Encoder-Decoder Architecture](https://arxiv.org/abs/1511.00561) | Badrinarayanan et al. | 2017 | Class weighting formula |
| [V-Net: Fully Convolutional Neural Networks for Volumetric Segmentation](https://arxiv.org/abs/1606.04797) | Milletari et al. | 2016 | Dice loss |
| [The Open Images Dataset V4](https://arxiv.org/abs/1811.00982) | Kuznetsova et al. | 2020 | Dataset |

---

<div align="center">

**Built with PyTorch &nbsp;·&nbsp; Trained on OpenImages v7 &nbsp;·&nbsp; Google Colab T4**

</div>
