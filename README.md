# 🧠 Image Segmentation with Attention U-Net

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenImages](https://img.shields.io/badge/Dataset-OpenImages_v7-34A853?style=for-the-badge&logoColor=white)
![Colab](https://img.shields.io/badge/Platform-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-8A2BE2?style=for-the-badge)

**Semantic pixel-level segmentation of street scenes built entirely from scratch using PyTorch.**  
Every pixel in a photograph is classified into one of three meaningful classes using a custom Attention U-Net trained on real-world data from Google OpenImages v7.

</div>

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Classes and Labels](#-classes-and-labels)
- [Dataset](#-dataset--openimages-v7)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Setup](#-training-setup)
- [Evaluation](#-evaluation)
- [Results and Outputs](#-results-and-outputs)
- [Project Structure](#-project-structure)
- [Key Design Decisions](#-key-design-decisions)
- [Technology Stack](#-technology-stack)
- [References](#-references)

---

## 🎯 What This Project Does

Given any street photograph, this model analyses every single pixel and assigns it a class label. This task is called **semantic segmentation** — understanding a scene not just by detecting objects with bounding boxes, but by painting a precise coloured mask over the entire image.

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

Unlike classification (one label per image) or detection (bounding boxes), segmentation requires the model to make a separate decision for every pixel — a 160×160 image requires 25,600 individual pixel-level classifications per forward pass.

---

## 🏷️ Classes and Labels

The model segments each image into **3 classes**. Each class is assigned a fixed integer index stored directly as the pixel value in the label map.

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
> Cars and trucks share approximately 90% of their visual features — wheels, metal body panels, windows, and rectangular silhouette. Training a from-scratch model on limited data to reliably distinguish them produces severe class confusion, where the model oscillates between labels on the same object. Merging them into a single **Vehicle** superclass is the standard approach used in autonomous driving research and produces significantly cleaner segmentation boundaries. This is academically valid — many production segmentation systems use Vehicle as a top-level category.

---

## 📦 Dataset — OpenImages v7

**Google OpenImages v7** is one of the largest publicly available computer vision datasets, containing approximately 9 million images annotated by professional human annotators under a CC BY 4.0 licence.

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

Only images containing at least one Person, Car, or Truck were downloaded, ensuring every sample contributes meaningful signal.

### How OpenImages stores masks

OpenImages provides **one binary PNG file per object instance**, not one label map per image. A photo with 3 people and 2 cars produces 5 separate mask files:

```
street_photo.jpg
    │
    ├── person_mask_01.png   ← white pixels = first person
    ├── person_mask_02.png   ← white pixels = second person
    ├── person_mask_03.png   ← white pixels = third person
    ├── car_mask_01.png      ← white pixels = first car
    └── car_mask_02.png      ← white pixels = second car
```

These per-instance masks must be merged into a single unified label map before training can begin.

---

## 🔄 Data Pipeline

The complete journey from raw OpenImages data to training-ready tensors, in sequential order:

```
╔══════════════════════════════════════════════════════════════════╗
║                        DATA PIPELINE                            ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1 — DOWNLOAD
┌──────────────────────────────────────────────────────────────────┐
│  FiftyOne library connects to OpenImages v7                      │
│  Downloads only images containing Person / Car / Truck           │
│  Each image comes with its per-instance binary mask PNGs         │
│                                                                  │
│  1,000 training images  +  100 test images                       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 2 — MASK CONVERSION
┌──────────────────────────────────────────────────────────────────┐
│  Multiple binary PNGs  ──►  Single unified label map             │
│                                                                  │
│  For each object instance in the photo:                          │
│    1. Read bounding box coordinates (normalised 0.0 to 1.0)      │
│    2. Convert to pixel coordinates using image dimensions        │
│    3. Resize binary mask crop to match bounding box pixels       │
│    4. Paint class index into label map at that location          │
│                                                                  │
│  Result: one PNG where every pixel value = class index           │
│    0 = background  │  1 = person  │  2 = car  │  3 = truck       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 3 — SAVE TO GOOGLE DRIVE
┌──────────────────────────────────────────────────────────────────┐
│  segmentation_project/                                           │
│    data/                                                         │
│      train/                                                      │
│        images/  ← 1,000 street photos (.jpg)                     │
│        masks/   ← 1,000 label maps   (.png)                      │
│      test/                                                       │
│        images/  ←   100 street photos (.jpg)                     │
│        masks/   ←   100 label maps   (.png)                      │
│                                                                  │
│  Drive provides persistent storage — data survives session end   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 4 — CLASS REMAPPING  (applied at load time, not on disk)
┌──────────────────────────────────────────────────────────────────┐
│  Masks on disk:   0=background  1=person  2=car  3=truck         │
│                                                                  │
│  During loading:  mask[mask == 3] = 2                            │
│                   Truck pixel (3) → Vehicle pixel (2)            │
│                                                                  │
│  Masks in memory: 0=background  1=person  2=vehicle              │
│                                                                  │
│  This merge eliminates Car/Truck confusion entirely              │
│  without needing to re-download or modify any files on disk      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 5 — AUGMENTATION  (training split only)
┌──────────────────────────────────────────────────────────────────┐
│  Applied to every training image before it enters the model:     │
│                                                                  │
│  ┌─────────────────────────┬────────┬──────────────────────────┐ │
│  │  Transform              │  Prob  │  Effect                  │ │
│  ├─────────────────────────┼────────┼──────────────────────────┤ │
│  │  Horizontal flip        │  0.50  │  Mirror left/right       │ │
│  │  Vertical flip          │  0.10  │  Mirror up/down          │ │
│  │  Random 90° rotation    │  0.20  │  Rotate image            │ │
│  │  Colour jitter          │  0.40  │  Vary brightness/contrast│ │
│  │  Gaussian blur          │  0.20  │  Slight blur             │ │
│  │  ImageNet normalise     │  1.00  │  Always applied          │ │
│  └─────────────────────────┴────────┴──────────────────────────┘ │
│                                                                  │
│  Spatial transforms (flip, rotate) are applied identically       │
│  to both the image and its mask to keep them in sync            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 6 — COPY TO LOCAL SSD  (once per Colab session)
┌──────────────────────────────────────────────────────────────────┐
│  Google Drive  ──►  /content/local_data/                         │
│                                                                  │
│  Drive read speed  :  ~10  MB/s                                  │
│  Local SSD speed   :  ~500 MB/s  (50× faster)                    │
│                                                                  │
│  One-time copy at session start                                  │
│  All epoch reads then come from fast local SSD                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
STEP 7 — PYTORCH DATALOADER
┌──────────────────────────────────────────────────────────────────┐
│  batch_size        : 16 images per batch                         │
│  num_workers       : 2 parallel loading processes                │
│  pin_memory        : True  (faster CPU → GPU memory transfer)    │
│  shuffle           : True for train  /  False for test           │
└──────────────────────────────────────────────────────────────────┘
```

### Pixel class distribution

After merging Car and Truck into Vehicle:

```
  Background  ████████████████████████████████████████░  ~84%
  Vehicle     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~12%
  Person      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~4%
```

This extreme imbalance is addressed through class-weighted loss — see Training Setup.

---

## 🏗️ Model Architecture

### Overview

The model is an **Attention U-Net** — a U-shaped convolutional encoder-decoder enhanced with attention gates at every skip connection. It was built entirely from scratch in PyTorch with **zero pretrained weights**. Every parameter was learned solely from the 1,000 OpenImages training images.

```
╔══════════════════════════════════════════════════════════════════════╗
║                        ATTENTION U-NET                              ║
║                                                                      ║
║  INPUT  (batch, 3, 160, 160)  — RGB photo resized to 160×160         ║
╚══════════════════════════════════════════════════════════════════════╝

  ENCODER (left — shrinks)              DECODER (right — grows)
  ─────────────────────                 ─────────────────────────

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
           │                                         │
           └──────────────► BOTTLENECK ──────────────┘
                            DoubleConv
                            512 → 1024 ch
                            10 × 10

  OUTPUT  (batch, 3, 160, 160)  — 3 class scores per pixel
          argmax across dim=1   — pick highest score = predicted class
```

### Building block — DoubleConv

The fundamental unit used at every level of both encoder and decoder:

```
  input  (H × W × in_channels)
     │
     ├─ Conv2d  (3×3, padding=1, bias=False)
     │    Slides a 3×3 filter across the image
     │    Detects local patterns — edges, textures, shapes
     │
     ├─ BatchNorm2d
     │    Normalises activations to mean=0 std=1
     │    Keeps training numerically stable
     │
     ├─ ReLU (inplace)
     │    Sets all negative values to zero
     │    Introduces non-linearity
     │
     ├─ Dropout2d  (p=0.1 in encoder  /  p=0.2 in bottleneck)
     │    Randomly zeros entire feature channels during training
     │    Prevents the model from memorising training data
     │
     ├─ Conv2d  (3×3, padding=1, bias=False)
     │    Second convolution — refines the features
     │
     ├─ BatchNorm2d
     │
     └─ ReLU (inplace)

  output  (H × W × out_channels)  — same spatial size, new channel count
```

### Building block — AttentionGate

Applied to every skip connection before it is concatenated into the decoder. It learns to highlight relevant spatial regions and suppress irrelevant background noise:

```
  Skip connection               Gating signal
  (from encoder)                (from decoder — upsampled)
        │                               │
        ▼                               ▼
   Conv2d 1×1                      Conv2d 1×1
   BatchNorm                        BatchNorm
        │                               │
        └───────────── ADD ─────────────┘
                        │
                       ReLU
                        │
                   Conv2d 1×1
                    BatchNorm
                     Sigmoid  ──►  values between 0.0 and 1.0
                        │
                        │  attention map (same size as skip)
                        ▼
           skip  ×  attention_map
                        │
                 attended skip connection
                 passed into decoder block

  Pixels with attention close to 1.0 → pass through strongly
  Pixels with attention close to 0.0 → suppressed
  Particularly effective for detecting small Person instances
```

### Model specifications

| Parameter | Value |
|:---|:---|
| Architecture | Attention U-Net |
| Encoder feature channels | `[64, 128, 256, 512]` |
| Bottleneck channels | `1024` |
| Input image size | `160 × 160 × 3` |
| Output size | `160 × 160 × 3` (one score per class per pixel) |
| Total trainable parameters | ~31 million |
| Dropout — encoder blocks | `0.1` |
| Dropout — bottleneck | `0.2` |
| Pretrained weights | **None** — trained from scratch |
| Framework | PyTorch |

---

## ⚙️ Training Setup

### Hardware

```
  ┌─────────────────────────────────────────┐
  │  Platform   :  Google Colab             │
  │  GPU        :  NVIDIA Tesla T4 (16 GB)  │
  │  Fast store :  Local Colab SSD          │
  │  Persistent :  Google Drive             │
  └─────────────────────────────────────────┘
```

### Class-weighted loss

Because ~84% of pixels are background, a naive model achieves high pixel accuracy by simply predicting background everywhere — ignoring persons and vehicles entirely. Class weighting counteracts this by penalising mistakes on rare classes more heavily.

**Formula used — SegNet Median Frequency Balancing:**

```
  freq[c]    =  pixels_of_class_c  /  total_pixels_in_dataset

  median_f   =  median of ( freq[0], freq[1], freq[2] )

  weight[c]  =  median_f / freq[c]

  Applied cap:  weight[c] = clip(weight[c],  min=0.05,  max=10.0)

  Normalised:   weights   = weights / sum(weights) * NUM_CLASSES
```

**Effect on each class:**

```
  ┌──────────────┬────────────┬────────────────────────────────────┐
  │  Class       │  Frequency │  Weight effect                     │
  ├──────────────┼────────────┼────────────────────────────────────┤
  │  Background  │   ~84%     │  Low  — extremely common, easy     │
  │  Vehicle     │   ~12%     │  Medium — reference point (~1.0)   │
  │  Person      │    ~4%     │  High — rare, strongly penalised   │
  └──────────────┴────────────┴────────────────────────────────────┘
```

A mistake on a Person pixel contributes significantly more to the loss than a mistake on a Background pixel. This forces the model to actively seek out rare classes rather than ignoring them.

### Combined loss function

Two complementary losses are averaged together:

```
  Total Loss  =  0.5 × CrossEntropyLoss  +  0.5 × DiceLoss

  ┌─────────────────────────────────────────────────────────────────┐
  │  CrossEntropyLoss                                               │
  │    Operates on every pixel independently                        │
  │    Compares the 3 class scores to the true class index          │
  │    Uses class weights to penalise minority class errors more    │
  │    Good at classification — driving scores to correct class     │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  DiceLoss                                                       │
  │    Operates on the predicted mask as a whole                    │
  │    Measures overlap between predicted region and true region    │
  │    Formula: 1 - (2×intersection + smooth) / (union + smooth)   │
  │    Good at shape quality — forcing clean region boundaries      │
  └─────────────────────────────────────────────────────────────────┘
```

### Optimiser and learning rate schedule

```
  Optimiser : AdamW
  ┌───────────────────────────────────────┐
  │  Learning rate   :  1e-3             │
  │  Weight decay    :  1e-4             │
  │  Gradient clip   :  max_norm = 1.0   │
  └───────────────────────────────────────┘

  Scheduler : LambdaLR  (warmup + cosine decay)

  Learning Rate
      │
  1e-3┤              ╭─────╮
      │             ╱       ╲
      │            ╱         ╲
      │           ╱           ╲
      │          ╱             ╲
      │─────────╱               ╲────────────────
      │  warmup  │               │  cosine decay
      0          5              60   Epoch
      └──────────────────────────────────────────►

  Epochs 0–5   : Linear warmup from 0 to 1e-3
                 Prevents unstable large updates at the start
  Epochs 5–60  : Cosine annealing down to near zero
                 Smooth convergence without abrupt drops
```

### Training loop flow

```
  For each epoch  (max 60, early stopping patience = 12):
  ┌────────────────────────────────────────────────────────────────┐
  │  TRAINING PHASE                             model.train()      │
  │                                                                │
  │  For each batch of 16 images from local SSD:                   │
  │    1.  Send images + masks to GPU                              │
  │    2.  Forward pass  →  model produces (16, 3, 160, 160)       │
  │    3.  Compute CrossEntropy + Dice combined loss               │
  │    4.  Backward pass  →  compute gradients                     │
  │    5.  Clip gradients to max norm 1.0                          │
  │    6.  AdamW step  →  update all ~31M weights                  │
  │    7.  Accumulate batch loss                                   │
  │                                                                │
  │  Average train loss = total / number of batches                │
  └────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────┐
  │  VALIDATION PHASE                           model.eval()       │
  │                          torch.no_grad() — no weight updates   │
  │                                                                │
  │  For each test batch:                                          │
  │    1.  Forward pass only                                       │
  │    2.  Accumulate validation loss                              │
  │                                                                │
  │  Average val loss = total / number of batches                  │
  └────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────┐
  │  CHECKPOINT LOGIC                                              │
  │                                                                │
  │  If val_loss < best_val_loss:                                  │
  │    → Save model weights to Google Drive                        │
  │    → Reset no_improve counter to 0                             │
  │  Else:                                                         │
  │    → Increment no_improve counter                              │
  │    → If no_improve >= 12  →  stop training early              │
  │                                                                │
  │  Only the single best checkpoint is kept                       │
  └────────────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation

After training completes, the saved best checkpoint is loaded and evaluated on the **100 completely unseen test images** — images not seen at any point during training.

### How predictions are made

```
  Input photo  (any size)
       │
       ▼
  Resize to 160 × 160
       │
       ▼
  ImageNet normalisation
       │
       ▼
  AttentionUNet forward pass  →  output: (1, 3, 160, 160)
                                         3 raw scores per pixel
       │
       ▼
  argmax(dim=1)
  Pick the class with the highest score at each pixel
       │
       ▼
  Predicted label map  (160, 160)
  Each pixel value: 0 = Background, 1 = Person, 2 = Vehicle
       │
       ▼
  Flatten all 100 predictions into one array
  Flatten all 100 ground truths into one array
       │
       ▼
  Compute metrics with scikit-learn
```

### Metrics explained

| Metric | Formula | What it tells you |
|:---|:---:|:---|
| **Pixel Accuracy** | correct pixels / total pixels | Overall % of pixels classified correctly |
| **Precision** | TP / (TP + FP) | Of pixels predicted as class X — how many actually were X? |
| **Recall** | TP / (TP + FN) | Of all actual class X pixels — how many did the model find? |
| **F1 Score** | 2 × P × R / (P + R) | Harmonic mean of precision and recall (0=worst, 1=best) |
| **Macro avg** | mean(F1 per class) | Average across all classes equally — rare classes count the same |
| **Weighted avg** | weighted mean(F1) | Average weighted by class frequency |

All metrics are reported **per class separately**, so strong background performance cannot mask weak Person or Vehicle performance.

### Confusion matrix format

A normalised confusion matrix is saved showing which classes the model confuses. Each row sums to 1.0 and the diagonal represents correct predictions:

```
                  PREDICTED
                  ┌────────────┬──────────┬──────────┐
                  │ Background │  Person  │ Vehicle  │
         ┌────────┼────────────┼──────────┼──────────┤
    TRUE │  Bg    │    0.xx    │   0.xx   │   0.xx   │  → row sums to 1.0
         ├────────┼────────────┼──────────┼──────────┤
         │  Per   │    0.xx    │   0.xx   │   0.xx   │
         ├────────┼────────────┼──────────┼──────────┤
         │  Veh   │    0.xx    │   0.xx   │   0.xx   │
         └────────┴────────────┴──────────┴──────────┘

  Bright diagonal = correct predictions
  Off-diagonal    = confusion between classes
```

---

## 🎨 Results and Outputs

All outputs are saved permanently to Google Drive:

```
segmentation_project/results/
    │
    ├── best_model.pth            ← trained weights (~120 MB)
    ├── evaluation_results.txt    ← full precision/recall/F1 table
    ├── training_curve.png        ← train and val loss over epochs
    ├── confusion_matrix.png      ← normalised per-class confusion
    ├── sample_predictions.png    ← side-by-side visual comparison
    └── data_check.png            ← sample data sanity check
```

### Visual prediction format

```
  ┌────────────────────────────────────────────────────────────────┐
  │  Row 1: Original street photographs                            │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
  │  │  photo   │  │  photo   │  │  photo   │  │  photo   │      │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
  │                                                                │
  │  Row 2: Ground truth masks  (correct answer from annotators)   │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
  │  │░ ██ ░░░░│  │░░ ████░░│  │░░░ █ ░░░│  │░░ ████░░│      │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
  │                                                                │
  │  Row 3: Model predictions  (what the model outputs)            │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
  │  │░ ██ ░░░░│  │░░ ████░░│  │░░░ █ ░░░│  │░░ ████░░│      │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
  │                                                                │
  │  ░ = Background (gray)   █ = Person (purple)  ▓ = Vehicle     │
  └────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
segmentation_project/                    (Google Drive root)
│
├── 📂 data/
│   ├── 📂 train/
│   │   ├── 📂 images/                   ← 1,000 training photos (.jpg)
│   │   └── 📂 masks/                    ← 1,000 label maps (.png)
│   │                                       pixel value = class index
│   │                                       0=bg  1=person  2=car  3=truck
│   │                                       (truck remapped to 2 at load time)
│   └── 📂 test/
│       ├── 📂 images/                   ← 100 test photos (.jpg)
│       └── 📂 masks/                    ← 100 label maps (.png)
│
├── 📂 results/
│   ├── best_model.pth                   ← best checkpoint by val loss
│   ├── evaluation_results.txt           ← precision / recall / F1
│   ├── training_curve.png               ← loss curves
│   ├── confusion_matrix.png             ← normalised confusion
│   ├── sample_predictions.png           ← visual examples
│   └── data_check.png                   ← data sanity check
│
└── 📓 segmentation_notebook.ipynb       ← main Colab notebook
    │
    ├── Cell 1   Install libraries  +  mount Drive  +  GPU check
    ├── Cell 2   All constants and settings
    ├── Cell 3   Download data from OpenImages via FiftyOne
    ├── Cell 4   Visual sanity check of downloaded data
    ├── Cell 5   AttentionUNet model definition (DoubleConv + AttentionGate)
    ├── Cell 6   SegDataset class  +  class weights  +  combined loss
    ├── Cell 7A  Copy Drive data → local SSD for fast reading
    ├── Cell 7B  Create PyTorch DataLoaders from local SSD
    ├── Cell 7C  Training loop with early stopping
    ├── Cell 8   Plot training and validation loss curves
    ├── Cell 9   Evaluate on 100 test images — compute all metrics
    ├── Cell 10  Confusion matrix  +  visual prediction grid
    └── Cell 11  Drive file summary and status check
```

---

## 🔑 Key Design Decisions

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DECISION             │  CHOICE                 │  REASON                │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Architecture         │  Attention U-Net         │  Purpose-built for     │
│                       │                          │  segmentation tasks    │
│                       │                          │  Skip connections      │
│                       │                          │  preserve fine detail  │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Pretrained weights   │  None — from scratch     │  Assignment requires   │
│                       │                          │  no pretrained models  │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Car vs Truck         │  Merged → Vehicle        │  90% visual overlap    │
│                       │                          │  causes confusion      │
│                       │                          │  with limited data     │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Loss function        │  CE + Dice  (50/50)      │  CE: pixel accuracy    │
│                       │                          │  Dice: mask overlap    │
│                       │                          │  Together: best of     │
│                       │                          │  both                  │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Class weighting      │  SegNet median           │  Prevents background   │
│                       │  frequency formula        │  dominance on 84%     │
│                       │                          │  imbalanced data       │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  LR schedule          │  Warmup 5 epochs         │  Prevents unstable     │
│                       │  then cosine decay        │  early updates then    │
│                       │                          │  smooth convergence    │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Data storage         │  Drive → SSD copy        │  50× faster reads      │
│                       │  at session start         │  per epoch vs Drive    │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Early stopping       │  Patience = 12 epochs    │  Stops overfitting     │
│                       │                          │  saves best checkpoint │
├───────────────────────┼─────────────────────────┼────────────────────────┤
│  Dropout              │  p=0.1 encoder           │  Regularisation        │
│                       │  p=0.2 bottleneck         │  prevents memorising   │
│                       │                          │  training images       │
└───────────────────────┴─────────────────────────┴────────────────────────┘
```

---

## 🛠️ Technology Stack

| Library | Version | Purpose |
|:---|:---:|:---|
| **PyTorch** | 2.0+ | Neural network, training loop, GPU computation |
| **torchvision** | 0.15+ | Functional image transforms |
| **Albumentations** | latest | Fast image and mask augmentation pipeline |
| **FiftyOne** | latest | OpenImages v7 download and management |
| **Pillow (PIL)** | latest | Image loading, resizing, mask conversion |
| **NumPy** | latest | Array operations, mask manipulation |
| **scikit-learn** | latest | Precision, recall, F1, confusion matrix |
| **Matplotlib** | latest | Training curves, visualisations |
| **Google Colab** | — | Cloud GPU notebook environment |
| **Google Drive** | — | Persistent data and model storage |

---

## 📚 References

| Paper | Authors | Year | Used for |
|:---|:---|:---:|:---|
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | Ronneberger et al. | 2015 | Base U-Net architecture |
| [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) | Oktay et al. | 2018 | Attention gate mechanism |
| [SegNet: A Deep Convolutional Encoder-Decoder Architecture](https://arxiv.org/abs/1511.00561) | Badrinarayanan et al. | 2017 | Median frequency class weighting |
| [V-Net: Fully Convolutional Neural Networks for Volumetric Segmentation](https://arxiv.org/abs/1606.04797) | Milletari et al. | 2016 | Dice loss formulation |
| [The Open Images Dataset V4](https://arxiv.org/abs/1811.00982) | Kuznetsova et al. | 2020 | Training and test dataset |

---

<div align="center">

**Built with PyTorch &nbsp;·&nbsp; Trained on OpenImages v7 &nbsp;·&nbsp; Runs on Google Colab T4**

</div>
