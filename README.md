# 🧠 Image Segmentation: Car / Airplane / Dog

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-T4_GPU-76B900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> A pixel-level image segmentation project that takes any photo and labels every single pixel as **Background**, **Car**, **Airplane**, or **Dog**. Built around two competing architectures — a U-Net with a pretrained ResNet34 encoder and a vanilla U-Net trained from scratch — to study what pretrained features actually buy you on a real-world segmentation task.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [What Semantic Segmentation Really Is](#-what-semantic-segmentation-really-is)
- [The Dataset](#-the-dataset)
- [Architecture Deep Dive](#%EF%B8%8F-architecture-deep-dive)
- [Class Imbalance and Loss Design](#%EF%B8%8F-class-imbalance-and-loss-design)
- [Augmentation Strategy](#-augmentation-strategy)
- [Training Recipe](#-training-recipe)
- [Results](#-results)
- [Head-to-Head Comparison](#%EF%B8%8F-head-to-head-comparison)
- [Technology Stack](#%EF%B8%8F-technology-stack)
- [References](#-references)

---

## 🎯 Project Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Input Photo (any size)                                         │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────┐         ┌─────────────────────────┐    │
│   │  Resize to 224×224  │  ────▶  │  Neural Network (U-Net) │    │
│   │  ImageNet normalize │         │   ResNet34 + decoder    │    │
│   └─────────────────────┘         └────────────┬────────────┘    │
│                                                │                 │
│                                                ▼                 │
│                                     Logits (4, 224, 224)         │
│                                                │                 │
│                                                ▼                 │
│                                     Softmax → Argmax             │
│                                                │                 │
│                                                ▼                 │
│                                     Predicted Mask (224, 224)    │
│                                     each pixel ∈ {0,1,2,3}       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

This is not classification. The model does not output one label for the whole image. It outputs **a label for every pixel** — about 50,000 decisions per photo at 224×224 resolution. Every pixel is independently classified into one of four classes, and together those classifications form a coloured mask that traces out the objects in the photo.

The whole project lives in a single Google Colab notebook structured around a complete experimental study from raw data download through final evaluation, comparing the pretrained-encoder approach against a vanilla baseline trained from scratch.

---

## 🔬 What Semantic Segmentation Really Is

Most computer vision people first meet is **classification** — "is this a dog or a cat?". That is one decision per image. The next step up is **detection** — "where is the dog, draw a box around it". That is a few decisions per image.

Segmentation is the next step beyond that:

```
   Classification         Detection            Segmentation
   ─────────────────      ──────────────       ──────────────────
   "There is a dog"       Box around dog       Every dog-pixel marked
   1 decision per image   ~4 numbers           ~50,000 decisions
```

This is **single-label-per-pixel** segmentation. Every pixel gets exactly one of four classes, decided by `argmax(softmax(logits))`. There is no scenario where a pixel is both "Car" and "Background" at the same time — that would be multi-label, which uses sigmoid and thresholds and is a different problem entirely.

The output we care about is a 2D mask of shape `(H, W)` where each value is a class index:

```
0 = Background    (everything that is not a vehicle or animal)
1 = Car           (cars, sedans, hatchbacks, SUVs)
2 = Airplane      (airliners, jets, propeller planes)
3 = Dog           (any breed)
```

For visualization we map each class to a colour: gray, orange, blue, green respectively.

---

## 📦 The Dataset

We use **OpenImages v7** — Google's public dataset with 9 million annotated images and around 350 segmentation classes. We pull only the photos that have segmentation masks for our three target classes via the **FiftyOne** library.

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   FiftyOne ──downloads──▶  OpenImages v7 (Google)                  │
│                                    │                               │
│                                    │ filter to {Car, Airplane,Dog} │
│                                    ▼                               │
│   For each image:                                                  │
│     ▶ Original photo (saved as .jpg)                               │
│     ▶ List of detections, each with:                               │
│         • bounding box                                             │
│         • binary mask cropped to that box                          │
│                                    │                               │
│                                    │ rasterize and composite       │
│                                    ▼                               │
│   Final mask: single (H,W) array, values in {0,1,2,3}              │
│   saved as .png                                                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

| | Count |
|:---|---:|
| Training images | 4,000 |
| Test images | 2,000 |
| Total pixels in train set | ~3.0 billion |
| Total pixels in test set | ~100 million |

### Pixel Distribution — the elephant in the room

```
Background  ████████████████████████████████████████  77.97%
Car         ████████                                  16.60%
Dog         ██                                         3.63%
Airplane    █                                          1.80%
```

Background pixels are **dominant**. Airplane is a **rare class**. This single fact drives most of the design choices that follow — the loss function, the class weighting, even how we evaluate. A naïve model that predicts "Background" for every pixel would already score 78% pixel accuracy and learn absolutely nothing useful. This is exactly why pixel accuracy alone is a misleading metric for imbalanced segmentation, and why we lean on macro F1 as the headline number throughout.

---

## 🏗️ Architecture Deep Dive

We train **two models** so we can directly compare what a pretrained encoder buys you. Both models share the same data, loss, optimizer, augmentations — only the architecture differs.

### Model 1 — Pretrained ResNet34 Encoder + Custom Decoder

```
                  INPUT  3 × 224 × 224
                         │
                         ▼
           ┌──────────────────────────────┐
           │   ResNet34 (PRETRAINED on    │
           │   ImageNet 1.2M images)      │
           │   ────────────────────────   │
           │                              │
           │   conv1+bn+relu  ───────► s1 (64ch, 112×112)
           │   maxpool                    │
           │   layer1         ───────► s2 (64ch,  56×56)
           │   layer2         ───────► s3 (128ch, 28×28)
           │   layer3         ───────► s4 (256ch, 14×14)
           │   layer4         ───────► b  (512ch,  7×7) bottleneck
           └──────────────────────────────┘
                         │
                         ▼
           ┌──────────────────────────────┐
           │   CUSTOM DECODER (our code)  │
           │   ────────────────────────   │
           │                              │
           │   UpBlock4(b , s4) ────► d4 (256ch, 14×14)
           │   UpBlock3(d4, s3) ────► d3 (128ch, 28×28)
           │   UpBlock2(d3, s2) ────► d2 (64ch,  56×56)
           │   UpBlock1(d2, s1) ────► d1 (64ch, 112×112)
           │   FinalUp(d1)      ────► d0 (32ch, 224×224)
           │                              │
           │   1×1 Conv ────────────► 4 logits per pixel
           └──────────────────────────────┘
                         │
                         ▼
                  OUTPUT  4 × 224 × 224
```

The encoder is **ResNet34 from torchvision**, loaded with `weights="IMAGENET1K_V1"`. ResNet34 was trained on 1.2 million ImageNet images for 1000-class classification. We don't care about its classification head — we only want its feature extraction capability. By the time training starts, this half of our network already understands edges, textures, fur, metal, sky, wheels, eyes, fabric, and a thousand other low- and mid-level concepts that emerge from massive-scale pretraining.

The decoder is **written from scratch in our notebook**. Each `UpBlock` does three things in sequence: upsamples the feature map by 2× with a transposed convolution, concatenates the matching skip connection from the encoder, then applies a `DoubleConv` (two 3×3 convolutions with BatchNorm and ReLU). The skip connections are how the network preserves fine spatial detail — without them the decoder would only have access to highly abstracted features and the output mask would be blurry.

```
┌──────────────────── UpBlock anatomy ────────────────────┐
│                                                         │
│   x (deep features, low res)    skip (early features)   │
│              │                          │               │
│              ▼                          │               │
│      ┌───────────────┐                  │               │
│      │ ConvTranspose │  upsample 2×     │               │
│      └───────┬───────┘                  │               │
│              │                          │               │
│              ▼                          ▼               │
│         ┌────────── concatenate ──────────┐             │
│         │                                 │             │
│         ▼                                 │             │
│    ┌──────────────────────────────────────┘             │
│    │  DoubleConv (Conv→BN→ReLU→Conv→BN→ReLU)            │
│    └──────────────────────────────────────┐             │
│                       │                                 │
│                       ▼                                 │
│                  next stage                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Component | Params | Source |
|:---|---:|:---|
| Encoder (ResNet34) | ~21M | Pretrained on ImageNet |
| Decoder (UpBlocks) | ~5M | Written from scratch |
| Final 1×1 conv | ~70 | Written from scratch |
| **Total** | **~26M** | |

### Model 2 — Vanilla U-Net (the fair baseline)

This is the original 2015 Ronneberger U-Net, no pretrained weights, every parameter starts from random Gaussian noise. It exists to answer the question "what does pretraining actually buy you?" by giving us a clean apples-to-apples comparison.

```
        Input 3 × 224 × 224
              │
              ▼
   ┌─────── ENCODER (from scratch) ──────────┐
   │ DoubleConv  3 → 64       ──────► skip₁  │
   │ MaxPool                                 │
   │ DoubleConv  64 → 128     ──────► skip₂  │
   │ MaxPool                                 │
   │ DoubleConv  128 → 256    ──────► skip₃  │
   │ MaxPool                                 │
   │ DoubleConv  256 → 512    ──────► skip₄  │
   │ MaxPool                                 │
   └─────────────────────────────────────────┘
              │
              ▼
   ┌────── BOTTLENECK ──────┐
   │ DoubleConv  512 → 1024 │
   └────────────────────────┘
              │
              ▼
   ┌─────── DECODER (from scratch) ──────────┐
   │ ConvTranspose 1024 → 512  + skip₄       │
   │ DoubleConv  1024 → 512                  │
   │ ConvTranspose  512 → 256  + skip₃       │
   │ DoubleConv   512 → 256                  │
   │ ConvTranspose  256 → 128  + skip₂       │
   │ DoubleConv   256 → 128                  │
   │ ConvTranspose  128 →  64  + skip₁       │
   │ DoubleConv   128 →  64                  │
   └─────────────────────────────────────────┘
              │
              ▼
   ┌─── 1×1 Conv  64 → 4 ───┐
   └─────────┬──────────────┘
             ▼
        Output 4 × 224 × 224
```

| Component | Params |
|:---|---:|
| Total | ~31M |

The vanilla U-Net is actually **larger** (31M vs 26M) than the pretrained model. So when the pretrained model wins, it cannot be because "more parameters" — it has to be because of the quality of those parameters at initialization.

### The architectural insight

Both networks have the same U-shape: encoder shrinks, bottleneck, decoder grows back, skip connections preserve detail. The difference lies entirely in **what state the encoder is in when training begins**:

```
   ┌────────────────────────┬─────────────────────────────────┐
   │                        │                                 │
   │    Vanilla U-Net       │    ResNet34 + Custom Decoder    │
   │                        │                                 │
   │   Encoder weights:     │    Encoder weights:             │
   │   random Gaussian      │    learned from ImageNet        │
   │   noise                │    1.2M images                  │
   │                        │                                 │
   │   Must learn from      │    Already knows edges,         │
   │   scratch on 4000      │    textures, shapes, parts —    │
   │   images               │    just needs to learn          │
   │                        │    "where are cars/planes/dogs" │
   │                        │                                 │
   └────────────────────────┴─────────────────────────────────┘
```

This is the central experiment of the project.

---

## ⚖️ Class Imbalance and Loss Design

Background takes up 78% of all pixels. Airplane takes up 1.8%. If we just throw plain CrossEntropy at this and train, the loss is dominated by Background pixels, gradient signal for the rare classes is overwhelmed, and we end up with a model that is great at predicting "Background" and terrible at everything else.

We attack this with **two complementary techniques** layered on top of each other.

### Technique 1 — Square-Root Inverse-Frequency Class Weights

The standard approach to imbalance is `weight = 1 / freq`. We tried that. It over-corrects spectacularly:

```
   Standard inverse-frequency weights (do NOT use)
   ────────────────────────────────────
   Background  freq=78%  →  weight = 0.05  (almost ignored)
   Car         freq=16%  →  weight = 0.26
   Airplane    freq= 2%  →  weight = 50.0  (dominates!)
   Dog         freq= 4%  →  weight = 22.0
```

With weights this extreme, the model learns to over-predict foreground everywhere. Background recall collapses (this exact failure mode showed up in an earlier experiment — 37% of background pixels were getting painted as foreground). Visually it looks like the model is hallucinating cars on walls and planes in empty sky.

The fix is a **gentler weighting scheme** based on the square root:

```
   weight_class = sqrt( median_freq / freq_class )
```

This compresses the dynamic range without flattening it:

```
   Square-root inverse-frequency weights (what we use)
   ───────────────────────────────────────────────────
   Background  freq=77.97%  →  weight = 0.28   (small but non-zero)
   Car         freq=16.60%  →  weight = 0.60
   Airplane    freq= 1.80%  →  weight = 1.83
   Dog         freq= 3.63%  →  weight = 1.29
```

Now Background still gets some attention, rare classes are amplified moderately, and no class dominates. This is **the single most important fix** for this dataset.

### Technique 2 — Dice Loss combined with CrossEntropy

Dice loss measures **region overlap** between the predicted and ground truth mask:

```
                2 × |prediction ∩ ground_truth|
   Dice = ────────────────────────────────────────
              |prediction| + |ground_truth|
```

The beautiful property of Dice is that it computes **one score per class regardless of how many pixels that class has**. A perfectly predicted Airplane scores 1.0 even if Airplane only takes up 2% of the image. A perfectly predicted Background also scores 1.0. The two contribute equally to the loss.

We use Dice as `1 - dice` (so that lower is better) and average across classes.

### Combined loss

```
   total_loss = 0.5 × CrossEntropy(weights) + 0.5 × Dice
```

| Component | What it gives the model |
|:---|:---|
| Weighted CE | Per-pixel gradient signal — good for learning fine boundaries |
| Dice | Region-level signal — good for under-represented classes |
| 50/50 mix | Balanced training, neither term dominates |

Empirically this combination outperforms either component alone on imbalanced segmentation datasets.

---

## 🎨 Augmentation Strategy

Augmentations apply to **training only**. The test set is just resized and ImageNet-normalized — never augmented — so evaluation is on real, unaltered images.

```
┌──────────────────── Train transform ────────────────────┐
│                                                         │
│  1. Resize 224 × 224              (always)              │
│  2. HorizontalFlip                p = 0.5               │
│  3. ColorJitter                   p = 0.4               │
│       brightness ± 0.3                                  │
│       contrast   ± 0.3                                  │
│       saturation ± 0.2                                  │
│       hue        ± 0.1                                  │
│  4. GaussianBlur                  p = 0.2               │
│  5. ImageNet Normalize            (always)              │
│       mean = [0.485, 0.456, 0.406]                      │
│       std  = [0.229, 0.224, 0.225]                      │
│  6. ToTensor                                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Augmentation | Why it's there |
|:---|:---|
| **Resize 224×224** | Required for ImageNet pretrained encoder |
| **Horizontal flip** | Cars, planes, dogs look natural mirrored — effectively doubles dataset size for free |
| **Color jitter** | Robustness to different lighting conditions, camera sensors, indoor vs outdoor |
| **Gaussian blur** | Robustness to slightly out-of-focus photos |
| **ImageNet normalize** | Required so input pixel statistics match what ResNet34 saw during pretraining |

### Augmentations we deliberately removed

```
   ✗  Vertical flip      — upside-down cars don't exist in real photos
   ✗  90° rotation       — same problem, all three classes sit upright
   ✗  Mosaic / mixup     — combines images, useful but adds complexity
   ✗  Cutout / erasing   — risks erasing the only object in the photo
```

Adding unnatural augmentations would teach the model to handle scenarios it will never see at inference time, at the cost of capacity for the scenarios that actually occur.

---

## 🚀 Training Recipe

```
┌─── HYPERPARAMETERS ─────────────────────────────────────────────────┐
│                                                                     │
│  Optimizer        : AdamW                                           │
│  Learning rate    : 1e-3                                            │
│  Weight decay     : 1e-4                                            │
│  Batch size       : 8                                               │
│  Image size       : 224 × 224                                       │
│  Mixed precision  : AMP (autocast + GradScaler)                     │
│  Gradient clipping: max_norm = 1.0                                  │
│                                                                     │
│  ResNet34 epochs  : up to 35   (early stop patience = 12)           │
│  Vanilla epochs   : up to 30   (early stop patience = 8)            │
│                                                                     │
│  Encoder freeze   : first 3 epochs (ResNet34 only)                  │
│  LR halved on unfreeze: 1e-3 → 5e-4 (ResNet34 only)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The encoder freeze trick

This is the single most important detail of training the pretrained model. For the first 3 epochs we set `requires_grad = False` on the entire ResNet34 encoder. Only the decoder parameters update. The pretrained ImageNet features are preserved exactly as they were learned.

```
   Phase 1 — epochs 1 to 3       Phase 2 — epochs 4 onwards
   ─────────────────────────     ──────────────────────────
                                                          
   ┌─────────────┐               ┌─────────────┐          
   │   FROZEN    │               │   TRAINING  │          
   │  encoder    │               │   encoder   │          
   │  (ImageNet) │               │  (ImageNet+)│          
   └──────┬──────┘               └──────┬──────┘          
          │                             │                 
          ▼                             ▼                 
   ┌─────────────┐               ┌─────────────┐          
   │  TRAINING   │               │   TRAINING  │          
   │  decoder    │               │   decoder   │          
   └─────────────┘               └─────────────┘          
                                                          
   Decoder learns to use         End-to-end fine-tuning   
   pretrained features           with halved LR for       
   without disturbing them       gentler encoder updates  
```

When the encoder unfreezes, three things happen simultaneously to make the transition smooth:

1. `requires_grad` flips to `True` on the encoder
2. Learning rate is halved (1e-3 → 5e-4) so the encoder updates gently rather than getting blown around
3. Early stopping patience counter is reset to 0 — without this, the brief loss spike that often follows unfreezing can trigger early stopping prematurely

This trick alone added several F1 points compared to a naïve "train everything from epoch 1" approach.

### Cosine learning rate schedule

```
   LR
    │
   1e-3 ┤        ╱─────╮
        │       ╱       ╲
        │      ╱           ╲
        │     ╱               ╲
   5e-4 ┤    ╱                   ╲___
        │   ╱                         ╲___
        │  ╱                              ╲___
        │ ╱                                   ╲___
      0 └─┴────┬────┬────┬────┬────┬────┬────┬────  epoch
        0    5     10   15   20   25   30   35
        warmup     ▲ cosine decay
                   │
                   encoder unfreeze + LR halve
```

5 epochs of linear warmup (LR climbs from 0 to 1e-3), then cosine decay all the way down to 0 over the remaining epochs. This is gentle at both ends — slow start to avoid disturbing the pretrained encoder during warmup, and slow finish to let the model settle into a good optimum.

---

## 📊 Results

The full results below are on **2000 unseen test images**. The test set was held out from training entirely — neither model ever saw a single test pixel during training.

### ResNet34 + Custom Decoder

```
RESULTS ON 2000 UNSEEN TEST IMAGES
═══════════════════════════════════════════════════════════════════
              precision    recall  f1-score   support
─────────────────────────────────────────────────────────────────
  Background     0.9699    0.8192    0.8882   75,968,594
         Car     0.4765    0.9397    0.6324    7,342,051
    Airplane     0.6091    0.8556    0.7116    5,231,596
         Dog     0.7571    0.9206    0.8309   11,809,759
─────────────────────────────────────────────────────────────────
   macro avg     0.7032    0.8838    0.7658
weighted avg     0.8900    0.8419    0.8536
─────────────────────────────────────────────────────────────────

  Overall pixel accuracy : 84.19%
  Macro F1               : 0.7658
═══════════════════════════════════════════════════════════════════
```

### Vanilla U-Net (baseline)

```
RESULTS ON 2000 UNSEEN TEST IMAGES
═══════════════════════════════════════════════════════════════════
              precision    recall  f1-score   support
─────────────────────────────────────────────────────────────────
  Background     0.9519    0.7983    0.8684   75,968,594
         Car     0.4381    0.8831    0.5856    7,342,051
    Airplane     0.5861    0.7640    0.6633    5,231,596
         Dog     0.6887    0.8758    0.7710   11,809,759
─────────────────────────────────────────────────────────────────
   macro avg     0.6662    0.8303    0.7221
weighted avg     0.8643    0.8119    0.8256
─────────────────────────────────────────────────────────────────

  Overall pixel accuracy : 81.19%
  Macro F1               : 0.7221
═══════════════════════════════════════════════════════════════════
```

### Confusion Matrix — ResNet34 + Custom Decoder (normalized rows)

```
                  Predicted →
                  Background    Car   Airplane    Dog
   ┌───────────┬────────────┬───────┬─────────┬───────┐
   │Background │    0.82    │ 0.10  │  0.04   │ 0.05  │
   ├───────────┼────────────┼───────┼─────────┼───────┤
True│       Car │    0.06    │ 0.94  │  0.00   │ 0.00  │  ✓
   ↓│  Airplane │    0.11    │ 0.03  │  0.86   │ 0.01  │  ✓
    │       Dog │    0.08    │ 0.00  │  0.00   │ 0.92  │  ✓
   └───────────┴────────────┴───────┴─────────┴───────┘
```

Read it row by row: **of all pixels that are truly Class X, what fraction did the model predict?**

- **94% of true Car pixels** are correctly identified as Car
- **86% of true Airplane pixels** are correctly identified as Airplane
- **92% of true Dog pixels** are correctly identified as Dog
- **82% of true Background pixels** are correctly kept as Background (the remaining 18% leak as foreground predictions — this is the main remaining error mode)

The big diagonal numbers are exactly what we want to see. Almost no class confuses with another foreground class (Car never predicts as Dog, Airplane never as Car, etc.) — confusion is almost entirely between foreground classes and Background, which is the easiest kind of error to interpret.

### Confusion Matrix — Vanilla U-Net (for comparison)

```
                  Predicted →
                  Background    Car   Airplane    Dog
   ┌───────────┬────────────┬───────┬─────────┬───────┐
   │Background │    0.80    │ 0.11  │  0.04   │ 0.06  │
   ├───────────┼────────────┼───────┼─────────┼───────┤
True│       Car │    0.10    │ 0.88  │  0.02   │ 0.00  │
   ↓│  Airplane │    0.17    │ 0.05  │  0.76   │ 0.01  │
    │       Dog │    0.12    │ 0.00  │  0.00   │ 0.88  │
   └───────────┴────────────┴───────┴─────────┴───────┘
```

The vanilla model is slightly weaker on every diagonal. Most notably it loses 10 points of recall on Airplane (0.76 vs 0.86) and 4 points on Dog. This is consistent with the macro-F1 gap.

---

## ⚔️ Head-to-Head Comparison

The point of training two models on identical data with identical recipes is to isolate the effect of the architecture/pretraining choice.

```
   Q13 — VANILLA U-NET   vs   RESNET34 + CUSTOM DECODER

   ┌─────────────┬─────────────┬──────────────┬───────────────┐
   │   Class     │ Vanilla F1  │ ResNet34 F1  │  Improvement  │
   ├─────────────┼─────────────┼──────────────┼───────────────┤
   │  Background │   0.8684    │    0.8882    │    +0.0198    │
   │  Car        │   0.5856    │    0.6324    │    +0.0468    │
   │  Airplane   │   0.6633    │    0.7116    │    +0.0483    │
   │  Dog        │   0.7710    │    0.8309    │    +0.0598    │
   ├─────────────┼─────────────┼──────────────┼───────────────┤
   │  Macro F1   │   0.7221    │    0.7658    │    +0.0437    │
   │  Pixel acc  │   81.19%    │    84.19%    │    +3.00%     │
   └─────────────┴─────────────┴──────────────┴───────────────┘
```

The ResNet34 model **wins on every single class**. The improvements are not enormous — about 4 macro-F1 points — but they are real and consistent. A few observations worth calling out:

```
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │   The biggest gains are on the rarer/harder classes:    │
   │     Dog       +0.060   ▲ largest improvement            │
   │     Airplane  +0.048                                    │
   │     Car       +0.047                                    │
   │     Background+0.020   ▲ smallest improvement           │
   │                                                         │
   │   This makes sense — Background is easy for both models │
   │   so there's less room for pretraining to help. Rare    │
   │   classes benefit more from pretrained features that    │
   │   already know what fur/metal/wings/wheels look like.   │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
```

The vanilla U-Net is *not* a bad model. 0.7221 macro F1 is a respectable result. The fact that the gap between the two models is only 4 F1 points (and not 10+) is itself an interesting finding: at 4000 training images, vanilla U-Net has enough data to learn its own features and almost catch up. The pretrained model still wins, but the margin is smaller than it would be in a low-data regime (think a few hundred images per class, where pretraining typically dominates).

---

## 🛠️ Technology Stack

| Library | Purpose |
|:---|:---|
| **PyTorch** | Neural network, training loop, GPU computation, AMP mixed precision |
| **torchvision** | ResNet34 with pretrained ImageNet weights, functional image transforms |
| **Albumentations** | Fast image-and-mask augmentation pipeline |
| **FiftyOne** | OpenImages v7 download and segmentation mask handling |
| **Pillow** | Image I/O, resizing, mask conversion |
| **NumPy** | Array operations, mask manipulation, pixel counting |
| **scikit-learn** | Precision, recall, F1, confusion matrix, classification report |
| **Matplotlib** | Training curves, confusion matrix heatmaps, sample predictions |
| **Google Colab** | Cloud GPU notebook (Tesla T4 / L4) |
| **Google Drive** | Persistent data and model checkpoint storage |

---

## 📚 References

| Paper | Authors | Year | Used for |
|:---|:---|:---:|:---|
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | Ronneberger, Fischer, Brox | 2015 | Vanilla U-Net architecture |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He et al. | 2015 | ResNet34 encoder |
| [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) | Milletari, Navab, Ahmadi | 2016 | Dice loss formulation |
| [SegNet: A Deep Convolutional Encoder-Decoder Architecture](https://arxiv.org/abs/1511.00561) | Badrinarayanan et al. | 2017 | Class weighting principles |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | Loshchilov, Hutter | 2017 | AdamW optimizer |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | Loshchilov, Hutter | 2016 | Cosine LR schedule |
| [The Open Images Dataset V4](https://arxiv.org/abs/1811.00982) | Kuznetsova et al. | 2020 | Source dataset |
| [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton | 2012 | ImageNet pretraining concept |

---

<div align="center">

**Built with PyTorch &nbsp;·&nbsp; Trained on OpenImages v7 &nbsp;·&nbsp; Google Colab T4**

</div>
