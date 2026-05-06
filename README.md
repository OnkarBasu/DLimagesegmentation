# Image Segmentation: Car / Airplane / Dog

A PyTorch image segmentation project that takes a photo and labels every pixel as one of four classes: **Background**, **Car**, **Airplane**, or **Dog**.

Two models are trained and compared head-to-head on the same data:

1. **U-Net with a pretrained ResNet34 encoder + custom decoder** (main model)
2. **Vanilla U-Net** trained from scratch (Ronneberger 2015 — fair baseline)

Built as a defense-ready project for an MSc Data Science course at Vilnius University.

---

## Headline result

| Model | Macro F1 | Pixel accuracy |
|---|---|---|
| **ResNet34 + custom decoder** | **0.7658** | **84.19%** |
| Vanilla U-Net | 0.7221 | 81.19% |
| **Improvement** | **+0.0437** | **+3.00%** |

The pretrained encoder wins on every single class. Detailed comparison below.

---

## Dataset

- **Source:** OpenImages v7 (Google), pulled via FiftyOne
- **Classes:** Car, Airplane, Dog (+ Background)
- **Train:** 4000 images with segmentation masks
- **Test:** 2000 images, never seen during training
- **Resolution:** Resized to 224×224 for both training and evaluation

### Pixel distribution (training set)

| Class | Pixels | Share |
|---|---|---|
| Background | 2.35B | 77.97% |
| Car | 499M | 16.60% |
| Airplane | 54M | 1.80% |
| Dog | 109M | 3.63% |

The dataset is **severely imbalanced** — Background dominates, Airplane is rare. This drove most of our loss-function design choices (see Class imbalance section below).

---

## Architecture

### Model 1 — Pretrained ResNet34 encoder + custom decoder

The U-shape of U-Net with one half borrowed from ImageNet:

- **Encoder:** ResNet34 pretrained on ImageNet, loaded from `torchvision`. About 21M parameters, all initialized from years of ImageNet training so it already knows edges, textures, fur, metal, sky.
- **Decoder:** Written from scratch in this notebook. Four `UpBlock` modules (each: `ConvTranspose2d` upsample → concat skip connection → DoubleConv) plus a final upsample + 1×1 conv to produce 4-class logits. About 5M parameters.
- **Skip connections:** Tap ResNet34 at five depth levels (after `conv1`, `layer1`, `layer2`, `layer3`, `layer4`) and feed those features into matching decoder levels.
- **Total:** ~26M parameters

### Model 2 — Vanilla U-Net (baseline for Q13)

A clean reimplementation of the original 2015 U-Net paper, no pretrained weights anywhere:

- 4 encoder blocks: features `[64, 128, 256, 512]`
- Bottleneck: doubles to 1024 channels
- 4 decoder blocks (transposed conv upsampling + skip concat + DoubleConv)
- Final 1×1 conv to 4 output classes
- **Total:** ~31M parameters (heavier decoder than ResNet34's)

Both models share **everything else**: training data, image size, loss, class weights, augmentations, optimizer. Any difference in F1 is attributable to architecture/pretraining.

---

## Augmentations

Applied to **training only**. Test images are just resized and normalized.

| Augmentation | Probability | Why |
|---|---|---|
| Resize 224×224 | always | Required for ImageNet pretrained encoder |
| Horizontal flip | 0.5 | Cars/planes/dogs look natural mirrored — doubles effective dataset size |
| Color jitter | 0.4 | Brightness, contrast, saturation, hue — robustness to lighting and cameras |
| Gaussian blur | 0.2 | Robustness to slightly out-of-focus images |
| ImageNet normalization | always | Required so input matches ResNet34's pretraining distribution |

**Deliberately removed:** vertical flip and 90° rotation — upside-down cars and planes don't occur in real photos, and adding them would teach the model unnatural orientations.

---

## Class imbalance handling

Background takes up ~78% of all pixels. Without correction, a model that just predicts "Background everywhere" scores 78% pixel accuracy and never learns the rare classes at all.

Two techniques applied together:

### 1. Square-root inverse-frequency class weights in CrossEntropy

Standard inverse-frequency weights over-correct: rare classes get weight ~50×, the model over-predicts foreground, and Background recall collapses (we hit this exact problem in an earlier run — Background recall fell to 0.63).

Square-root inverse-frequency is gentler:

```
weight_class = sqrt(median_freq / freq_class)
```

Final weights: Background 0.28, Car 0.60, Airplane 1.83, Dog 1.29. Balanced — no class is ignored, no class dominates.

### 2. Dice loss combined 50/50 with CE

Dice loss measures region overlap rather than per-pixel correctness. It naturally handles imbalance because each class contributes equally regardless of pixel count.

```
total_loss = 0.5 × CrossEntropy(weights) + 0.5 × Dice
```

CE provides per-pixel gradients (good for fine boundaries). Dice provides region-level signal (good for under-represented classes). The combination outperforms either alone.

---

## Training setup

| Component | Value | Why |
|---|---|---|
| Optimizer | AdamW | Handles sparse segmentation gradients well |
| Learning rate | 1e-3 | Standard starting point for AdamW with cosine decay |
| Weight decay | 1e-4 | Mild regularization without distorting pretrained features |
| Batch size | 8 | Fits 224×224 + ResNet34 on Colab T4/L4 |
| LR schedule | Cosine decay with 5-epoch warmup | Gentle ramp at start, smooth decay to zero |
| Mixed precision | AMP (autocast + GradScaler) | ~2× speedup on T4/L4, negligible accuracy cost |
| Gradient clipping | max_norm=1.0 | Prevents loss spikes |
| Encoder freeze | First 3 epochs | Decoder adapts to pretrained features without disturbing them |
| LR halved on unfreeze | 1e-3 → 5e-4 | Gentler fine-tuning of the encoder |

### Training duration

- **ResNet34 + custom decoder:** Trained for 11 epochs (early stopping triggered after the encoder unfreeze caused a temporary loss spike. Investigation and fix: reset patience counter on unfreeze, halve LR on unfreeze, increase patience from 8 to 12 — final run completed cleanly).
- **Vanilla U-Net:** Trained for full 30 epochs without early stopping. No pretrained weights to protect, so it converges more slowly.

Total training time: ~70 minutes on a single Colab T4 GPU.

---

## Results

### ResNet34 + custom decoder — full metrics on 2000 test images

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Background | 0.9699 | 0.8192 | **0.8882** | 75.97M |
| Car | 0.4765 | 0.9397 | **0.6324** | 7.34M |
| Airplane | 0.6091 | 0.8556 | **0.7116** | 5.23M |
| Dog | 0.7571 | 0.9206 | **0.8309** | 11.81M |
| **Macro avg** | 0.7032 | 0.8838 | **0.7658** | — |
| **Weighted avg** | 0.8900 | 0.8419 | 0.8536 | — |

**Overall pixel accuracy: 84.19%**

### Vanilla U-Net — same test set

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Background | 0.9519 | 0.7983 | 0.8684 |
| Car | 0.4381 | 0.8831 | 0.5856 |
| Airplane | 0.5861 | 0.7640 | 0.6633 |
| Dog | 0.6887 | 0.8758 | 0.7710 |
| **Macro avg** | 0.6662 | 0.8303 | **0.7221** |

**Overall pixel accuracy: 81.19%**

### Q13 — Side-by-side comparison

| Class | Vanilla F1 | ResNet34 F1 | Δ |
|---|---|---|---|
| Background | 0.8684 | 0.8882 | **+0.0198** |
| Car | 0.5856 | 0.6324 | **+0.0468** |
| Airplane | 0.6633 | 0.7116 | **+0.0483** |
| Dog | 0.7710 | 0.8309 | **+0.0598** |
| **Macro F1** | **0.7221** | **0.7658** | **+0.0437** |
| Pixel accuracy | 81.19% | 84.19% | +3.00% |

The pretrained ResNet34 wins on every class, with the biggest gains on the rarer/harder classes (Dog and Airplane). The improvement is real but modest — at 4000 training images, vanilla U-Net has enough data to learn its own features and almost catch up. We expect this gap to widen significantly in low-data regimes (under 500 images per class).

---

## Repository structure

```
segmentation_project_v3/
├── data/
│   ├── train/     (4000 images + masks)
│   └── test/      (2000 images + masks)
├── models/
│   ├── best_resnet34.pth        ← main model
│   └── best_vanilla_unet.pth    ← Q13 baseline
└── results/
    ├── data_check.png
    ├── pixel_distribution.txt
    ├── training_curves_comparison.png
    ├── confusion_matrix_resnet34.png
    ├── confusion_matrix_vanilla.png
    ├── sample_predictions_resnet34.png
    ├── pixel_pipeline_demo.png
    ├── evaluation_resnet34.txt
    ├── evaluation_vanilla.txt
    ├── comparison_table.txt
    ├── sanity_check_predictions.png    (Q3)
    └── internet_image_predictions.png  (Q1)
```

---

## How to run

The notebook is built for Google Colab with a GPU runtime (T4 or better).

1. Open the notebook in Colab
2. Run cells 1–7 to install, mount Drive, download data, build models
3. Run cells 8–14 to train both models and evaluate
4. Run cells 15–18 for the defense demos (pixel pipeline, internet images, sanity checks)
5. Cell 22 is a standalone inference block — paste it into any new Colab to use the trained model on a new image without re-running everything

End-to-end runtime: ~90 minutes including data download.

---

## Defense demonstrations

The notebook is structured around 15 defense questions covering data, model, training, metrics, and edge cases:

- **Q1** — Internet image predictions
- **Q3** — Behavior on multi-object images, no-object images, and random pixel noise
- **Q4–Q6** — Pixel-level pipeline walkthrough at pixel (123, 123): logits → softmax → argmax
- **Q7** — Six metrics on 2000 unseen test images
- **Q8, Q12** — Training duration, stopping criteria, loss + accuracy curves
- **Q9** — Augmentation rationale
- **Q10–Q11** — Architecture and optimizer choices
- **Q13** — Vanilla U-Net comparison (above)
- **Q14** — Next steps to improve
- **Q15** — Class imbalance handling (above)

Each question has a dedicated markdown cell explaining the reasoning, followed by a code cell demonstrating it.

---

## Next steps

Concrete things that would push macro F1 from 0.77 toward 0.82+:

1. **Larger encoder** — ResNet50 or EfficientNet-B3. Expected gain: +0.03–0.05 macro F1.
2. **Higher input resolution** — 224 → 384. Helps small objects (especially Airplane). Expected gain: +0.03–0.05.
3. **More Car training data** — Car F1 of 0.63 is the bottleneck. The Car class has the most visual ambiguity (cars on roads, parked overlapping, partial occlusions) and would benefit most from more diverse examples.
4. **Test-time augmentation** — Average predictions on image + horizontal flip. Cheap +0.01–0.02.
5. **Focal loss** — Replace weighted CE. Better at handling "easy background" pixels.

The realistic ceiling on this exact dataset is probably around 0.80–0.82 macro F1. Past that, the bottleneck becomes the quality of OpenImages segmentation masks themselves.

---

## Tech stack

- PyTorch 2.x + torchvision
- albumentations (augmentations)
- FiftyOne (OpenImages download)
- scikit-learn (metrics)
- matplotlib (plots)
- Colab T4/L4 GPU
