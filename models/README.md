# SemiconSR Model

## Overview

**SemiconSR** is a lightweight convolutional neural network developed for **2× super-resolution and restoration of degraded SEM inspection images**. It takes a noisy low-resolution SEM image and reconstructs a higher-resolution image while aiming to preserve important edges, structures, and fine textures.

## Architecture

The complete network follows this processing pipeline:

```text
Noisy Low-Resolution SEM Image
          │
          ▼
   Feature Extraction
      Conv2D 3×3
       1 → 64
          │
          ▼
   Residual Learning
    8 Residual Blocks
          │
          ▼
      Feature Conv
       64 → 64
          │
          ▼
       Upsampling
   Conv2D 64 → 256
          │
          ▼
    PixelShuffle 2×
          │
          ▼
     Reconstruction
      Conv2D 3×3
       64 → 1
          │
          ▼
Restored High-Resolution SEM Image
```

### 1. Input

The network receives a single-channel grayscale **128 × 128 noisy low-resolution SEM image**.

```text
Input shape: 1 × 128 × 128
```

The input contains noise and degraded structural information that must be recovered by the network.

### 2. Feature Extraction

A **3 × 3 convolution** converts the single-channel input into **64 feature maps**.

```text
1 channel → 64 feature channels
```

This initial layer extracts basic image information such as edges, intensity patterns, and local structures.

### 3. Residual Learning

The extracted features pass through **8 residual blocks**.

Each residual block uses convolutional layers and ReLU activation with a skip connection:

```text
Input Feature
     │
     ├───────────────┐
     ▼               │
   Conv 3×3          │
     │               │
    ReLU              │
     │               │
   Conv 3×3          │
     │               │
     └────── + ◄──────┘
            │
            ▼
     Refined Feature
```

Residual learning allows the network to focus on learning the missing or degraded information while preserving useful features from the original input.

### 4. Feature Refinement

After the residual blocks, another **3 × 3 convolution** processes the refined 64-channel feature representation before upsampling.

```text
64 → 64 feature channels
```

### 5. 2× Upsampling

The network increases the spatial resolution using a convolution followed by **PixelShuffle**.

The convolution generates the required feature channels:

```text
64 → 64 × 4 = 256 channels
```

PixelShuffle then rearranges these channels spatially to achieve **2× upscaling**:

```text
128 × 128 → 256 × 256
```

This provides efficient learned upsampling while avoiding a simple interpolation-based enlargement.

### 6. Image Reconstruction

A final **3 × 3 convolution** converts the upsampled 64-channel representation back into a single grayscale image.

```text
64 feature channels → 1 output channel
```

The resulting image is the restored high-resolution SEM image.

## Model Specifications

| Specification    |         Value |
| ---------------- | ------------: |
| Input            | 128 × 128 × 1 |
| Output           | 256 × 256 × 1 |
| Upscaling        |            2× |
| Feature Channels |            64 |
| Residual Blocks  |             8 |
| Activation       |          ReLU |
| Upsampling       |  PixelShuffle |
| Total Parameters |       776,705 |

## Why This Architecture?

The architecture combines **feature extraction, residual learning, and learned upsampling** in a compact network. Residual blocks help recover degraded SEM structures, while PixelShuffle provides efficient 2× resolution enhancement. The relatively small parameter count makes the model suitable for practical image-restoration applications where computational efficiency is important.

## Training Objective

The model was trained using a combined restoration objective containing:

* **Charbonnier Loss** — robust pixel-level reconstruction
* **SSIM Loss** — structural similarity preservation
* **Edge Loss** — recovery of important image boundaries
* **Sharpness Loss** — preservation of fine image details

The final controlled training configuration used **2,560 training images and 640 validation images**.

## Final Validation Reference

The trained model achieved the following results on the fixed 640-image validation set:

| Metric     |     Result |
| ---------- | ---------: |
| Mean PSNR  | 28.3130 dB |
| Mean SSIM  |     0.7673 |
| Mean LPIPS |     0.3008 |

**PSNR and SSIM:** higher is better.
**LPIPS:** lower is better.
