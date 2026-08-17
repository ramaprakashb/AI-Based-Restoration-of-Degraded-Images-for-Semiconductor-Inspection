# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

Semiconductor inspection requires high-quality microscopic images to identify defects and verify fine structures. Noise, blur, and resolution loss can hide important details and reduce inspection accuracy.

This project develops an **AI-based image restoration and 2× super-resolution system** that converts degraded low-resolution SEM images into clearer, higher-resolution images while preserving important structural details.

## Problem Statement

Microscopic semiconductor inspection images can suffer from:

* Speckle and acquisition noise
* Loss of fine structural details
* Blur and reduced spatial resolution
* Brightness and contrast degradation

These degradations can make small defects and semiconductor structures difficult to distinguish.

## Proposed Solution

The proposed system uses a lightweight deep-learning restoration network to learn the mapping between degraded low-resolution SEM images and high-quality ground-truth images.

**Degraded SEM Image → Normalization → AI Restoration & 2× Super-Resolution → Contrast/Sharpness Refinement → Restored Image**

The model takes **128×128 grayscale images** and generates **256×256 restored images**.

## AI Model Architecture

The restoration model used in this project is **SemiconSR**, a lightweight convolutional neural network developed for semiconductor image restoration and 2× super-resolution.

```text
Noisy Low-Resolution SEM Image
          │
          ▼
     Feature Extraction
        Conv2D
          │
          ▼
   Residual Learning
     8 Residual Blocks
          │
          ▼
      Upsampling
     PixelShuffle 2×
          │
          ▼
      Reconstruction
        Conv2D
          │
          ▼
Restored High-Resolution SEM Image
```

### Model Characteristics

* Input: **1 × 128 × 128 grayscale image**
* Output: **1 × 256 × 256 grayscale image**
* Super-resolution factor: **2×**
* Parameters: **776,705**
* Framework: **PyTorch**
* GPU acceleration: **CUDA**
* Mixed-precision training: **Automatic Mixed Precision (AMP)**

## Image Normalization

The degraded images are normalized using statistical values calculated from the training dataset.

Final normalization configuration:

* Mean: **0.4335362882**
* Standard deviation: **0.2847866113**

This provides a consistent input distribution for the restoration network while preserving the original image information during reconstruction.

## Training Methodology

The model was trained using a fixed paired dataset containing:

* Total images: **3,200**
* Training images: **2,560**
* Validation images: **640**
* Training/validation overlap: **0**

The training objective combines multiple losses to improve reconstruction quality and structural preservation:

* **Charbonnier Loss** — robust pixel-level reconstruction
* **SSIM Loss** — structural similarity preservation
* **Edge Loss** — fine-edge and boundary preservation
* **Sharpness Loss** — reduction of excessive smoothing

The final controlled training configuration used **100 epochs** with a **batch size of 16**.

## Image Restoration Pipeline

```text
Degraded / Noisy SEM Image
             │
             ▼
   Training-Set Normalization
             │
             ▼
       SemiconSR Network
             │
             ├── Feature Extraction
             ├── Residual Learning
             ├── Detail Reconstruction
             └── 2× Upsampling
             │
             ▼
      256 × 256 Restoration
             │
             ▼
   Contrast & Sharpness Refinement
             │
             ▼
      Final Restored SEM Image
```

## Evaluation Metrics

The model was evaluated on the fixed **640-image validation set** using three metrics:

* **PSNR (Peak Signal-to-Noise Ratio)** — measures pixel-level reconstruction quality. Higher is better.
* **SSIM (Structural Similarity Index)** — measures preservation of structural information. Higher is better.
* **LPIPS (Learned Perceptual Image Patch Similarity)** — measures perceptual similarity between the restored image and ground truth. Lower is better.

### Final Validation Results

| Metric            |        Result |
| ----------------- | ------------: |
| Validation Images |       **640** |
| Mean PSNR         | **28.313 dB** |
| Mean SSIM         |    **0.7673** |
| Mean LPIPS        |    **0.3008** |

These values were calculated against the corresponding ground-truth images from the validation dataset.

## Blind Test Evaluation

The official test set contained **400 degraded low-resolution images** without corresponding ground-truth images.

Therefore:

* All **400 test images** were processed by the trained AI model.
* Restored `.npy` outputs were generated.
* Restored PNG images were generated for visualization.
* Comparison images were generated between the noisy input and AI-restored output.
* PSNR and SSIM were **not calculated for the blind test set** because matching ground-truth images were unavailable.

This prevents unsupported or fabricated quantitative results.

## Visual Quality Considerations

The restoration approach focuses on:

**Noise Reduction:** Suppresses unwanted acquisition noise while preserving useful structures.

**Deblurring:** Reconstructs sharper boundaries and microscopic features from degraded inputs.

**Super-Resolution:** Enhances spatial resolution from 128×128 to 256×256.

**Contrast Preservation:** Maintains meaningful intensity differences and reduces washed-out appearance.

**Sharpness Preservation:** Uses edge-aware and sharpness-aware objectives to reduce excessive smoothing.

## Technology Stack

### Software

* Python
* PyTorch
* NumPy
* OpenCV
* Pillow
* scikit-image
* LPIPS
* Google Colab
* CUDA

### Compute Platform

* NVIDIA Tesla T4 GPU
* CUDA acceleration
* Automatic Mixed Precision for efficient training and inference

## Repository Structure

```text
AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection/
│
├── src/
│   ├── training/
│   ├── validation/
│   └── testing/
│
├── models/
│   └── model configuration / checkpoint information
│
├── results/
│   ├── comparison_images/
│   └── metrics/
│
└── README.md
```

## Applications

The proposed solution can support:

* Semiconductor wafer inspection
* SEM image enhancement
* Microscopic defect analysis
* Low-resolution inspection-image restoration
* Automated visual inspection
* AI-assisted semiconductor quality control

## Key Advantages

1. **Lightweight architecture** with only 776,705 parameters.
2. **2× resolution enhancement** from 128×128 to 256×256.
3. **Structure-aware restoration** using SSIM and edge-based objectives.
4. **Sharpness-aware reconstruction** to reduce excessive smoothing.
5. **Multi-metric evaluation** using PSNR, SSIM, and LPIPS.
6. **Blind-test compatibility** for degraded images where ground truth is unavailable.

## Reproducibility

The project uses a fixed **2,560/640 training-validation split** and fixed training-set normalization statistics to maintain consistent and comparable experiments.

The repository contains the implementation and evaluation scripts required to reproduce the restoration workflow without exposing the original dataset or unnecessarily large training artifacts.

## Conclusion

This project demonstrates an AI-based approach for restoring degraded semiconductor inspection images through **noise reduction, structural reconstruction, deblurring, and 2× super-resolution**.

The final validation results demonstrate measurable image reconstruction quality while maintaining a lightweight model suitable for further optimization and deployment in semiconductor inspection workflows.
