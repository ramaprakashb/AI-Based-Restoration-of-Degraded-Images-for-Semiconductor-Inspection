# AI-Based Restoration of Degraded Images for Semiconductor Inspection


## Overview


Semiconductor inspection systems rely on high-quality microscopy images to identify defects, structural irregularities, and manufacturing variations. Noise, blur, resolution loss, and intensity degradation can reduce the visibility of critical features and affect inspection reliability.


This project proposes a lightweight deep-learning image restoration system that reconstructs high-resolution SEM images from degraded low-resolution inputs while preserving important structural details.


---


## Problem Statement


SEM inspection images can suffer from:


- Speckle and acquisition noise
- Loss of fine structural details
- Reduced spatial resolution
- Blur and edge degradation
- Brightness and contrast variations


These degradations can make small semiconductor defects and microscopic structures difficult to distinguish.


Traditional interpolation methods can increase image size but cannot effectively recover information lost during image degradation.


---


## Proposed Solution


The proposed system uses a lightweight convolutional neural network to learn the mapping between degraded low-resolution SEM images and their corresponding high-resolution ground-truth images.


**Degraded SEM Image → Normalization → AI Restoration & 2× Super-Resolution → Restored High-Resolution SEM Image**


The model takes a **128 × 128 grayscale SEM image** and generates a **256 × 256 restored image**.


The training objective combines pixel-level reconstruction, structural preservation, edge preservation, and sharpness-aware learning to reduce noise and excessive smoothing while retaining important image details.


---


# Model Architecture


## Architecture Overview


The restoration network consists of five main stages:


1. Feature extraction
2. Residual feature learning
3. Feature refinement
4. Learned 2× upsampling
5. High-resolution image reconstruction


The architecture is designed to improve spatial resolution while preserving important SEM structures and fine details.


---


## 1. Feature Extraction


A 3 × 3 convolution converts the single-channel SEM input into **64 feature channels**.


```text
1 × 128 × 128
      ↓
3 × 3 Conv2D
      ↓
64 × 128 × 128

This stage extracts local information such as:

Edges
Intensity transitions
Surface patterns
Local structural features
2. Residual Feature Learning

The extracted features pass through 8 residual blocks.

Each residual block uses convolutional feature transformation together with a skip connection. The skip connection allows useful information from earlier layers to pass forward while the residual path learns additional image information.

This helps recover degraded structures without unnecessarily modifying useful features.

Input Features
      │
      ├──────────────────┐
      ▼                  │
   Conv2D                │
      ↓                  │
    ReLU                 │
      ↓                  │
   Conv2D                │
      │                  │
      └────── Add ◄──────┘
             │
             ▼
       Refined Features

The feature representation passes through 8 residual blocks.

3. Feature Refinement

After residual learning, another 3 × 3 convolution processes the refined feature representation.

64 Feature Channels
        ↓
     Conv2D
        ↓
64 Refined Feature Channels

This prepares the learned features for the resolution enhancement stage.

4. Learned 2× Upsampling

The network performs learned upsampling using a convolution followed by PixelShuffle.

The convolution produces the feature channels required by PixelShuffle, which rearranges the learned features spatially to increase the image resolution.

128 × 128
    ↓
Conv2D
    ↓
PixelShuffle ×2
    ↓
256 × 256

Unlike conventional interpolation, the upsampling operation is learned from the training data.

5. Image Reconstruction

The final 3 × 3 convolution converts the high-resolution feature representation into a single-channel grayscale SEM image.

64 Feature Channels
        ↓
     3 × 3 Conv2D
        ↓
1 × 256 × 256

The resulting image is the restored high-resolution SEM image.

SemiconSR

The complete lightweight restoration architecture described above is referred to in this project as SemiconSR.

SemiconSR is designed for SEM image restoration and 2× super-resolution, combining convolutional feature extraction, residual learning, feature refinement, and efficient learned upsampling.

Architecture Flow
Degraded Low-Resolution SEM
          │
          ▼
     3 × 3 Conv2D
        1 → 64
          │
          ▼
   8 Residual Blocks
          │
          ▼
     3 × 3 Conv2D
        64 → 64
          │
          ▼
      Upsampling
   Conv2D + PixelShuffle
          │
          ▼
      2× Resolution
  128 × 128 → 256 × 256
          │
          ▼
     3 × 3 Conv2D
        64 → 1
          │
          ▼
Restored High-Resolution SEM
Model Specifications
Specification	Value
Architecture	SemiconSR
Input	128 × 128 × 1
Output	256 × 256 × 1
Upscaling	2×
Feature Channels	64
Residual Blocks	8
Activation	ReLU
Upsampling	PixelShuffle
Total Parameters	776,705
Framework	PyTorch
Image Normalization

The degraded NoisyLR images are normalized using statistical values calculated from the training dataset.

The final normalization configuration is:

Mean = 0.4335362882
Std  = 0.2847866113

The same normalization configuration is used consistently during training, validation, and model inference.

Training Methodology

The model is trained using paired degraded low-resolution SEM images and their corresponding high-resolution ground-truth images.

NoisyLR Image
      │
      ▼
Training-Set Normalization
      │
      ▼
   SemiconSR
      │
      ├── Feature Extraction
      ├── Residual Learning
      ├── Feature Refinement
      └── 2× Learned Upsampling
      │
      ▼
AI Restored 256 × 256 SEM
      │
      ▼
Comparison with Ground Truth
      │
      ├── PSNR
      ├── SSIM
      └── LPIPS

The dataset contains:

3,200 NoisyLR images
3,200 corresponding GT images

A fixed split is used:

2,560 training images
640 validation images
No train-validation overlap
Training Configuration
Parameter	Configuration
Total Paired Images	3,200
Training Images	2,560
Validation Images	640
Input Resolution	128 × 128
Output Resolution	256 × 256
Batch Size	16
Training Epochs	100
Initial Learning Rate	0.0001
Optimizer	Adam
Mixed Precision	Automatic Mixed Precision
Model Parameters	776,705
Training Objective

The training objective combines multiple loss components to balance reconstruction accuracy, structural preservation, and detail recovery.

Charbonnier Loss

Provides robust pixel-level reconstruction and reduces sensitivity to individual noisy pixels.

SSIM Loss

Encourages preservation of structural similarity between the restored image and the ground-truth image.

Edge Loss

Encourages recovery of important image boundaries and structural transitions.

Sharpness Loss

Encourages preservation of high-frequency information and helps reduce excessive smoothing.

The combined objective helps the network optimize both numerical reconstruction quality and important visual structures.

Validation Evaluation

The final model was evaluated on the fixed 640-image validation set.

Three complementary metrics were used.

PSNR

Peak Signal-to-Noise Ratio measures pixel-level reconstruction fidelity.

Higher PSNR is better.

SSIM

Structural Similarity Index measures similarity in structural information between the restored image and ground truth.

Higher SSIM is better.

LPIPS

Learned Perceptual Image Patch Similarity measures perceptual similarity using deep visual features.

Lower LPIPS is better.

Final Validation Results
Metric	Mean Result
PSNR	28.3130 dB
SSIM	0.7673
LPIPS	0.3008

These values were calculated by directly comparing the AI-restored validation images with their corresponding ground-truth images.

The quantitative validation evaluation used the raw AI restoration output. No contrast correction or sharpening was applied when calculating these reported metrics.

Official Blind Test

The official test set contains 400 NoisyLR images.

The supplied test set does not contain corresponding ground-truth images.

Therefore, quantitative PSNR, SSIM, and LPIPS values cannot be legitimately calculated for the official test images.

All 400 test images were processed by the trained model to generate restored outputs.

Official Test NoisyLR
          │
          ▼
       SemiconSR
          │
          ▼
AI Restored 256 × 256 SEM

The blind-test pipeline generates:

Restored .npy images
Restored PNG images
NoisyLR vs AI-restored comparison images
Per-image inference results

No unsupported PSNR or SSIM values are reported for the blind test because matching ground-truth images are unavailable.

Visual Quality Refinement

The restoration model is trained to address several image-quality issues:

Noise Reduction

Suppresses unwanted acquisition noise while retaining meaningful SEM structures.

Deblurring

Improves degraded boundaries and microscopic structural features through learned restoration.

Super-Resolution

Increases the spatial resolution from 128 × 128 to 256 × 256 using learned 2× upsampling.

Edge Preservation

Uses edge-aware learning to preserve important structural boundaries.

Sharpness Preservation

Uses sharpness-aware learning to reduce excessive smoothing.

Contrast and Brightness

The quantitative validation metrics are evaluated without post-processing. Optional contrast or sharpness adjustments used for blind-test visualization are treated separately from the reported model metrics.

Technology Stack
Software
Python
PyTorch
Torchvision
NumPy
OpenCV
Pillow
scikit-image
Pandas
LPIPS
Matplotlib
Deep Learning
Convolutional Neural Networks
Residual Learning
Image Restoration
2× Super-Resolution
PixelShuffle Upsampling
Automatic Mixed Precision
Structural and Perceptual Evaluation
Compute Platform
Google Colab
NVIDIA Tesla T4 GPU
CUDA acceleration
Automatic Mixed Precision for efficient training and inference
Repository Structure
AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── train_model.py
│   ├── validate_model.py
│   ├── test_model.py
│   └── evaluate_metrics.py
│
├── models/
│   ├── .gitkeep
│   └── README.md
│
└── results/
    ├── .gitkeep
    └── README.md
Installation

Clone the repository:

git clone https://github.com/ramaprakashb/AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection.git
cd AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection

Install the required dependencies:

pip install -r requirements.txt
Usage

The repository provides separate scripts for the main stages of the restoration workflow.

Training
python src/train_model.py
Validation
python src/validate_model.py
Metric Evaluation
python src/evaluate_metrics.py
Blind Test Inference
python src/test_model.py

The scripts separate training, quantitative validation, metric evaluation, and blind test inference.

Outputs

The evaluation pipeline can generate:

Restored high-resolution SEM images
NoisyLR vs AI-restored comparison images
Per-image PSNR, SSIM, and LPIPS values for validation
Aggregate validation statistics
Blind-test restored images
Inference results for the official test set

Large datasets, generated image collections, and trained model checkpoints are not included directly in this repository.

Applications

The proposed solution can support:

Semiconductor wafer inspection
SEM image restoration
Microscopic defect analysis
Low-resolution inspection-image enhancement
Automated visual inspection
AI-assisted semiconductor quality control
Pre-processing for downstream semiconductor image analysis
Key Advantages
Lightweight Architecture — Only 776,705 trainable parameters.
2× Learned Super-Resolution — Converts 128 × 128 inputs into 256 × 256 restored images.
Residual Learning — Preserves useful features while learning degraded or missing information.
Structure-Aware Restoration — Uses SSIM and edge-aware objectives to preserve structural information.
Sharpness-Aware Training — Helps reduce excessive smoothing and preserve fine details.
Multi-Metric Evaluation — Uses PSNR, SSIM, and LPIPS for quantitative validation.
Blind-Test Capability — Generates restored images even when ground-truth images are unavailable.
Reproducibility

The project uses a fixed 2,560/640 training-validation split and fixed training-set normalization statistics to maintain consistent and comparable experiments.

The validation results are calculated using the same fixed validation set and evaluation procedure.

The repository provides the implementation and evaluation scripts required to reproduce the restoration workflow without exposing the original dataset or unnecessarily large training artifacts.

Conclusion

This project demonstrates a lightweight AI-based approach for restoring degraded semiconductor inspection images.

The proposed SemiconSR architecture combines convolutional feature extraction, residual learning, feature refinement, and learned 2× upsampling to reconstruct higher-resolution SEM images.

The final validation results demonstrate measurable reconstruction and perceptual quality:

PSNR: 28.3130 dB
SSIM: 0.7673
LPIPS: 0.3008

The approach is designed to improve visibility of fine semiconductor structures while maintaining a compact architecture suitable for further optimization and deployment in semiconductor inspection workflows.
