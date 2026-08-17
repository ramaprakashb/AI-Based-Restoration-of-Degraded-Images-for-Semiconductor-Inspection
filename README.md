AI-Based Restoration of Degraded Images for Semiconductor Inspection
Overview

Semiconductor inspection systems rely on high-quality microscopy images to identify defects, structural irregularities, and manufacturing variations. Noise, resolution loss, and signal degradation can reduce the visibility of critical features and affect inspection reliability.

This project proposes a lightweight deep-learning image restoration system that reconstructs high-resolution SEM images from degraded low-resolution inputs while preserving important structural details.

Problem Statement

SEM inspection images can suffer from:

Speckle and signal-dependent noise
Loss of fine structural details
Reduced spatial resolution
Blur and edge degradation
Intensity and contrast variations

These degradations can make small semiconductor defects difficult to identify accurately.

Traditional interpolation methods can increase image size but cannot effectively recover information lost during image degradation.

Proposed Solution

The proposed system uses a lightweight convolutional neural network to learn the mapping between degraded low-resolution SEM images and their corresponding high-resolution ground-truth images.

Noisy Low-Resolution SEM → Feature Extraction → Residual Feature Learning → 2× Learned Upsampling → Image Reconstruction → Restored SEM Image

The model is trained using reconstruction, structural, edge, and sharpness-aware objectives to improve image fidelity while reducing excessive smoothing.

Model Architecture
Architecture Overview

The restoration network consists of five major stages:

Feature extraction
Residual feature learning
Feature refinement
Learned 2× upsampling
High-resolution image reconstruction

This architecture is designed to improve resolution while retaining important SEM image structures and fine details.

1. Feature Extraction

A 3 × 3 convolution converts the single-channel SEM input into 64 feature channels.

1 × 128 × 128
      ↓
3 × 3 Conv2D
      ↓
64 × 128 × 128

This stage extracts local information such as edges, intensity transitions, surface patterns, and structural features.

2. Residual Feature Learning

The extracted features pass through 8 residual blocks.

Each residual block uses convolutional feature transformation together with a skip connection. The skip connection allows useful information from earlier layers to pass forward while the residual path learns additional degraded or missing image information.

This helps the network recover structural details without unnecessarily modifying useful input features.

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

The complete feature representation passes through 8 such residual blocks.

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

The convolution produces the feature channels required for PixelShuffle, which rearranges them spatially to increase the image resolution.

128 × 128
    ↓
Conv2D
    ↓
PixelShuffle ×2
    ↓
256 × 256

This provides learned 2× upsampling rather than simply enlarging the image using conventional interpolation.

5. Image Reconstruction

The final convolution converts the high-resolution feature representation into a single-channel grayscale SEM image.

64 Feature Channels
        ↓
     3 × 3 Conv2D
        ↓
1 × 256 × 256

The resulting image is the restored high-resolution SEM image.

SemiconSR

The complete lightweight restoration architecture described above is referred to in this project as SemiconSR.

SemiconSR is designed for SEM image restoration and 2× super-resolution, combining convolutional feature extraction, residual learning, and efficient learned upsampling.

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
Training Methodology

The model is trained using paired degraded low-resolution SEM images and their corresponding high-resolution ground-truth images.

NoisyLR Image
      │
      ▼
   SemiconSR
      │
      ▼
AI Restored Image
      │
      ├───────────────┐
      │               │
      ▼               ▼
    PSNR             SSIM
      │               │
      └────── GT ─────┘

The dataset contains:

3,200 NoisyLR images
3,200 corresponding GT images

A fixed dataset split is used:

2,560 training images
640 validation images
No train-validation overlap
Training Configuration
Parameter	Configuration
Training Images	2,560
Validation Images	640
Input Resolution	128 × 128
Output Resolution	256 × 256
Batch Size	16
Training Epochs	100
Initial Learning Rate	0.0001
Optimizer	Adam
Mixed Precision	AMP
Model Parameters	776,705
Input Normalization

The NoisyLR inputs use the audited training-set statistics:

Mean = 0.4335362882
Std  = 0.2847866113

The same normalization is used during validation and model inference.

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

The combined objective allows the model to optimize both numerical reconstruction quality and important visual structures.

Validation Evaluation

The final model was evaluated on the fixed 640-image validation set.

Three complementary metrics were used:

PSNR

Measures pixel-level reconstruction fidelity.

Higher PSNR is better.

SSIM

Measures structural similarity between the restored image and ground truth.

Higher SSIM is better.

LPIPS

Measures perceptual similarity using deep visual features.

Lower LPIPS is better.

Final Validation Results
Metric	Mean Result
PSNR	28.3130 dB
SSIM	0.7673
LPIPS	0.3008

These values were calculated by comparing the AI-restored validation images directly against their corresponding ground-truth images.

No contrast correction or sharpening was applied during this quantitative validation evaluation.

Official Test Set

The official test set contains 400 NoisyLR images.

The supplied test set does not contain corresponding ground-truth images. Therefore, quantitative PSNR, SSIM, and LPIPS values cannot be legitimately calculated for these official test images.

The trained model is used to perform blind inference and generate restored high-resolution SEM images.

Official Test NoisyLR
          │
          ▼
       SemiconSR
          │
          ▼
AI Restored 256 × 256 SEM
Quantitative vs Qualitative Evaluation

The validation set is used for quantitative evaluation because ground-truth images are available.

The official test set is used for blind inference and qualitative inspection because ground-truth images are not provided.

Any optional contrast or sharpness adjustment used for visualization of blind-test outputs is treated separately from the reported quantitative validation metrics.

Technologies Used
Hardware
NVIDIA Tesla T4 GPU
Google Colab
Google Drive
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
Deep Learning Techniques
Convolutional Neural Networks
Residual Learning
2× Super-Resolution
PixelShuffle Upsampling
Mixed-Precision Training
Image Restoration
Perceptual Evaluation
