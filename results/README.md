# Results

## Final Validation Results

The final SemiconSR model was evaluated on a fixed 640-image validation set containing paired NoisyLR and ground-truth SEM images.

| Metric | Result |
|---|---:|
| Mean PSNR | 28.3130 dB |
| Mean SSIM | 0.7673 |
| Mean LPIPS | 0.3008 |
| Median PSNR | 27.947 dB |
| Median SSIM | 0.8005 |
| Median LPIPS | 0.2597 |

## Model

- Architecture: SemiconSR
- Input: 128 × 128 × 1
- Output: 256 × 256 × 1
- Parameters: 776,705
- Upscaling: 2×
- Validation samples: 640

## Interpretation

Higher PSNR and SSIM indicate better reconstruction quality, while lower LPIPS indicates better perceptual similarity to the ground-truth image.

The results demonstrate that the trained model can restore degraded SEM images while recovering higher-resolution structural details and reducing visible noise.

## Evaluation

PSNR, SSIM, and LPIPS were calculated by comparing the AI-restored images directly against the corresponding ground-truth images in the validation set.
