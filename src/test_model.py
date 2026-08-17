"""
SEMICON PS01
AI-Based Restoration of Degraded Images for Semiconductor Inspection

Standalone Test / Inference Script
-----------------------------------

This script loads the trained SemiconSR model and restores all
supported test images from an input directory.

Required arguments:
    --input-dir
    --output-dir

Example:
    python src/test_model.py \
        --input-dir test_images \
        --output-dir restored_test_outputs

Expected repository structure:

    project/
    ├── README.md
    ├── requirements.txt
    ├── checkpoints/
    │   └── best_psnr.pt
    └── src/
        └── test_model.py

The script does NOT require:
    - Google Drive
    - Google Colab
    - manual code editing

Official blind test images do not contain GT, therefore this script
does not calculate PSNR, SSIM, or LPIPS for the blind test set.
"""

from pathlib import Path
import argparse
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "best_psnr.pt"
)

# Verified from the project validation configuration.
NORMALIZATION_MEAN = 0.4335362882
NORMALIZATION_STD = 0.2847866113

EXPECTED_PARAMETERS = 776_705

INPUT_HEIGHT = 128
INPUT_WIDTH = 128

OUTPUT_HEIGHT = 256
OUTPUT_WIDTH = 256


# ============================================================
# SEMICONSR
# ============================================================

class ResidualBlock(nn.Module):
    """Residual feature-learning block."""

    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + residual


class SemiconSR(nn.Module):
    """
    Lightweight SEM image restoration network.

    Input:
        1 × 128 × 128

    Output:
        1 × 256 × 256

    Architecture:
        Feature extraction
        -> 8 residual blocks
        -> feature refinement
        -> learned 2x upsampling
        -> image reconstruction
    """

    def __init__(
        self,
        channels=64,
        num_blocks=8,
        scale=2,
    ):
        super().__init__()

        self.head = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.body = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * scale * scale,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        )

        self.tail = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        features = self.head(x)

        residual = features

        features = self.residual_blocks(
            features
        )

        features = self.body(features)

        features = features + residual

        features = self.upsample(features)

        output = self.tail(features)

        return output


# ============================================================
# PARAMETER CHECK
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model(checkpoint_path, device):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Place best_psnr.pt in the repository's "
            "checkpoints/ directory or use --checkpoint."
        )

    model = SemiconSR()

    parameter_count = count_parameters(model)

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "\nArchitecture parameter mismatch.\n"
            f"Expected : {EXPECTED_PARAMETERS:,}\n"
            f"Found    : {parameter_count:,}\n"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:
        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict' or 'state_dict'."
        )

    # Strict loading is intentional.
    # It prevents silently using an incompatible model.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# INPUT LOADING
# ============================================================

def load_npy_image(path):

    image = np.load(path)

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # Allow a single-channel array saved as
    # (1, H, W) or (H, W).
    if image.ndim == 3:

        if image.shape[0] == 1:
            image = image[0]

        elif image.shape[-1] == 1:
            image = image[..., 0]

        else:
            raise ValueError(
                f"Unsupported image shape "
                f"{image.shape}: {path}"
            )

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D image, "
            f"received {image.shape}: {path}"
        )

    if image.shape != (
        INPUT_HEIGHT,
        INPUT_WIDTH,
    ):
        raise ValueError(
            f"Expected "
            f"{INPUT_HEIGHT}x{INPUT_WIDTH}, "
            f"received {image.shape}: {path}"
        )

    return image


# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def restore_image(model, image, device):

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        device=device,
        dtype=torch.float32,
    )

    # Project normalization.
    tensor = (
        tensor - NORMALIZATION_MEAN
    ) / NORMALIZATION_STD

    restored = model(tensor)

    if restored.shape[-2:] != (
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
    ):
        raise RuntimeError(
            "Unexpected model output size: "
            f"{tuple(restored.shape)}"
        )

    # Return from normalized space.
    restored = (
        restored * NORMALIZATION_STD
        + NORMALIZATION_MEAN
    )

    restored = restored.squeeze(
        0
    ).squeeze(
        0
    )

    restored = restored.detach().cpu().numpy()

    restored = np.clip(
        restored,
        0.0,
        1.0,
    )

    return restored.astype(np.float32)


# ============================================================
# OUTPUT SAVING
# ============================================================

def save_outputs(
    restored,
    output_dir,
    filename,
):

    npy_dir = output_dir / "npy"
    png_dir = output_dir / "png"

    npy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Numerical restored image.
    np.save(
        npy_dir / filename,
        restored,
    )

    # Displayable grayscale PNG.
    png_image = (
        np.clip(
            restored,
            0.0,
            1.0,
        ) * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(
        png_image,
        mode="L",
    ).save(
        png_dir / f"{Path(filename).stem}.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Standalone SemiconSR test inference."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing test .npy images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where restored images "
            "will be written."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to trained best_psnr.pt. "
            "Default: checkpoints/best_psnr.pt"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n"
            f"{args.input_dir}"
        )

    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory:\n"
            f"{args.input_dir}"
        )

    input_files = sorted(
        args.input_dir.glob("*.npy")
    )

    if not input_files:
        raise RuntimeError(
            "No .npy images were found in:\n"
            f"{args.input_dir}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "SEMICON PS01 — STANDALONE TEST INFERENCE"
    )
    print("=" * 72)

    print(
        f"Input directory : {args.input_dir}"
    )

    print(
        f"Output directory: {args.output_dir}"
    )

    print(
        f"Checkpoint      : {args.checkpoint}"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Input size      : "
        f"{INPUT_HEIGHT} × {INPUT_WIDTH}"
    )

    print(
        f"Output size     : "
        f"{OUTPUT_HEIGHT} × {OUTPUT_WIDTH}"
    )

    print(
        f"Test images     : {len(input_files)}"
    )

    print(
        f"Normalization    : "
        f"mean={NORMALIZATION_MEAN}, "
        f"std={NORMALIZATION_STD}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, checkpoint = load_model(
        args.checkpoint,
        device,
    )

    print(
        f"Parameters      : "
        f"{count_parameters(model):,}"
    )

    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "val_psnr" in checkpoint:
        print(
            f"Checkpoint PSNR : "
            f"{checkpoint['val_psnr']}"
        )

    if "val_ssim" in checkpoint:
        print(
            f"Checkpoint SSIM : "
            f"{checkpoint['val_ssim']}"
        )

    print("=" * 72)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    for index, input_path in enumerate(
        input_files,
        start=1,
    ):

        image = load_npy_image(
            input_path
        )

        restored = restore_image(
            model,
            image,
            device,
        )

        save_outputs(
            restored,
            args.output_dir,
            input_path.name,
        )

        if (
            index == 1
            or index % 25 == 0
            or index == len(input_files)
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            speed = (
                index / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"[{index:04d}/"
                f"{len(input_files):04d}] "
                f"{input_path.name} | "
                f"{speed:.2f} images/sec"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    speed = (
        len(input_files) / elapsed
        if elapsed > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("INFERENCE COMPLETE")
    print("=" * 72)

    print(
        f"Images processed : "
        f"{len(input_files)}"
    )

    print(
        f"Total time       : "
        f"{elapsed:.3f} seconds"
    )

    print(
        f"Speed            : "
        f"{speed:.2f} images/sec"
    )

    print(
        f"NPY outputs      : "
        f"{args.output_dir / 'npy'}"
    )

    print(
        f"PNG outputs      : "
        f"{args.output_dir / 'png'}"
    )

    print("\nMetric note:")
    print(
        "The official blind test set has no ground-truth "
        "reference images. PSNR, SSIM and LPIPS are "
        "therefore not calculated by this inference script."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
