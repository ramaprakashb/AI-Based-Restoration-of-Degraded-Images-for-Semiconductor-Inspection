"""
SEMICON PS01
AI-Based Restoration of Degraded Images for Semiconductor Inspection

Standalone test/inference script.

Usage:
    python src/test_model.py \
        --input-dir path/to/test/images \
        --output-dir path/to/output

Example:
    python src/test_model.py \
        --input-dir data/test/NoisyLR \
        --output-dir restored_test_outputs

The script:
    1. Loads the trained SemiconSR checkpoint.
    2. Reads all .npy images from the input directory.
    3. Applies the project normalization.
    4. Runs 2× image restoration.
    5. Saves restored .npy files.
    6. Saves restored PNG files.

No Google Drive or Colab dependency is required.
"""

from pathlib import Path
import argparse
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ============================================================
# VERIFIED PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_psnr.pt"
)

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "test"
    / "NoisyLR"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "restored_test_outputs"
)

NORMALIZATION_MEAN = 0.4335362882
NORMALIZATION_STD = 0.2847866113

EXPECTED_PARAMETERS = 776_705


# ============================================================
# SEMICONSR ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual feature-learning block.
    """

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
    Lightweight 2× SEM restoration network.

    Input:
        1 × 128 × 128

    Output:
        1 × 256 × 256

    Architecture:
        3×3 convolution
        ↓
        8 residual blocks
        ↓
        feature refinement
        ↓
        convolution + PixelShuffle ×2
        ↓
        reconstruction convolution
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
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    model = SemiconSR(
        channels=64,
        num_blocks=8,
        scale=2,
    )

    parameter_count = count_parameters(model)

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "SemiconSR parameter-count mismatch.\n"
            f"Expected: {EXPECTED_PARAMETERS:,}\n"
            f"Found   : {parameter_count:,}"
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Place best_psnr.pt inside the checkpoints "
            "directory or provide --checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

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

def load_input(path):

    image = np.load(path)

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image: {path}\n"
            f"Received shape: {image.shape}"
        )

    if image.shape != (128, 128):
        raise ValueError(
            f"Expected input shape (128, 128): {path}\n"
            f"Received shape: {image.shape}"
        )

    return image


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    output,
    npy_path,
    png_path,
):

    output = np.asarray(
        output,
        dtype=np.float32,
    )

    # Save the restored numerical image.
    np.save(
        npy_path,
        output,
    )

    # Convert to displayable 8-bit grayscale PNG.
    display = np.clip(
        output,
        0.0,
        1.0,
    )

    display = (
        display * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(
        display,
        mode="L",
    ).save(
        png_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run SemiconSR inference on a directory "
            "of 128×128 .npy SEM images."
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
            "Directory where restored outputs "
            "will be written."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to trained .pt checkpoint."
        ),
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    files = sorted(
        input_dir.glob("*.npy")
    )

    if not files:
        raise RuntimeError(
            f"No .npy files found in:\n{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    npy_output_dir = (
        output_dir / "npy"
    )

    png_output_dir = (
        output_dir / "png"
    )

    npy_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("SEMICON PS01 — TEST INFERENCE")
    print("=" * 70)

    print(f"Input directory  : {input_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Device            : {device}")

    if torch.cuda.is_available():

        print(
            "GPU               : "
            + torch.cuda.get_device_name(0)
        )

    print(
        f"Test images       : {len(files)}"
    )

    print(
        f"Normalization mean: "
        f"{NORMALIZATION_MEAN}"
    )

    print(
        f"Normalization std : "
        f"{NORMALIZATION_STD}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, checkpoint = load_model(
        args.checkpoint,
        device,
    )

    print(
        f"Model parameters  : "
        f"{count_parameters(model):,}"
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch  : "
                f"{checkpoint['epoch']}"
            )

        if "val_psnr" in checkpoint:

            print(
                f"Checkpoint PSNR   : "
                f"{checkpoint['val_psnr']}"
            )

        if "val_ssim" in checkpoint:

            print(
                f"Checkpoint SSIM   : "
                f"{checkpoint['val_ssim']}"
            )

    print("=" * 70)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = time.time()

    with torch.inference_mode():

        for index, path in enumerate(
            files,
            start=1,
        ):

            image = load_input(path)

            tensor = torch.from_numpy(
                image
            ).unsqueeze(0).unsqueeze(0)

            tensor = tensor.to(
                device,
                dtype=torch.float32,
            )

            # Project normalization.
            tensor = (
                tensor - NORMALIZATION_MEAN
            ) / NORMALIZATION_STD

            restored = model(
                tensor
            )

            restored = restored.squeeze(
                0
            ).squeeze(0)

            restored = restored.detach().cpu()

            # The model output represents the
            # normalized reconstruction space.
            #
            # Convert back to image space.
            restored = (
                restored * NORMALIZATION_STD
                + NORMALIZATION_MEAN
            )

            restored = restored.clamp(
                0.0,
                1.0,
            )

            restored = restored.numpy()

            npy_path = (
                npy_output_dir
                / path.name
            )

            png_path = (
                png_output_dir
                / f"{path.stem}.png"
            )

            save_output(
                restored,
                npy_path,
                png_path,
            )

            if (
                index == 1
                or index % 25 == 0
                or index == len(files)
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                speed = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"[{index:04d}/"
                    f"{len(files):04d}] "
                    f"{path.name} | "
                    f"{speed:.2f} img/s"
                )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TEST INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : {len(files)}"
    )

    print(
        f"Total time       : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Speed            : "
        f"{len(files) / max(elapsed, 1e-9):.2f} img/s"
    )

    print(
        f"Restored NPY     : "
        f"{npy_output_dir}"
    )

    print(
        f"Restored PNG     : "
        f"{png_output_dir}"
    )

    print("\nNOTE:")
    print(
        "PSNR/SSIM/LPIPS are not calculated here "
        "because the official blind test set may "
        "not contain ground-truth images."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
