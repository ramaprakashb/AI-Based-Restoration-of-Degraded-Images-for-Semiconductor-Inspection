"""
SEMICON PS01
AI-Based Restoration of Degraded Images for Semiconductor Inspection

Standalone Test / Inference Script
-----------------------------------

Usage:
    python src/test_model.py \
        --input-dir <TEST_IMAGES> \
        --output-dir <OUTPUT_DIRECTORY>

The script:
1. Loads the trained SemiconSR model.
2. Loads best_psnr.pt.
3. Reads all 128x128 grayscale NumPy test images.
4. Applies the project normalization.
5. Performs 2x learned super-resolution.
6. Saves restored 256x256 images as PNG and NPY.

No Google Drive or Google Colab is required.
"""

from pathlib import Path
import argparse
import time
from collections import OrderedDict

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

NORMALIZATION_MEAN = 0.4335362882
NORMALIZATION_STD = 0.2847866113

EXPECTED_PARAMETERS = 776_705

INPUT_SIZE = (128, 128)
OUTPUT_SIZE = (256, 256)


# ============================================================
# SEMICONSR
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual feature-learning block.

    Two 3x3 convolutions with a skip connection.
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
    Lightweight SEM image restoration and 2x
    super-resolution network.

    Input:
        1 x 128 x 128

    Output:
        1 x 256 x 256

    Architecture:
        3x3 feature extraction
        -> 8 residual blocks
        -> feature refinement
        -> Conv + PixelShuffle 2x upsampling
        -> image reconstruction
    """

    def __init__(
        self,
        channels=64,
        num_blocks=8,
        scale=2,
    ):
        super().__init__()

        # Feature extraction
        self.head = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1,
        )

        # Residual feature learning
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Feature refinement
        self.body = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        # Learned 2x upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * scale * scale,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        )

        # Image reconstruction
        self.tail = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        features = self.head(x)

        residual = features

        features = self.residual_blocks(features)

        features = self.body(features)

        features = features + residual

        features = self.upsample(features)

        output = self.tail(features)

        return output


# ============================================================
# MODEL UTILITIES
# ============================================================

def count_parameters(model):
    """Return number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def extract_state_dict(checkpoint):
    """Extract model weights from the project checkpoint."""

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    raise RuntimeError(
        "Checkpoint does not contain "
        "'model_state_dict' or 'state_dict'."
    )


def load_model(checkpoint_path, device):
    """
    Load SemiconSR and trained checkpoint.

    Direct strict loading is attempted first.

    If checkpoint parameter names differ while the
    tensor shapes and parameter ordering are identical,
    a shape-verified compatibility mapping is used.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{checkpoint_path}\n\n"
            "Expected location:\n"
            f"{DEFAULT_CHECKPOINT}"
        )

    model = SemiconSR()

    parameter_count = count_parameters(model)

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            "\nSemiconSR parameter-count mismatch.\n"
            f"Expected : {EXPECTED_PARAMETERS:,}\n"
            f"Found    : {parameter_count:,}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    checkpoint_state = extract_state_dict(checkpoint)

    # --------------------------------------------------------
    # First attempt: exact checkpoint key matching
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            checkpoint_state,
            strict=True,
        )

    except RuntimeError as direct_error:

        # ----------------------------------------------------
        # Compatibility check for different parameter names
        # ----------------------------------------------------

        model_state = model.state_dict()

        checkpoint_items = list(
            checkpoint_state.items()
        )

        model_items = list(
            model_state.items()
        )

        if len(checkpoint_items) != len(model_items):
            raise RuntimeError(
                "Checkpoint/model parameter count mismatch.\n\n"
                f"Checkpoint tensors: "
                f"{len(checkpoint_items)}\n"
                f"Model tensors: "
                f"{len(model_items)}\n\n"
                "Original checkpoint loading error:\n"
                f"{direct_error}"
            )

        remapped_state = OrderedDict()

        for (
            (checkpoint_name, checkpoint_tensor),
            (model_name, model_tensor),
        ) in zip(
            checkpoint_items,
            model_items,
        ):

            if not isinstance(
                checkpoint_tensor,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "Checkpoint contains a non-tensor "
                    f"parameter: {checkpoint_name}"
                )

            if checkpoint_tensor.shape != model_tensor.shape:
                raise RuntimeError(
                    "\nCheckpoint architecture mismatch.\n"
                    f"Checkpoint parameter: {checkpoint_name}\n"
                    f"Checkpoint shape: "
                    f"{tuple(checkpoint_tensor.shape)}\n"
                    f"Model parameter: {model_name}\n"
                    f"Model shape: "
                    f"{tuple(model_tensor.shape)}"
                )

            remapped_state[model_name] = checkpoint_tensor

        model.load_state_dict(
            remapped_state,
            strict=True,
        )

    model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# INPUT LOADING
# ============================================================

def load_input_image(path):
    """
    Load a grayscale 128x128 NumPy image.
    """

    image = np.load(path)

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # Accept (1,H,W) as well as (H,W).
    if image.ndim == 3:

        if image.shape[0] == 1:
            image = image[0]

        elif image.shape[-1] == 1:
            image = image[..., 0]

        else:
            raise ValueError(
                f"Unsupported image shape "
                f"{image.shape}: {path.name}"
            )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale 2D image, "
            f"received {image.shape}: {path.name}"
        )

    if image.shape != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE[0]}x{INPUT_SIZE[1]}, "
            f"received {image.shape}: {path.name}"
        )

    return image


# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def restore_image(model, image, device):
    """
    Run one image through SemiconSR.
    """

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

    if tuple(restored.shape[-2:]) != OUTPUT_SIZE:
        raise RuntimeError(
            "Unexpected model output size: "
            f"{tuple(restored.shape)}"
        )

    # Convert back from normalized space.
    restored = (
        restored * NORMALIZATION_STD
        + NORMALIZATION_MEAN
    )

    restored = restored.squeeze(
        0
    ).squeeze(
        0
    )

    restored = (
        restored
        .detach()
        .cpu()
        .numpy()
    )

    # Valid image range.
    restored = np.clip(
        restored,
        0.0,
        1.0,
    )

    return restored.astype(np.float32)


# ============================================================
# OUTPUT
# ============================================================

def save_output(restored, output_dir, filename):
    """
    Save restored image as NPY and PNG.
    """

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

    # Numerical output.
    np.save(
        npy_dir / filename,
        restored,
    )

    # 8-bit grayscale visualization.
    png = (
        np.clip(
            restored,
            0.0,
            1.0,
        )
        * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(
        png,
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
            "Standalone SemiconSR test-set inference."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing 128x128 "
            "grayscale .npy test images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where restored images "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to best_psnr.pt. "
            "Default: checkpoints/best_psnr.pt"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate input
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
            "No .npy images found in:\n"
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
        "SEMICON PS01 — TEST INFERENCE"
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
        f"{INPUT_SIZE[0]} x {INPUT_SIZE[1]}"
    )

    print(
        f"Output size     : "
        f"{OUTPUT_SIZE[0]} x {OUTPUT_SIZE[1]}"
    )

    print(
        f"Test images     : {len(input_files)}"
    )

    print(
        "Normalization    : "
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
            f"Validation PSNR : "
            f"{checkpoint['val_psnr']}"
        )

    if "val_ssim" in checkpoint:
        print(
            f"Validation SSIM : "
            f"{checkpoint['val_ssim']}"
        )

    print("=" * 72)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    for index, input_path in enumerate(
        input_files,
        start=1,
    ):

        image = load_input_image(
            input_path
        )

        restored = restore_image(
            model,
            image,
            device,
        )

        save_output(
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

    print()
    print("=" * 72)
    print("INFERENCE COMPLETE")
    print("=" * 72)

    print(
        f"Images processed : {len(input_files)}"
    )

    print(
        f"Total time       : {elapsed:.3f} s"
    )

    print(
        f"Average/image    : "
        f"{elapsed / len(input_files) * 1000:.3f} ms"
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

    print()
    print(
        "Blind-test metrics are not calculated because "
        "ground-truth images are not provided."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
