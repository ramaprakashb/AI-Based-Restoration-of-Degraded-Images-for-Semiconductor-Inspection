"""
SEMICON PS01
Standalone Evaluation Script
AI-Based Restoration of Degraded Images for Semiconductor Inspection

Usage:
    python src/evaluate.py <test_images_directory> <output_directory>

Example:
    python src/evaluate.py ./test_images ./results/restored_test_outputs

Input:
    Directory containing .npy NoisyLR images.
    Expected input shape: 128 x 128

Output:
    Restored .npy images
    Restored .png images

Model:
    SemiconSR
    Parameters: 776,705
    Input:  1 x 128 x 128
    Output: 1 x 256 x 256

Normalization:
    Mean = 0.4335362882
    Std  = 0.2847866113
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

NORMALIZATION_MEAN = 0.4335362882
NORMALIZATION_STD = 0.2847866113

EXPECTED_INPUT_SIZE = (128, 128)
EXPECTED_OUTPUT_SIZE = (256, 256)
EXPECTED_PARAMETERS = 776705

SUPPORTED_EXTENSIONS = {".npy"}


# ============================================================
# SEMICONSR ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual block used by SemiconSR.

    Structure:
        Conv2D -> ReLU -> Conv2D
        + skip connection
    """

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x)


class SemiconSR(nn.Module):
    """
    Lightweight SEM image restoration and 2x
    super-resolution network.

    Input:
        1 x 128 x 128

    Output:
        1 x 256 x 256

    Parameters:
        776,705
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # 1. Feature extraction
        # ----------------------------------------------------
        self.head = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ----------------------------------------------------
        # 2. Residual feature learning
        # ----------------------------------------------------
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(64)
                for _ in range(8)
            ]
        )

        # ----------------------------------------------------
        # 3. Feature refinement
        # ----------------------------------------------------
        self.refine = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ----------------------------------------------------
        # 4. Learned 2x upsampling
        #
        # PixelShuffle(2) requires:
        # 64 * (2^2) = 256 channels
        # ----------------------------------------------------
        self.upsample = nn.Sequential(
            nn.Conv2d(
                64,
                256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(2)
        )

        # ----------------------------------------------------
        # 5. Image reconstruction
        # ----------------------------------------------------
        self.tail = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):

        # Feature extraction
        features = self.head(x)

        # Residual learning
        residual_features = self.residual_blocks(features)

        # Global feature connection
        features = features + residual_features

        # Feature refinement
        features = self.refine(features)

        # Learned 2x upsampling
        features = self.upsample(features)

        # Final image reconstruction
        output = self.tail(features)

        return output


# ============================================================
# PARAMETER VERIFICATION
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def find_checkpoint():

    # Repository root:
    # src/evaluate.py
    # therefore parent.parent = repository root

    script_directory = Path(__file__).resolve().parent
    repository_root = script_directory.parent

    models_directory = repository_root / "models"

    if not models_directory.exists():
        raise FileNotFoundError(
            "\nERROR: models/ directory was not found.\n"
            f"Expected location:\n{models_directory}\n\n"
            "Place the trained SemiconSR checkpoint inside "
            "the repository's models/ directory."
        )

    # Preferred checkpoint names
    preferred_names = [
        "semiconsr_best_psnr.pt",
        "best_psnr.pt",
        "semiconsr_best_psnr.pth",
        "best_psnr.pth",
        "semiconsr.pt",
        "semiconsr.pth",
    ]

    for name in preferred_names:

        candidate = models_directory / name

        if candidate.is_file():
            return candidate

    # Automatic fallback:
    # search recursively for .pt/.pth files

    checkpoints = sorted(
        list(models_directory.rglob("*.pt")) +
        list(models_directory.rglob("*.pth"))
    )

    if len(checkpoints) == 1:
        return checkpoints[0]

    if len(checkpoints) > 1:

        # Prefer files containing "best" and "psnr"
        preferred = [
            p for p in checkpoints
            if "best" in p.name.lower()
            and "psnr" in p.name.lower()
        ]

        if len(preferred) == 1:
            return preferred[0]

        # Prefer files containing "semiconsr"
        semiconsr = [
            p for p in checkpoints
            if "semiconsr" in p.name.lower()
        ]

        if len(semiconsr) == 1:
            return semiconsr[0]

        raise RuntimeError(
            "\nERROR: Multiple model checkpoints were found "
            "but the correct one could not be identified.\n\n"
            "Found checkpoints:\n"
            + "\n".join(str(p) for p in checkpoints)
            + "\n\n"
            "Keep the final SemiconSR checkpoint as "
            "models/semiconsr_best_psnr.pt."
        )

    raise FileNotFoundError(
        "\nERROR: No trained model checkpoint was found.\n\n"
        f"Search directory:\n{models_directory}\n\n"
        "Expected filename:\n"
        "models/semiconsr_best_psnr.pt\n"
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path, device):

    print("=" * 70)
    print("CHECKPOINT")
    print("=" * 70)

    print(f"Checkpoint : {checkpoint_path}")
    print(
        f"Size       : "
        f"{checkpoint_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support both:
    #
    # 1. Complete checkpoint dictionary
    # 2. Raw state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

            print("Checkpoint type : training checkpoint")

            if "epoch" in checkpoint:
                print(
                    f"Training epoch  : "
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

        else:
            # Assume the dictionary itself is a state_dict
            state_dict = checkpoint
            print("Checkpoint type : raw state_dict")

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove DataParallel "module." prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Strict loading guarantees architecture compatibility
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    print("Model state : LOADED SUCCESSFULLY")
    print("Strict load : PASSED")

    return model


# ============================================================
# INPUT DISCOVERY
# ============================================================

def find_input_files(input_directory):

    input_directory = Path(input_directory)

    if not input_directory.exists():
        raise FileNotFoundError(
            f"\nERROR: Input directory does not exist:\n"
            f"{input_directory}"
        )

    if not input_directory.is_dir():
        raise NotADirectoryError(
            f"\nERROR: Input path is not a directory:\n"
            f"{input_directory}"
        )

    files = sorted(
        [
            p
            for p in input_directory.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not files:
        raise FileNotFoundError(
            "\nERROR: No .npy test images were found in:\n"
            f"{input_directory}"
        )

    return files


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_input(npy_path):

    image = np.load(npy_path)

    # Convert to float32
    image = image.astype(np.float32)

    # --------------------------------------------------------
    # Accept common possible shapes:
    #
    # (128,128)
    # (1,128,128)
    # (128,128,1)
    # --------------------------------------------------------

    if image.ndim == 2:

        if image.shape != EXPECTED_INPUT_SIZE:
            raise ValueError(
                f"Invalid input shape for {npy_path.name}: "
                f"{image.shape}. "
                f"Expected {EXPECTED_INPUT_SIZE}."
            )

        image = image[np.newaxis, :, :]

    elif image.ndim == 3:

        if image.shape == (1, 128, 128):

            pass

        elif image.shape == (128, 128, 1):

            image = np.transpose(
                image,
                (2, 0, 1)
            )

        else:
            raise ValueError(
                f"Invalid input shape for {npy_path.name}: "
                f"{image.shape}. "
                "Expected (128,128), "
                "(1,128,128), or (128,128,1)."
            )

    else:

        raise ValueError(
            f"Unsupported input dimensions for "
            f"{npy_path.name}: {image.ndim}"
        )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image = torch.from_numpy(image)

    image = image.unsqueeze(0)

    return image


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_outputs(
    restored,
    input_path,
    output_npy_directory,
    output_png_directory
):

    # Convert tensor to numpy
    restored = restored.detach().cpu().numpy()

    # Shape:
    # (1, 1, 256, 256)

    restored = restored[0, 0]

    # Keep numerical output in [0,1]
    restored = np.clip(
        restored,
        0.0,
        1.0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Preserve original filename
    # --------------------------------------------------------

    filename = input_path.stem

    # --------------------------------------------------------
    # Save NPY
    # --------------------------------------------------------

    npy_output_path = (
        output_npy_directory /
        f"{filename}.npy"
    )

    np.save(
        npy_output_path,
        restored
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    png_image = np.round(
        restored * 255.0
    ).astype(np.uint8)

    png_output_path = (
        output_png_directory /
        f"{filename}.png"
    )

    Image.fromarray(
        png_image,
        mode="L"
    ).save(
        png_output_path
    )

    return npy_output_path, png_output_path


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Standalone SemiconSR evaluation script "
            "for SEM image restoration."
        )
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing test .npy images."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where restored outputs will be written."
    )

    args = parser.parse_args()

    input_directory = Path(
        args.input_dir
    ).resolve()

    output_directory = Path(
        args.output_dir
    ).resolve()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("SEMICON PS01 — STANDALONE EVALUATION")
    print("=" * 70)

    print(f"Input directory  : {input_directory}")
    print(f"Output directory : {output_directory}")
    print(f"Device           : {device}")

    if torch.cuda.is_available():

        print(
            f"GPU              : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Normalization    : "
        f"mean={NORMALIZATION_MEAN}, "
        f"std={NORMALIZATION_STD}"
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_npy_directory = (
        output_directory / "restored_npy"
    )

    output_png_directory = (
        output_directory / "restored_png"
    )

    output_npy_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_png_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find test files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    input_files = find_input_files(
        input_directory
    )

    print(
        f"Test images found : {len(input_files)}"
    )

    # --------------------------------------------------------
    # Find checkpoint
    # --------------------------------------------------------

    checkpoint_path = find_checkpoint()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    model = SemiconSR()

    parameter_count = count_parameters(
        model
    )

    print(
        f"Architecture : SemiconSR"
    )

    print(
        f"Parameters   : {parameter_count:,}"
    )

    if parameter_count != EXPECTED_PARAMETERS:

        raise RuntimeError(
            "\nERROR: Architecture parameter count mismatch.\n"
            f"Expected : {EXPECTED_PARAMETERS:,}\n"
            f"Actual   : {parameter_count:,}\n"
        )

    print(
        "Architecture verification : PASSED"
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INFERENCE")
    print("=" * 70)

    print(
        f"Total images : {len(input_files)}"
    )

    print(
        "PSNR / SSIM / LPIPS : "
        "NOT calculated because this standalone "
        "evaluation accepts test images only."
    )

    print()

    start_time = time.perf_counter()

    successful = 0
    failed = 0

    with torch.inference_mode():

        for index, input_path in enumerate(
            input_files,
            start=1
        ):

            try:

                # --------------------------------------------
                # Load input
                # --------------------------------------------

                input_tensor = prepare_input(
                    input_path
                )

                input_tensor = input_tensor.to(
                    device,
                    non_blocking=True
                )

                # --------------------------------------------
                # Same normalization used during training
                # --------------------------------------------

                input_tensor = (
                    input_tensor -
                    NORMALIZATION_MEAN
                ) / NORMALIZATION_STD

                # --------------------------------------------
                # Model inference
                # --------------------------------------------

                restored = model(
                    input_tensor
                )

                # --------------------------------------------
                # Verify output
                # --------------------------------------------

                if restored.ndim != 4:
                    raise RuntimeError(
                        f"Unexpected model output dimensions: "
                        f"{restored.shape}"
                    )

                if tuple(
                    restored.shape[-2:]
                ) != EXPECTED_OUTPUT_SIZE:

                    raise RuntimeError(
                        f"Unexpected output size: "
                        f"{tuple(restored.shape[-2:])}. "
                        f"Expected {EXPECTED_OUTPUT_SIZE}."
                    )

                # --------------------------------------------
                # Save
                # --------------------------------------------

                save_outputs(
                    restored,
                    input_path,
                    output_npy_directory,
                    output_png_directory
                )

                successful += 1

                # --------------------------------------------
                # Progress
                # --------------------------------------------

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
                        successful / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining = (
                        len(input_files)
                        - index
                    )

                    eta = (
                        remaining / speed
                        if speed > 0
                        else 0
                    )

                    print(
                        f"[{index:04d}/"
                        f"{len(input_files):04d}] "
                        f"{input_path.name} | "
                        f"{speed:.2f} img/s | "
                        f"ETA {eta:.1f}s"
                    )

            except Exception as error:

                failed += 1

                print()
                print(
                    f"ERROR processing "
                    f"{input_path}:"
                )
                print(
                    f"    {error}"
                )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    average_time = (
        total_time / successful
        if successful > 0
        else 0
    )

    speed = (
        successful / total_time
        if total_time > 0
        else 0
    )

    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    summary_path = (
        output_directory /
        "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SEMICON PS01 — EVALUATION SUMMARY\n"
        )
        file.write(
            "=================================\n\n"
        )

        file.write(
            f"Input directory: "
            f"{input_directory}\n"
        )

        file.write(
            f"Output directory: "
            f"{output_directory}\n\n"
        )

        file.write(
            "MODEL\n"
        )
        file.write(
            "-----\n"
        )

        file.write(
            "Architecture: SemiconSR\n"
        )

        file.write(
            f"Parameters: "
            f"{parameter_count:,}\n"
        )

        file.write(
            f"Checkpoint: "
            f"{checkpoint_path}\n\n"
        )

        file.write(
            "NORMALIZATION\n"
        )
        file.write(
            "-------------\n"
        )

        file.write(
            f"Mean: "
            f"{NORMALIZATION_MEAN}\n"
        )

        file.write(
            f"Std: "
            f"{NORMALIZATION_STD}\n\n"
        )

        file.write(
            "INFERENCE\n"
        )
        file.write(
            "---------\n"
        )

        file.write(
            f"Total input images: "
            f"{len(input_files)}\n"
        )

        file.write(
            f"Successful: "
            f"{successful}\n"
        )

        file.write(
            f"Failed: "
            f"{failed}\n"
        )

        file.write(
            f"Total inference time: "
            f"{total_time:.4f} seconds\n"
        )

        file.write(
            f"Average time/image: "
            f"{average_time:.6f} seconds\n"
        )

        file.write(
            f"Throughput: "
            f"{speed:.4f} images/second\n\n"
        )

        file.write(
            "OUTPUT\n"
        )
        file.write(
            "------\n"
        )

        file.write(
            f"Restored NPY: "
            f"{output_npy_directory}\n"
        )

        file.write(
            f"Restored PNG: "
            f"{output_png_directory}\n"
        )

        file.write(
            "\nPSNR/SSIM/LPIPS were not calculated "
            "because this standalone evaluator "
            "does not require ground-truth images.\n"
        )

    # --------------------------------------------------------
    # Final console report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Input images       : {len(input_files)}"
    )

    print(
        f"Successfully       : {successful}"
    )

    print(
        f"Failed             : {failed}"
    )

    print(
        f"Total time         : "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Average/image      : "
        f"{average_time:.6f} seconds"
    )

    print(
        f"Throughput         : "
        f"{speed:.2f} images/second"
    )

    print()
    print(
        f"Restored NPY       : "
        f"{output_npy_directory}"
    )

    print(
        f"Restored PNG       : "
        f"{output_png_directory}"
    )

    print(
        f"Summary            : "
        f"{summary_path}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Exit with failure status if any image failed
    # --------------------------------------------------------

    if failed > 0:

        print()
        print(
            "WARNING: Some test images failed."
        )

        sys.exit(1)

    print()
    print(
        "ALL TEST IMAGES PROCESSED SUCCESSFULLY."
    )

    sys.exit(0)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
