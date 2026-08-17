#!/usr/bin/env python3
"""
SEMICON PS01 — FINAL 7B-3 CONTROLLED 100-EPOCH TRAINING

GitHub-ready training script.

Expected dataset:

    <project-root>/
    ├── train/
    │   ├── GT/
    │   │   ├── sample_0001.npy
    │   │   └── ...
    │   └── NoisyLR/
    │       ├── sample_0001.npy
    │       └── ...
    │
    └── train_model.py

Training design:
    - 3200 paired samples
    - 2560 training / 640 validation
    - fixed seed = 42
    - 128x128 -> 256x256
    - grayscale, single channel
    - exact 776,705-parameter 7B-3 model
    - verified normalization
    - Charbonnier + SSIM + Edge + small HF loss
    - geometric augmentation only
    - AdamW
    - 5 epoch warm-up
    - cosine LR decay
    - 100 epochs
    - AMP on CUDA
    - gradient clipping
    - checkpoint/resume support
    - metrics CSV
    - final configuration JSON

Run from the project root:

    python train_model.py

Or:

    python train_model.py --project-root /path/to/SEMICON_PS01

Resume:

    python train_model.py \
        --project-root /path/to/SEMICON_PS01 \
        --resume

Generate final validation comparisons:

    python train_model.py \
        --project-root /path/to/SEMICON_PS01 \
        --make-comparisons
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ============================================================
# 1. FIXED EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

TARGET_EPOCHS = 100

BATCH_SIZE = 16

NUM_FEATURES = 64

NUM_BLOCKS = 8

SCALE = 2

INITIAL_LR = 1e-4

MIN_LR = 1e-6

WARMUP_EPOCHS = 5

EXPECTED_PAIRS = 3200

TRAIN_COUNT = 2560

VAL_COUNT = 640

NORMALIZATION_MEAN = 0.4335362882

NORMALIZATION_STD = 0.2847866113

CHARBONNIER_WEIGHT = 1.00

SSIM_WEIGHT = 0.15

EDGE_WEIGHT = 0.08

SHARPNESS_WEIGHT = 0.02

CHECKPOINT_EVERY = 5

NUM_COMPARISON_IMAGES = 20

EXPECTED_PARAMETER_COUNT = 776_705


# ============================================================
# 2. ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SEMICON PS01 final 7B-3 training"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help=(
            "Project root containing "
            "train/GT and train/NoisyLR."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Training output directory. "
            "Default: "
            "<project-root>/FINAL_7B3_CONTROLLED_100EPOCH"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from latest_checkpoint.pt "
            "if available."
        ),
    )

    parser.add_argument(
        "--make-comparisons",
        action="store_true",
        help=(
            "Generate deterministic validation "
            "comparison images after training."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker count. "
            "Default: 2 on CUDA, 0 on CPU."
        ),
    )

    return parser.parse_args()


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. DATASET
# ============================================================

class SemiconDataset(Dataset):

    def __init__(
        self,
        noisy_dir,
        gt_dir,
        ids,
        training=False,
    ):

        self.noisy_dir = Path(noisy_dir)

        self.gt_dir = Path(gt_dir)

        self.ids = list(ids)

        self.training = training


    def __len__(self):

        return len(self.ids)


    def __getitem__(self, index):

        sample_id = self.ids[index]

        noisy_path = (
            self.noisy_dir /
            f"{sample_id}.npy"
        )

        gt_path = (
            self.gt_dir /
            f"{sample_id}.npy"
        )

        noisy = np.load(
            noisy_path
        ).astype(
            np.float32
        )

        gt = np.load(
            gt_path
        ).astype(
            np.float32
        )


        if noisy.shape != (128, 128):

            raise RuntimeError(
                f"Invalid NoisyLR shape "
                f"for {sample_id}: "
                f"{noisy.shape}. "
                f"Expected (128, 128)."
            )


        if gt.shape != (256, 256):

            raise RuntimeError(
                f"Invalid GT shape "
                f"for {sample_id}: "
                f"{gt.shape}. "
                f"Expected (256, 256)."
            )


        if not np.isfinite(noisy).all():

            raise RuntimeError(
                f"NaN/Inf detected in "
                f"NoisyLR: {sample_id}"
            )


        if not np.isfinite(gt).all():

            raise RuntimeError(
                f"NaN/Inf detected in "
                f"GT: {sample_id}"
            )


        noisy = torch.from_numpy(
            noisy
        ).unsqueeze(0)


        gt = torch.from_numpy(
            gt
        ).unsqueeze(0)


        # ----------------------------------------------------
        # GT physical range
        # ----------------------------------------------------

        gt = torch.clamp(
            gt,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # GEOMETRIC AUGMENTATION ONLY
        # ----------------------------------------------------
        #
        # No:
        #   brightness changes
        #   contrast changes
        #   gamma
        #   histogram equalization
        #   CLAHE
        #   artificial noise
        #   artificial blur
        #   artificial sharpening
        #
        # ----------------------------------------------------

        if self.training:

            if random.random() < 0.5:

                noisy = torch.flip(
                    noisy,
                    dims=[-1]
                )

                gt = torch.flip(
                    gt,
                    dims=[-1]
                )


            if random.random() < 0.5:

                noisy = torch.flip(
                    noisy,
                    dims=[-2]
                )

                gt = torch.flip(
                    gt,
                    dims=[-2]
                )


            k = random.randint(
                0,
                3
            )


            if k:

                noisy = torch.rot90(
                    noisy,
                    k,
                    dims=[-2, -1]
                )

                gt = torch.rot90(
                    gt,
                    k,
                    dims=[-2, -1]
                )


        # ----------------------------------------------------
        # VERIFIED 7B-3 NORMALIZATION
        # ----------------------------------------------------

        noisy = (
            noisy -
            NORMALIZATION_MEAN
        ) / NORMALIZATION_STD


        return (
            noisy,
            gt,
            sample_id
        )


# ============================================================
# 5. EXACT 7B-3 ARCHITECTURE
# ============================================================

class BasicResidualBlock(
    nn.Module
):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )


    def forward(
        self,
        x
    ):

        return (
            x +
            self.block(x)
        )


class Semicon7B3(
    nn.Module
):

    def __init__(
        self,
        num_features=64,
        num_blocks=8,
        scale=2
    ):

        super().__init__()


        self.head = nn.Conv2d(
            1,
            num_features,
            kernel_size=3,
            padding=1
        )


        self.body = nn.Sequential(

            *[
                BasicResidualBlock(
                    num_features
                )

                for _ in range(
                    num_blocks
                )
            ]
        )


        self.body_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )


        self.upsample = nn.Sequential(

            nn.Conv2d(
                num_features,
                num_features * (
                    scale ** 2
                ),
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(
                scale
            )
        )


        self.tail = nn.Conv2d(
            num_features,
            1,
            kernel_size=3,
            padding=1
        )


    def forward(
        self,
        x
    ):

        shallow = self.head(
            x
        )


        body = self.body(
            shallow
        )


        body = self.body_conv(
            body
        )


        body = (
            body +
            shallow
        )


        body = self.upsample(
            body
        )


        output = self.tail(
            body
        )


        return output


# ============================================================
# 6. CHARBONNIER LOSS
# ============================================================

class CharbonnierLoss(
    nn.Module
):

    def __init__(
        self,
        eps=1e-3
    ):

        super().__init__()

        self.eps = eps


    def forward(
        self,
        prediction,
        target
    ):

        diff = (
            prediction -
            target
        )


        return torch.mean(
            torch.sqrt(
                diff * diff +
                self.eps ** 2
            )
        )


charbonnier_loss = (
    CharbonnierLoss()
)


# ============================================================
# 7. SSIM
# ============================================================

def create_gaussian_window(
    size,
    sigma,
    device
):

    coords = torch.arange(
        size,
        dtype=torch.float32,
        device=device
    )


    coords -= (
        size // 2
    )


    gaussian = torch.exp(
        -(
            coords ** 2
        )
        /
        (
            2.0 *
            sigma ** 2
        )
    )


    gaussian /= (
        gaussian.sum()
    )


    window = torch.outer(
        gaussian,
        gaussian
    )


    return window.view(
        1,
        1,
        size,
        size
    )


def ssim_value(
    prediction,
    target
):

    prediction = prediction.float()

    target = target.float()


    window = create_gaussian_window(
        11,
        1.5,
        prediction.device
    )


    mu_x = F.conv2d(
        prediction,
        window,
        padding=5
    )


    mu_y = F.conv2d(
        target,
        window,
        padding=5
    )


    mu_x2 = mu_x * mu_x

    mu_y2 = mu_y * mu_y

    mu_xy = mu_x * mu_y


    sigma_x2 = (
        F.conv2d(
            prediction * prediction,
            window,
            padding=5
        )
        -
        mu_x2
    )


    sigma_y2 = (
        F.conv2d(
            target * target,
            window,
            padding=5
        )
        -
        mu_y2
    )


    sigma_xy = (
        F.conv2d(
            prediction * target,
            window,
            padding=5
        )
        -
        mu_xy
    )


    sigma_x2 = torch.clamp(
        sigma_x2,
        min=0.0
    )


    sigma_y2 = torch.clamp(
        sigma_y2,
        min=0.0
    )


    C1 = (
        0.01 ** 2
    )


    C2 = (
        0.03 ** 2
    )


    numerator = (

        (
            2.0 *
            mu_xy +
            C1
        )

        *

        (
            2.0 *
            sigma_xy +
            C2
        )
    )


    denominator = (

        (
            mu_x2 +
            mu_y2 +
            C1
        )

        *

        (
            sigma_x2 +
            sigma_y2 +
            C2
        )
    )


    denominator = torch.clamp(
        denominator,
        min=1e-12
    )


    value = (
        numerator /
        denominator
    )


    return torch.clamp(
        value,
        -1.0,
        1.0
    ).mean()


# ============================================================
# 8. SOBEL EDGE LOSS
# ============================================================

SOBEL_X = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],
    dtype=torch.float32
).view(
    1,
    1,
    3,
    3
)


SOBEL_Y = torch.tensor(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],
    dtype=torch.float32
).view(
    1,
    1,
    3,
    3
)


def edge_loss(
    prediction,
    target
):

    device = (
        prediction.device
    )


    sobel_x = SOBEL_X.to(
        device
    )


    sobel_y = SOBEL_Y.to(
        device
    )


    px = F.conv2d(
        prediction,
        sobel_x,
        padding=1
    )


    py = F.conv2d(
        prediction,
        sobel_y,
        padding=1
    )


    tx = F.conv2d(
        target,
        sobel_x,
        padding=1
    )


    ty = F.conv2d(
        target,
        sobel_y,
        padding=1
    )


    return (

        F.l1_loss(
            px,
            tx
        )

        +

        F.l1_loss(
            py,
            ty
        )
    )


# ============================================================
# 9. HIGH-FREQUENCY LOSS
# ============================================================

def high_frequency_loss(
    prediction,
    target
):

    prediction = (
        prediction.float()
    )


    target = (
        target.float()
    )


    low_pred = F.avg_pool2d(
        prediction,
        kernel_size=3,
        stride=1,
        padding=1
    )


    low_target = F.avg_pool2d(
        target,
        kernel_size=3,
        stride=1,
        padding=1
    )


    high_pred = (
        prediction -
        low_pred
    )


    high_target = (
        target -
        low_target
    )


    return F.l1_loss(
        high_pred,
        high_target
    )


# ============================================================
# 10. COMBINED LOSS
# ============================================================

def combined_loss(
    prediction,
    target
):

    prediction = (
        prediction.float()
    )


    target = (
        target.float()
    )


    prediction = torch.clamp(
        prediction,
        0.0,
        1.0
    )


    loss_char = (
        charbonnier_loss(
            prediction,
            target
        )
    )


    loss_ssim = (
        1.0 -
        ssim_value(
            prediction,
            target
        )
    )


    loss_edge = (
        edge_loss(
            prediction,
            target
        )
    )


    loss_hf = (
        high_frequency_loss(
            prediction,
            target
        )
    )


    total = (

        CHARBONNIER_WEIGHT *
        loss_char

        +

        SSIM_WEIGHT *
        loss_ssim

        +

        EDGE_WEIGHT *
        loss_edge

        +

        SHARPNESS_WEIGHT *
        loss_hf
    )


    return total


# ============================================================
# 11. METRICS
# ============================================================

@torch.no_grad()
def calculate_batch_metrics(
    prediction,
    target
):

    prediction = torch.clamp(
        prediction.float(),
        0.0,
        1.0
    )


    target = (
        target.float()
    )


    prediction_np = (
        prediction
        .detach()
        .cpu()
        .numpy()
    )


    target_np = (
        target
        .detach()
        .cpu()
        .numpy()
    )


    psnr_values = []

    ssim_values = []


    for p, t in zip(
        prediction_np,
        target_np
    ):

        p = p[0]

        t = t[0]


        psnr_values.append(
            peak_signal_noise_ratio(
                t,
                p,
                data_range=1.0
            )
        )


        ssim_values.append(
            structural_similarity(
                t,
                p,
                data_range=1.0
            )
        )


    return (

        float(
            np.mean(
                psnr_values
            )
        ),

        float(
            np.mean(
                ssim_values
            )
        )
    )


# ============================================================
# 12. LEARNING RATE
# ============================================================

def get_learning_rate(
    epoch_zero_based
):

    epoch_number = (
        epoch_zero_based +
        1
    )


    if (
        epoch_number <=
        WARMUP_EPOCHS
    ):

        progress = (
            epoch_number /
            WARMUP_EPOCHS
        )


        return (
            INITIAL_LR *
            (
                0.5 +
                0.5 * progress
            )
        )


    progress = (

        epoch_number -
        WARMUP_EPOCHS

    ) / (

        TARGET_EPOCHS -
        WARMUP_EPOCHS
    )


    progress = min(
        max(
            progress,
            0.0
        ),
        1.0
    )


    cosine = (
        0.5 *
        (
            1.0 +
            math.cos(
                math.pi *
                progress
            )
        )
    )


    return (

        MIN_LR

        +

        (
            INITIAL_LR -
            MIN_LR
        )

        *

        cosine
    )


# ============================================================
# 13. DATASET VERIFICATION
# ============================================================

def verify_dataset(
    project_root,
    gt_dir,
    noisy_dir
):

    if not project_root.exists():

        raise FileNotFoundError(
            f"Project directory not found:\n"
            f"{project_root}"
        )


    if not gt_dir.exists():

        raise FileNotFoundError(
            f"GT directory not found:\n"
            f"{gt_dir}"
        )


    if not noisy_dir.exists():

        raise FileNotFoundError(
            f"NoisyLR directory not found:\n"
            f"{noisy_dir}"
        )


    gt_files = sorted(
        gt_dir.glob("*.npy")
    )


    noisy_files = sorted(
        noisy_dir.glob("*.npy")
    )


    gt_ids = {
        p.stem
        for p in gt_files
    }


    noisy_ids = {
        p.stem
        for p in noisy_files
    }


    common_ids = sorted(
        gt_ids.intersection(
            noisy_ids
        )
    )


    missing_gt = sorted(
        noisy_ids -
        gt_ids
    )


    missing_noisy = sorted(
        gt_ids -
        noisy_ids
    )


    print()
    print("=" * 72)
    print("DATASET VERIFICATION")
    print("=" * 72)

    print(
        "GT files       :",
        len(gt_files)
    )

    print(
        "NoisyLR files  :",
        len(noisy_files)
    )

    print(
        "Matched        :",
        len(common_ids)
    )

    print(
        "Missing GT     :",
        len(missing_gt)
    )

    print(
        "Missing Noisy  :",
        len(missing_noisy)
    )


    if len(common_ids) != (
        EXPECTED_PAIRS
    ):

        raise RuntimeError(
            "\nExpected exactly "
            f"{EXPECTED_PAIRS} paired samples.\n"
            f"Found {len(common_ids)}."
        )


    if missing_gt or missing_noisy:

        raise RuntimeError(
            "\nUnmatched dataset files "
            "detected.\n"
            "Training stopped."
        )


    sample_id = (
        common_ids[0]
    )


    sample_noisy = np.load(
        noisy_dir /
        f"{sample_id}.npy"
    ).astype(
        np.float32
    )


    sample_gt = np.load(
        gt_dir /
        f"{sample_id}.npy"
    ).astype(
        np.float32
    )


    print()
    print("=" * 72)
    print("SAMPLE AUDIT")
    print("=" * 72)

    print(
        "Sample        :",
        sample_id
    )

    print(
        "NoisyLR shape :",
        sample_noisy.shape
    )

    print(
        "GT shape      :",
        sample_gt.shape
    )

    print(
        "NoisyLR min   :",
        float(sample_noisy.min())
    )

    print(
        "NoisyLR max   :",
        float(sample_noisy.max())
    )

    print(
        "NoisyLR mean  :",
        float(sample_noisy.mean())
    )

    print(
        "NoisyLR std   :",
        float(sample_noisy.std())
    )

    print(
        "GT min        :",
        float(sample_gt.min())
    )

    print(
        "GT max        :",
        float(sample_gt.max())
    )

    print(
        "GT mean       :",
        float(sample_gt.mean())
    )

    print(
        "GT std        :",
        float(sample_gt.std())
    )


    if sample_noisy.shape != (
        128,
        128
    ):

        raise RuntimeError(
            "NoisyLR shape is not 128x128."
        )


    if sample_gt.shape != (
        256,
        256
    ):

        raise RuntimeError(
            "GT shape is not 256x256."
        )


    print(
        "OK: EXACT 3200-PAIR DATASET VERIFIED"
    )


    return common_ids


# ============================================================
# 14. FIXED SPLIT
# ============================================================

def make_split(
    common_ids
):

    shuffled_ids = list(
        common_ids
    )


    rng = random.Random(
        SEED
    )


    rng.shuffle(
        shuffled_ids
    )


    train_ids = (
        shuffled_ids[
            :TRAIN_COUNT
        ]
    )


    val_ids = (
        shuffled_ids[
            TRAIN_COUNT:
        ]
    )


    if len(train_ids) != (
        TRAIN_COUNT
    ):

        raise RuntimeError(
            "Training split is not "
            "exactly 2560."
        )


    if len(val_ids) != (
        VAL_COUNT
    ):

        raise RuntimeError(
            "Validation split is not "
            "exactly 640."
        )


    overlap = (
        set(train_ids)
        .intersection(
            set(val_ids)
        )
    )


    if overlap:

        raise RuntimeError(
            "Train/validation overlap detected."
        )


    return (
        train_ids,
        val_ids
    )


# ============================================================
# 15. CONFIGURATION
# ============================================================

def model_config():

    return {

        "architecture":
            "Verified 7B-3",

        "num_features":
            NUM_FEATURES,

        "num_blocks":
            NUM_BLOCKS,

        "scale":
            SCALE,

        "parameters":
            EXPECTED_PARAMETER_COUNT,

        "input_channels":
            1,

        "normalization_mean":
            NORMALIZATION_MEAN,

        "normalization_std":
            NORMALIZATION_STD,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "target_epochs":
            TARGET_EPOCHS,

        "initial_lr":
            INITIAL_LR,

        "minimum_lr":
            MIN_LR,

        "warmup_epochs":
            WARMUP_EPOCHS,

        "charbonnier_weight":
            CHARBONNIER_WEIGHT,

        "ssim_weight":
            SSIM_WEIGHT,

        "edge_weight":
            EDGE_WEIGHT,

        "sharpness_weight":
            SHARPNESS_WEIGHT
    }


# ============================================================
# 16. SAVE BEST CHECKPOINT
# ============================================================

def save_best_checkpoint(
    path,
    model,
    epoch,
    val_psnr,
    val_ssim,
    balanced_score=None
):

    payload = {

        "model_state_dict":
            model.state_dict(),

        "epoch":
            epoch,

        "val_psnr":
            val_psnr,

        "val_ssim":
            val_ssim,

        "parameters":
            EXPECTED_PARAMETER_COUNT,

        "config":
            model_config()
    }


    if balanced_score is not None:

        payload[
            "balanced_score"
        ] = balanced_score


    torch.save(
        payload,
        path
    )


# ============================================================
# 17. TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None
):

    training = (
        optimizer is not None
    )


    if training:

        model.train()

    else:

        model.eval()


    total_loss = 0.0

    total_psnr = 0.0

    total_ssim = 0.0

    batches = 0


    for noisy, target, _ in loader:

        noisy = noisy.to(
            device,
            non_blocking=True
        )


        target = target.to(
            device,
            non_blocking=True
        )


        if training:

            optimizer.zero_grad(
                set_to_none=True
            )


            if (
                device.type ==
                "cuda"
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    raw_prediction = (
                        model(noisy)
                    )


                loss = combined_loss(
                    raw_prediction,
                    target
                )


                if not torch.isfinite(
                    loss
                ):

                    raise RuntimeError(
                        "Non-finite training loss."
                    )


                scaler.scale(
                    loss
                ).backward()


                scaler.unscale_(
                    optimizer
                )


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )


                scaler.step(
                    optimizer
                )


                scaler.update()


            else:

                raw_prediction = (
                    model(noisy)
                )


                loss = combined_loss(
                    raw_prediction,
                    target
                )


                if not torch.isfinite(
                    loss
                ):

                    raise RuntimeError(
                        "Non-finite training loss."
                    )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )


                optimizer.step()


        else:

            with torch.no_grad():

                raw_prediction = (
                    model(noisy)
                )


                loss = combined_loss(
                    raw_prediction,
                    target
                )


        prediction = torch.clamp(
            raw_prediction.float(),
            0.0,
            1.0
        )


        psnr, ssim = (
            calculate_batch_metrics(
                prediction,
                target
            )
        )


        total_loss += float(
            loss.detach().item()
        )


        total_psnr += psnr

        total_ssim += ssim

        batches += 1


    if batches == 0:

        raise RuntimeError(
            "DataLoader produced zero batches."
        )


    return (

        total_loss /
        batches,

        total_psnr /
        batches,

        total_ssim /
        batches
    )


# ============================================================
# 18. OPTIONAL COMPARISON IMAGES
# ============================================================

@torch.no_grad()
def generate_comparisons(
    model,
    device,
    noisy_dir,
    gt_dir,
    val_ids,
    comparison_dir
):

    import matplotlib.pyplot as plt


    comparison_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model.eval()


    comparison_ids = (
        val_ids[
            :NUM_COMPARISON_IMAGES
        ]
    )


    for index, sample_id in enumerate(
        comparison_ids,
        start=1
    ):

        noisy = np.load(
            noisy_dir /
            f"{sample_id}.npy"
        ).astype(
            np.float32
        )


        gt = np.load(
            gt_dir /
            f"{sample_id}.npy"
        ).astype(
            np.float32
        )


        noisy_tensor = (
            torch.from_numpy(
                noisy
            )
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )


        normalized = (
            noisy_tensor -
            NORMALIZATION_MEAN
        ) / NORMALIZATION_STD


        restored = model(
            normalized
        )


        restored = torch.clamp(
            restored.float(),
            0.0,
            1.0
        )


        restored = (
            restored
            .squeeze()
            .cpu()
            .numpy()
        )


        noisy_upscaled = (
            F.interpolate(
                torch.from_numpy(
                    noisy
                )
                .float()
                .unsqueeze(0)
                .unsqueeze(0),
                size=(256, 256),
                mode="bicubic",
                align_corners=False
            )
            .squeeze()
            .numpy()
        )


        noisy_display = np.clip(
            noisy_upscaled,
            0.0,
            1.0
        )


        gt_display = np.clip(
            gt,
            0.0,
            1.0
        )


        fig = plt.figure(
            figsize=(15, 5)
        )


        ax1 = fig.add_subplot(
            1,
            3,
            1
        )


        ax1.imshow(
            noisy_display,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        ax1.set_title(
            "NoisyLR / Bicubic"
        )


        ax1.axis(
            "off"
        )


        ax2 = fig.add_subplot(
            1,
            3,
            2
        )


        ax2.imshow(
            restored,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        ax2.set_title(
            "7B-3 Restored"
        )


        ax2.axis(
            "off"
        )


        ax3 = fig.add_subplot(
            1,
            3,
            3
        )


        ax3.imshow(
            gt_display,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        ax3.set_title(
            "GT"
        )


        ax3.axis(
            "off"
        )


        fig.suptitle(
            f"SEMICON PS01 — {sample_id}"
        )


        output_path = (
            comparison_dir /
            f"sample_{index:02d}_{sample_id}.png"
        )


        plt.tight_layout()


        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


        print(
            "Comparison:",
            output_path
        )


# ============================================================
# 19. MAIN
# ============================================================

def main():

    args = parse_args()


    project_root = (
        args.project_root
        .resolve()
    )


    if args.output_dir is None:

        run_dir = (
            project_root /
            "FINAL_7B3_CONTROLLED_100EPOCH"
        )

    else:

        run_dir = (
            args.output_dir
            .resolve()
        )


    gt_dir = (
        project_root /
        "train" /
        "GT"
    )


    noisy_dir = (
        project_root /
        "train" /
        "NoisyLR"
    )


    checkpoint_dir = (
        run_dir /
        "checkpoints"
    )


    metrics_dir = (
        run_dir /
        "metrics"
    )


    comparison_dir = (
        run_dir /
        "comparison_images"
    )


    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if args.make_comparisons:

        comparison_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    seed_everything(
        SEED
    )


    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device(
            "cpu"
        )

        gpu_name = "CPU"


    print()
    print("=" * 72)
    print(
        "SEMICON PS01 — "
        "FINAL 7B-3 CONTROLLED "
        "100-EPOCH TRAINING"
    )
    print("=" * 72)


    print(
        "Project root    :",
        project_root
    )


    print(
        "Run directory   :",
        run_dir
    )


    print(
        "GT directory    :",
        gt_dir
    )


    print(
        "NoisyLR         :",
        noisy_dir
    )


    print(
        "Device          :",
        device
    )


    print(
        "GPU / CPU       :",
        gpu_name
    )


    print(
        "Seed            :",
        SEED
    )


    print(
        "Epochs          :",
        TARGET_EPOCHS
    )


    print(
        "Batch size      :",
        BATCH_SIZE
    )


    print(
        "Initial LR      :",
        INITIAL_LR
    )


    print(
        "Minimum LR      :",
        MIN_LR
    )


    print(
        "Warm-up epochs  :",
        WARMUP_EPOCHS
    )


    print(
        "Normalization μ :",
        NORMALIZATION_MEAN
    )


    print(
        "Normalization σ :",
        NORMALIZATION_STD
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    common_ids = verify_dataset(
        project_root,
        gt_dir,
        noisy_dir
    )


    train_ids, val_ids = (
        make_split(
            common_ids
        )
    )


    print()
    print("=" * 72)
    print("FIXED DATASET SPLIT")
    print("=" * 72)


    print(
        "Train     :",
        len(train_ids)
    )


    print(
        "Validation:",
        len(val_ids)
    )


    print(
        "Overlap   :",
        0
    )


    with open(
        run_dir /
        "dataset_split.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "seed":
                    SEED,

                "train_count":
                    len(train_ids),

                "val_count":
                    len(val_ids),

                "train_ids":
                    train_ids,

                "val_ids":
                    val_ids
            },
            f,
            indent=2
        )


    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_dataset = SemiconDataset(
        noisy_dir,
        gt_dir,
        train_ids,
        training=True
    )


    val_dataset = SemiconDataset(
        noisy_dir,
        gt_dir,
        val_ids,
        training=False
    )


    if args.num_workers is not None:

        num_workers = max(
            0,
            args.num_workers
        )

    else:

        num_workers = (
            2
            if device.type == "cuda"
            else 0
        )


    pin_memory = (
        device.type == "cuda"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        )
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        )
    )


    print()
    print(
        "Train batches:",
        len(train_loader)
    )


    print(
        "Val batches  :",
        len(val_loader)
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Semicon7B3().to(
        device
    )


    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print()
    print("=" * 72)
    print("MODEL VERIFICATION")
    print("=" * 72)


    print(
        "Parameters:",
        f"{parameter_count:,}"
    )


    print(
        "Trainable :",
        f"{trainable_count:,}"
    )


    if parameter_count != (
        EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(

            "\nARCHITECTURE "
            "VERIFICATION FAILED\n"

            f"Expected: "
            f"{EXPECTED_PARAMETER_COUNT:,}\n"

            f"Current : "
            f"{parameter_count:,}\n"

            "Training stopped."
        )


    print(
        "OK: EXACT 776,705-PARAMETER "
        "7B-3 ARCHITECTURE"
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=1e-6
    )


    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    if device.type == "cuda":

        try:

            scaler = torch.amp.GradScaler(
                "cuda"
            )

        except (
            AttributeError,
            TypeError
        ):

            scaler = (
                torch.cuda.amp.GradScaler()
            )

    else:

        scaler = None


    # --------------------------------------------------------
    # CHECKPOINT PATHS
    # --------------------------------------------------------

    latest_checkpoint_path = (
        checkpoint_dir /
        "latest_checkpoint.pt"
    )


    best_psnr_path = (
        checkpoint_dir /
        "best_psnr.pt"
    )


    best_ssim_path = (
        checkpoint_dir /
        "best_ssim.pt"
    )


    best_balanced_path = (
        checkpoint_dir /
        "best_balanced.pt"
    )


    csv_path = (
        metrics_dir /
        "training_metrics.csv"
    )


    # --------------------------------------------------------
    # RESUME STATE
    # --------------------------------------------------------

    start_epoch = 1


    best_psnr = (
        -float("inf")
    )


    best_ssim = (
        -float("inf")
    )


    best_score = (
        -float("inf")
    )


    best_psnr_epoch = 0

    best_ssim_epoch = 0

    best_score_epoch = 0


    if (
        args.resume
        and
        latest_checkpoint_path.exists()
    ):

        print()
        print("=" * 72)
        print("RESUME CHECKPOINT FOUND")
        print("=" * 72)


        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location=device
        )


        checkpoint_params = (
            checkpoint.get(
                "parameters"
            )
        )


        if checkpoint_params != (
            EXPECTED_PARAMETER_COUNT
        ):

            raise RuntimeError(
                "Checkpoint parameter count "
                "does not match 7B-3."
            )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )


        if (

            scaler is not None

            and

            "scaler_state_dict"
            in checkpoint

        ):

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )


        previous_epoch = int(
            checkpoint[
                "epoch"
            ]
        )


        start_epoch = (
            previous_epoch +
            1
        )


        best_psnr = float(
            checkpoint.get(
                "best_psnr",
                -float("inf")
            )
        )


        best_ssim = float(
            checkpoint.get(
                "best_ssim",
                -float("inf")
            )
        )


        best_score = float(
            checkpoint.get(
                "best_score",
                -float("inf")
            )
        )


        best_psnr_epoch = int(
            checkpoint.get(
                "best_psnr_epoch",
                0
            )
        )


        best_ssim_epoch = int(
            checkpoint.get(
                "best_ssim_epoch",
                0
            )
        )


        best_score_epoch = int(
            checkpoint.get(
                "best_score_epoch",
                0
            )
        )


        print(
            "Previous epoch:",
            previous_epoch
        )


        print(
            "Resume epoch:",
            start_epoch
        )


        print(
            "Best PSNR:",
            best_psnr
        )


        print(
            "Best SSIM:",
            best_ssim
        )


        print(
            "Best score:",
            best_score
        )


    elif args.resume:

        print()
        print(
            "Resume requested, "
            "but no checkpoint exists."
        )

        print(
            "Starting from epoch 1."
        )


    else:

        print()
        print(
            "Starting fresh from epoch 1."
        )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if not csv_path.exists():

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(
                f
            )


            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_psnr",
                    "train_ssim",
                    "val_loss",
                    "val_psnr",
                    "val_ssim",
                    "balanced_score",
                    "learning_rate",
                    "epoch_time_seconds"
                ]
            )


    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL TRAINING START")
    print("=" * 72)


    for epoch in range(
        start_epoch,
        TARGET_EPOCHS + 1
    ):

        epoch_start = (
            time.time()
        )


        # ----------------------------------------------------
        # LR
        # ----------------------------------------------------

        current_lr = (
            get_learning_rate(
                epoch - 1
            )
        )


        for param_group in (
            optimizer.param_groups
        ):

            param_group[
                "lr"
            ] = current_lr


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            train_psnr,
            train_ssim
        ) = run_epoch(

            model,

            train_loader,

            device,

            optimizer=optimizer,

            scaler=scaler
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            val_loss,
            val_psnr,
            val_ssim
        ) = run_epoch(

            model,

            val_loader,

            device,

            optimizer=None,

            scaler=None
        )


        elapsed = (
            time.time()
            -
            epoch_start
        )


        # ----------------------------------------------------
        # BALANCED CHECKPOINT SCORE
        # ----------------------------------------------------

        balanced_score = (
            val_psnr
            +
            20.0 *
            val_ssim
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{TARGET_EPOCHS}"
        )


        print(
            f"  Train Loss : "
            f"{train_loss:.6f}"
        )


        print(
            f"  Train PSNR : "
            f"{train_psnr:.4f} dB"
        )


        print(
            f"  Train SSIM : "
            f"{train_ssim:.6f}"
        )


        print(
            f"  Val Loss   : "
            f"{val_loss:.6f}"
        )


        print(
            f"  Val PSNR   : "
            f"{val_psnr:.4f} dB"
        )


        print(
            f"  Val SSIM   : "
            f"{val_ssim:.6f}"
        )


        print(
            f"  Score      : "
            f"{balanced_score:.6f}"
        )


        print(
            f"  LR         : "
            f"{current_lr:.8e}"
        )


        print(
            f"  Time       : "
            f"{elapsed:.1f}s"
        )


        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(
                f
            )


            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_psnr,
                    train_ssim,
                    val_loss,
                    val_psnr,
                    val_ssim,
                    balanced_score,
                    current_lr,
                    elapsed
                ]
            )


        # ----------------------------------------------------
        # BEST PSNR
        # ----------------------------------------------------

        if val_psnr > best_psnr:

            best_psnr = (
                val_psnr
            )


            best_psnr_epoch = (
                epoch
            )


            save_best_checkpoint(

                best_psnr_path,

                model,

                epoch,

                val_psnr,

                val_ssim
            )


            print(
                "  NEW BEST PSNR"
            )


        # ----------------------------------------------------
        # BEST SSIM
        # ----------------------------------------------------

        if val_ssim > best_ssim:

            best_ssim = (
                val_ssim
            )


            best_ssim_epoch = (
                epoch
            )


            save_best_checkpoint(

                best_ssim_path,

                model,

                epoch,

                val_psnr,

                val_ssim
            )


            print(
                "  NEW BEST SSIM"
            )


        # ----------------------------------------------------
        # BEST BALANCED
        # ----------------------------------------------------

        if (
            balanced_score >
            best_score
        ):

            best_score = (
                balanced_score
            )


            best_score_epoch = (
                epoch
            )


            save_best_checkpoint(

                best_balanced_path,

                model,

                epoch,

                val_psnr,

                val_ssim,

                balanced_score
            )


            print(
                "  NEW BEST BALANCED"
            )


        # ----------------------------------------------------
        # LATEST CHECKPOINT
        # ----------------------------------------------------

        latest_checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "best_epoch":
                best_score_epoch,

            "best_psnr":
                best_psnr,

            "best_ssim":
                best_ssim,

            "best_score":
                best_score,

            "best_psnr_epoch":
                best_psnr_epoch,

            "best_ssim_epoch":
                best_ssim_epoch,

            "best_score_epoch":
                best_score_epoch,

            "parameters":
                EXPECTED_PARAMETER_COUNT,

            "config":
                model_config()
        }


        if scaler is not None:

            latest_checkpoint[
                "scaler_state_dict"
            ] = scaler.state_dict()


        torch.save(
            latest_checkpoint,
            latest_checkpoint_path
        )


        # ----------------------------------------------------
        # PERIODIC CHECKPOINT
        # ----------------------------------------------------

        if (
            epoch %
            CHECKPOINT_EVERY
            == 0
        ):

            periodic_path = (

                checkpoint_dir /

                f"checkpoint_epoch_"
                f"{epoch:03d}.pt"
            )


            torch.save(
                latest_checkpoint,
                periodic_path
            )


            print(
                "  Periodic checkpoint:",
                periodic_path.name
            )


    # ========================================================
    # FINAL CONFIGURATION
    # ========================================================

    final_config = {

        "project":
            "SEMICON PS01",

        "run":
            "FINAL_7B3_CONTROLLED_100EPOCH",

        "dataset_pairs":
            EXPECTED_PAIRS,

        "train_samples":
            TRAIN_COUNT,

        "validation_samples":
            VAL_COUNT,

        "seed":
            SEED,

        "architecture":
            "Verified 7B-3",

        "parameters":
            EXPECTED_PARAMETER_COUNT,

        "input_shape":
            [1, 128, 128],

        "output_shape":
            [1, 256, 256],

        "normalization":
            {
                "mean":
                    NORMALIZATION_MEAN,

                "std":
                    NORMALIZATION_STD
            },

        "training":
            {
                "epochs":
                    TARGET_EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "initial_lr":
                    INITIAL_LR,

                "minimum_lr":
                    MIN_LR,

                "warmup_epochs":
                    WARMUP_EPOCHS,

                "optimizer":
                    "AdamW",

                "weight_decay":
                    1e-6,

                "gradient_clip_norm":
                    1.0,

                "amp":
                    device.type == "cuda"
            },

        "loss":
            {
                "charbonnier":
                    CHARBONNIER_WEIGHT,

                "ssim":
                    SSIM_WEIGHT,

                "edge":
                    EDGE_WEIGHT,

                "sharpness":
                    SHARPNESS_WEIGHT
            },

        "augmentation":
            {
                "horizontal_flip":
                    True,

                "vertical_flip":
                    True,

                "rotation_0_90_180_270":
                    True,

                "brightness_change":
                    False,

                "contrast_change":
                    False,

                "gamma_correction":
                    False,

                "histogram_equalization":
                    False,

                "clahe":
                    False,

                "artificial_noise":
                    False,

                "artificial_blur":
                    False,

                "post_sharpening":
                    False
            },

        "results":
            {
                "best_psnr":
                    best_psnr,

                "best_psnr_epoch":
                    best_psnr_epoch,

                "best_ssim":
                    best_ssim,

                "best_ssim_epoch":
                    best_ssim_epoch,

                "best_balanced_score":
                    best_score,

                "best_balanced_epoch":
                    best_score_epoch
            }
    }


    with open(
        run_dir /
        "final_configuration.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_config,
            f,
            indent=2
        )


    # ========================================================
    # OPTIONAL COMPARISONS
    # ========================================================

    if args.make_comparisons:

        print()
        print("=" * 72)
        print(
            "GENERATING "
            "FINAL COMPARISON IMAGES"
        )
        print("=" * 72)


        if not best_balanced_path.exists():

            raise RuntimeError(
                "Best balanced checkpoint "
                "does not exist."
            )


        best_checkpoint = torch.load(
            best_balanced_path,
            map_location=device
        )


        model.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ]
        )


        generate_comparisons(

            model,

            device,

            noisy_dir,

            gt_dir,

            val_ids,

            comparison_dir
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "SEMICON PS01 — "
        "FINAL TRAINING COMPLETE"
    )
    print("=" * 72)


    print(
        "Best validation PSNR:",
        f"{best_psnr:.4f} dB",
        f"(epoch {best_psnr_epoch})"
    )


    print(
        "Best validation SSIM:",
        f"{best_ssim:.6f}",
        f"(epoch {best_ssim_epoch})"
    )


    print(
        "Best balanced score:",
        f"{best_score:.6f}",
        f"(epoch {best_score_epoch})"
    )


    print(
        "Model parameters:",
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )


    print()
    print(
        "Best PSNR checkpoint:"
    )


    print(
        best_psnr_path
    )


    print()
    print(
        "Best SSIM checkpoint:"
    )


    print(
        best_ssim_path
    )


    print()
    print(
        "Best balanced checkpoint:"
    )


    print(
        best_balanced_path
    )


    print()
    print(
        "Latest checkpoint:"
    )


    print(
        latest_checkpoint_path
    )


    print()
    print(
        "Training metrics:"
    )


    print(
        csv_path
    )


    print()
    print("=" * 72)


# ============================================================
# 20. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
