"""
Synthetic Data Generation & Forensic Dataset Loading Module.

Provides:
1. Synthetic tamper generator (Splice, Copy-Move, Inpaint).
2. Strict source-image dataset splitter (prevents data leakage).
3. PyTorch ForensicDataset supporting 3-channel (ELA) and 9-channel (RGB + ELA + Noise) multi-input tensors.
"""

import io
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .forensic_features import compute_ela, compute_noise_residual


def generate_synthetic_tamper(
    src_img: np.ndarray,
    secondary_img: Optional[np.ndarray] = None,
    tamper_type: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates a tampered image and binary ground-truth mask from a clean source image.

    Tamper Types:
        - 'splice': Paste region from secondary_img onto src_img.
        - 'copy_move': Copy region from src_img, apply mild blur, paste elsewhere on src_img.
        - 'inpaint': Smear/Gaussian blur rectangular region on src_img.

    Args:
        src_img: Source RGB image uint8 (HxWx3).
        secondary_img: Optional secondary RGB image for splicing.
        tamper_type: Explicit tamper type ('splice', 'copy_move', 'inpaint') or None for random.

    Returns:
        Tuple of:
            - tampered_img (np.ndarray): Tampered RGB image uint8.
            - mask (np.ndarray): Ground truth binary mask (HxW uint8, 255 for tampered pixels).
            - chosen_type (str): Type of tampering applied.
    """
    h, w, c = src_img.shape
    tampered_img = src_img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if tamper_type is None:
        tamper_type = random.choice(["splice", "copy_move", "inpaint"])

    # Patch size: 10% to 35% of canvas dimensions
    pw = random.randint(int(w * 0.1), int(w * 0.35))
    ph = random.randint(int(h * 0.1), int(h * 0.35))
    pw = max(16, min(pw, w - 10))
    ph = max(16, min(ph, h - 10))

    dest_x = random.randint(0, w - pw)
    dest_y = random.randint(0, h - ph)

    if tamper_type == "splice":
        if secondary_img is not None and secondary_img.shape[0] >= ph and secondary_img.shape[1] >= pw:
            src_x = random.randint(0, secondary_img.shape[1] - pw)
            src_y = random.randint(0, secondary_img.shape[0] - ph)
            patch = secondary_img[src_y : src_y + ph, src_x : src_x + pw]
        else:
            # Generate artificial high-contrast noise texture patch if no secondary image available
            patch = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)

        tampered_img[dest_y : dest_y + ph, dest_x : dest_x + pw] = patch
        mask[dest_y : dest_y + ph, dest_x : dest_x + pw] = 255

    elif tamper_type == "copy_move":
        # Ensure source patch does not completely overlap destination
        src_x = random.randint(0, w - pw)
        src_y = random.randint(0, h - ph)
        patch = src_img[src_y : src_y + ph, src_x : src_x + pw].copy()

        # Apply subtle Gaussian blur to simulate smoothing after copy-move
        patch = cv2.GaussianBlur(patch, (5, 5), 1.0)
        tampered_img[dest_y : dest_y + ph, dest_x : dest_x + pw] = patch
        mask[dest_y : dest_y + ph, dest_x : dest_x + pw] = 255

    elif tamper_type == "inpaint":
        # Inpaint-like: heavy blur/median filter in rectangular target region
        patch = tampered_img[dest_y : dest_y + ph, dest_x : dest_x + pw]
        inpainted_patch = cv2.GaussianBlur(patch, (21, 21), 5.0)
        tampered_img[dest_y : dest_y + ph, dest_x : dest_x + pw] = inpainted_patch
        mask[dest_y : dest_y + ph, dest_x : dest_x + pw] = 255

    # Random post-processing re-save JPEG quality (70 to 95)
    quality = random.randint(70, 95)
    pil_img = Image.fromarray(tampered_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    tampered_img = np.array(Image.open(buf).convert("RGB"))

    return tampered_img, mask, tamper_type


def create_synthetic_dataset(
    output_dir: str,
    num_samples: int = 40,
    val_ratio: float = 0.25,
    img_size: Tuple[int, int] = (256, 256),
    seed: int = 42,
) -> Dict[str, int]:
    """Generates synthetic training and validation dataset with strict source-image splitting.

    Args:
        output_dir: Target root directory for dataset storage.
        num_samples: Total number of samples to generate.
        val_ratio: Fraction of dataset allocated for validation.
        img_size: Canvas size (height, width).
        seed: Random seed for reproducibility.

    Returns:
        Dict summarizing generated sample counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val

    splits = {
        "train": num_train,
        "val": num_val,
    }

    counts = {"train_authentic": 0, "train_tampered": 0, "val_authentic": 0, "val_tampered": 0}

    for split, count in splits.items():
        split_dir = os.path.join(output_dir, split)
        auth_dir = os.path.join(split_dir, "authentic")
        tamp_dir = os.path.join(split_dir, "tampered")
        mask_dir = os.path.join(split_dir, "masks")

        os.makedirs(auth_dir, exist_ok=True)
        os.makedirs(tamp_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(count):
            # Generate synthetic base image
            base_color = np.random.randint(50, 200, (3,), dtype=np.uint8)
            noise = np.random.randint(-20, 20, (img_size[0], img_size[1], 3))
            base_img = np.clip(base_color + noise, 0, 255).astype(np.uint8)

            # 50% Authentic, 50% Tampered
            is_tampered = (i % 2 == 1)

            if is_tampered:
                sec_color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                sec_noise = np.random.randint(-30, 30, (img_size[0], img_size[1], 3))
                sec_img = np.clip(sec_color + sec_noise, 0, 255).astype(np.uint8)

                tamp_img, mask, _ = generate_synthetic_tamper(base_img, sec_img)

                img_path = os.path.join(tamp_dir, f"sample_{i:04d}.jpg")
                mask_path = os.path.join(mask_dir, f"sample_{i:04d}.png")

                cv2.imwrite(img_path, cv2.cvtColor(tamp_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_path, mask)
                counts[f"{split}_tampered"] += 1
            else:
                img_path = os.path.join(auth_dir, f"sample_{i:04d}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR))
                counts[f"{split}_authentic"] += 1

    return counts


class ForensicDataset(Dataset):
    """PyTorch Dataset for Forensic Image Tamper Detection.

    Supports:
        - 3-channel input: ELA map only or RGB only.
        - 9-channel input: Concat([RGB (3 ch), ELA (3 ch), Noise Residual (3 ch)]).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_channels: int = 9,
        target_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
    ):
        """
        Args:
            root_dir: Path to dataset root directory.
            split: Split directory name ('train' or 'val').
            num_channels: Input channels (3 for ELA-only, 9 for Multi-channel).
            target_size: Resize dimensions for MobileNetV2 input (224, 224).
            is_train: Whether dataset is used for training (enables horizontal flip).
        """
        self.split_dir = os.path.join(root_dir, split)
        self.num_channels = num_channels
        self.target_size = target_size
        self.is_train = is_train

        self.samples: List[Tuple[str, int]] = []

        auth_dir = os.path.join(self.split_dir, "authentic")
        tamp_dir = os.path.join(self.split_dir, "tampered")

        if os.path.exists(auth_dir):
            for fname in os.listdir(auth_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(auth_dir, fname), 0))

        if os.path.exists(tamp_dir):
            for fname in os.listdir(tamp_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(tamp_dir, fname), 1))

        # ImageNet normalization stats
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))

        # Data augmentation: horizontal flip only (no resize/recompression)
        if self.is_train and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)

        ela_scaled, _, _ = compute_ela(rgb, quality=90)
        residual, _, _ = compute_noise_residual(rgb)

        # Convert residual (1 channel float32) to 3-channel normalized uint8
        res_abs = np.abs(residual)
        res_max = res_abs.max()
        if res_max > 0:
            res_uint8 = np.clip((res_abs / res_max) * 255.0, 0, 255).astype(np.uint8)
        else:
            res_uint8 = np.zeros_like(res_abs, dtype=np.uint8)

        res_3ch = cv2.cvtColor(res_uint8, cv2.COLOR_GRAY2RGB)

        if self.num_channels == 3:
            # ELA input only
            t_ela = torch.from_numpy(ela_scaled.transpose(2, 0, 1)).float() / 255.0
            t_norm = (t_ela - self.norm_mean) / self.norm_std
            return t_norm, label

        elif self.num_channels == 9:
            # Multi-channel: Concat([RGB, ELA, Noise])
            t_rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            t_ela = torch.from_numpy(ela_scaled.transpose(2, 0, 1)).float() / 255.0
            t_res = torch.from_numpy(res_3ch.transpose(2, 0, 1)).float() / 255.0

            t_rgb_norm = (t_rgb - self.norm_mean) / self.norm_std
            t_ela_norm = (t_ela - self.norm_mean) / self.norm_std
            t_res_norm = (t_res - self.norm_mean) / self.norm_std

            tensor_9ch = torch.cat([t_rgb_norm, t_ela_norm, t_res_norm], dim=0)
            return tensor_9ch, label

        else:
            raise ValueError(f"Unsupported num_channels={self.num_channels}. Must be 3 or 9.")
