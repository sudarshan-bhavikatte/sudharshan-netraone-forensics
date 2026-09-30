import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from forensic_features import extract_ela, extract_noise_residual


def generate_synthetic_surveillance_frame(width=640, height=480, scene_type="street"):
    """
    Generates a realistic synthetic surveillance camera frame with rich texture,
    geometric structures, sensor noise, and CCTV camera timestamp overlays.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if scene_type == "street":
        # Asphalt road surface gradient with texture
        for y in range(height // 3, height):
            val = int(40 + 50 * (y / height) + random.randint(-8, 8))
            img[y, :] = (val, val, val)

        # Sidewalk & background building wall top
        cv2.rectangle(img, (0, 0), (width, height // 3), (110, 100, 95), -1)
        # Brick texture lines on building
        for k in range(10, height // 3, 20):
            cv2.line(img, (0, k), (width, k), (80, 70, 65), 1)

        # Perspective lane markers
        cv2.line(img, (width // 4, height), (width // 2 - 30, height // 3), (210, 210, 210), 5)
        cv2.line(img, (3 * width // 4, height), (width // 2 + 30, height // 3), (210, 210, 210), 5)

    elif scene_type == "indoor":
        # Tiled floor pattern
        tile_size = 40
        for y in range(height // 4, height, tile_size):
            for x in range(0, width, tile_size):
                c = 170 + ((x // tile_size + y // tile_size) % 2) * 35 + random.randint(-5, 5)
                img[y:y+tile_size, x:x+tile_size] = (c, c - 12, c - 22)
                cv2.rectangle(img, (x, y), (x + tile_size, y + tile_size), (100, 90, 80), 1)

        # Office wall top
        cv2.rectangle(img, (0, 0), (width, height // 4), (190, 185, 175), -1)
        cv2.line(img, (0, height // 4), (width, height // 4), (70, 60, 50), 3)

    elif scene_type == "corridor":
        # Perspective corridor with side doors & hallway light
        img[:, :] = (140, 140, 140)
        # Ceiling & floor perspective lines
        pts_ceiling = np.array([[0, 0], [width, 0], [width // 2 + 50, height // 3], [width // 2 - 50, height // 3]], np.int32)
        cv2.fillPoly(img, [pts_ceiling], (180, 180, 185))
        pts_floor = np.array([[0, height], [width, height], [width // 2 + 50, height // 3], [width // 2 - 50, height // 3]], np.int32)
        cv2.fillPoly(img, [pts_floor], (90, 85, 80))

    # Add realistic sensor noise residual (simulating PRNU / ISO noise)
    sensor_noise = np.random.normal(0, random.uniform(4.0, 9.0), (height, width, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)

    # CCTV timestamp & camera ID overlay
    cam_id = f"CAM-0{random.randint(1, 8)}"
    sec = random.randint(10, 59)
    timestamp = f"{cam_id} | 2026-09-30 11:24:{sec:02d} UTC"
    cv2.putText(img, timestamp, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def tamper_image(image_np, tamper_type="splice"):
    """
    Applies realistic forensic tampering to an image:
    - 'splice': Pastes an object region from a different source compressed at distinct JPEG quality levels.
    - 'copy_move': Copies a patch within frame, alters contrast/blur, and pastes it with ELA mismatch.
    - 'inpaint': Spatial smoothing / object removal with local variance discontinuity.
    - 'deepfake': Injects high-frequency grid artifacts simulating synthetic AI generation.
    """
    tampered = image_np.copy()
    h, w, _ = image_np.shape

    # Select target ROI (15% to 35% of canvas size)
    rw = random.randint(int(w * 0.18), int(w * 0.35))
    rh = random.randint(int(h * 0.18), int(h * 0.35))
    rx = random.randint(40, w - rw - 40)
    ry = random.randint(50, h - rh - 50)

    mask = np.zeros((h, w), dtype=np.uint8)

    if tamper_type == "splice":
        # Generate foreign texture source
        foreign_img = generate_synthetic_surveillance_frame(w, h, scene_type="indoor")
        # Save foreign patch with significantly lower JPEG quality (50-70) to create clear ELA residual mismatch
        patch = foreign_img[ry:ry+rh, rx:rx+rw]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.choice([50, 60, 70])]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        foreign_patch_compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        tampered[ry:ry+rh, rx:rx+rw] = foreign_patch_compressed
        mask[ry:ry+rh, rx:rx+rw] = 255

    elif tamper_type == "copy_move":
        # Source region shifted
        src_x = (rx + rw + 40) % (w - rw - 20)
        src_y = (ry + 20) % (h - rh - 20)
        patch = tampered[src_y:src_y+rh, src_x:src_x+rw].copy()

        # Pre-compress source patch at different quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        patch = cv2.GaussianBlur(patch, (5, 5), 0)

        tampered[ry:ry+rh, rx:rx+rw] = patch
        mask[ry:ry+rh, rx:rx+rw] = 255

    elif tamper_type == "inpaint":
        # Heavy localized spatial smoothing to simulate object removal
        roi = tampered[ry:ry+rh, rx:rx+rw]
        inpainted_roi = cv2.GaussianBlur(roi, (25, 25), 0)
        # Add slight artificial brightness shift
        inpainted_roi = np.clip(inpainted_roi.astype(np.int16) + 15, 0, 255).astype(np.uint8)
        tampered[ry:ry+rh, rx:rx+rw] = inpainted_roi
        mask[ry:ry+rh, rx:rx+rw] = 255

    elif tamper_type == "deepfake":
        # Synthetic grid pattern insertion (FFT / frequency artifact simulation)
        roi = tampered[ry:ry+rh, rx:rx+rw].astype(np.float32)
        grid_y, grid_x = np.meshgrid(np.arange(rh), np.arange(rw), indexing='ij')
        pattern = 20.0 * np.sin(2 * np.pi * grid_x / 8.0) * np.sin(2 * np.pi * grid_y / 8.0)
        pattern = np.stack([pattern]*3, axis=-1)
        synthetic_roi = np.clip(roi + pattern, 0, 255).astype(np.uint8)
        tampered[ry:ry+rh, rx:rx+rw] = synthetic_roi
        mask[ry:ry+rh, rx:rx+rw] = 255

    # Post-process resave tampered frame at standard secondary JPEG quality (92)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    _, final_encoded = cv2.imencode('.jpg', tampered, encode_param)
    final_tampered = cv2.imdecode(final_encoded, cv2.IMREAD_COLOR)

    return final_tampered, mask, [rx, ry, rw, rh]


def create_mini_dataset(output_dir, num_authentic=100, num_tampered=100, seed=42):
    """
    Creates a reproducible benchmark dataset of surveillance frames.
    Default: 100 authentic and 100 tampered (200 total samples).
    """
    random.seed(seed)
    np.random.seed(seed)

    auth_dir = os.path.join(output_dir, "authentic")
    tamp_dir = os.path.join(output_dir, "tampered")
    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(tamp_dir, exist_ok=True)

    print(f"[Dataset] Generating {num_authentic} authentic and {num_tampered} tampered surveillance samples in '{output_dir}'...")

    scenes = ["street", "indoor", "corridor"]

    # Generate Authentic frames with uniform natural JPEG quality distribution
    for i in range(num_authentic):
        scene = scenes[i % len(scenes)]
        frame = generate_synthetic_surveillance_frame(640, 480, scene_type=scene)
        quality = random.choice([88, 90, 92, 95])
        filepath = os.path.join(auth_dir, f"authentic_{i+1:03d}.jpg")
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Generate Tampered frames with forensic manipulation types
    tamper_types = ["splice", "copy_move", "inpaint", "deepfake"]
    for i in range(num_tampered):
        scene = scenes[i % len(scenes)]
        frame = generate_synthetic_surveillance_frame(640, 480, scene_type=scene)
        ttype = tamper_types[i % len(tamper_types)]
        tamp_frame, mask, bbox = tamper_image(frame, tamper_type=ttype)
        filepath = os.path.join(tamp_dir, f"tampered_{i+1:03d}_{ttype}.jpg")
        cv2.imwrite(filepath, tamp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print("[Dataset] Dataset generation complete.")


def get_transforms(is_train=True):
    """Returns PyTorch vision transformations with optional data augmentation for training."""
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class ForensicDataset(Dataset):
    """
    PyTorch Dataset for forensic image classification.
    Extracts RGB, ELA, and Noise Residual features.
    """

    def __init__(self, samples_list, in_channels=3, quality=90, is_train=False, transform=None):
        """
        samples_list: list of tuples (img_path, label: 0 or 1)
        """
        self.samples = samples_list
        self.in_channels = in_channels
        self.quality = quality
        self.transform = transform or get_transforms(is_train=is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.in_channels == 3:
            # ELA 3-channel input representation
            ela_scaled, _, _ = extract_ela(img_rgb, quality=self.quality)
            tensor_input = self.transform(ela_scaled)
        elif self.in_channels == 9:
            # Multi-modal input: RGB (3) + ELA (3) + Noise Residual Heatmap (3)
            ela_scaled, _, _ = extract_ela(img_rgb, quality=self.quality)
            _, _, noise_heatmap = extract_noise_residual(img_rgb)
            noise_rgb = cv2.cvtColor(noise_heatmap, cv2.COLOR_BGR2RGB)

            t_rgb = self.transform(img_rgb)
            t_ela = self.transform(ela_scaled)
            t_noise = self.transform(noise_rgb)

            tensor_input = torch.cat([t_rgb, t_ela, t_noise], dim=0)  # 9 x 224 x 224
        else:
            # Fallback standard RGB
            tensor_input = self.transform(img_rgb)

        return tensor_input, torch.tensor(label, dtype=torch.long)

