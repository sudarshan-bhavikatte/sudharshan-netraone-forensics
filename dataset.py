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
    Generates a synthetic surveillance camera frame with realistic background patterns and timestamp.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if scene_type == "street":
        # Road asphalt gradient
        for y in range(height):
            val = int(50 + 40 * (y / height) + random.randint(-5, 5))
            img[y, :] = (val, val, val)

        # Draw lane markers
        cv2.line(img, (width // 3, height), (width // 2 - 20, height // 2), (200, 200, 200), 4)
        cv2.line(img, (2 * width // 3, height), (width // 2 + 20, height // 2), (200, 200, 200), 4)

        # Draw sidewalk / wall gradient top
        cv2.rectangle(img, (0, 0), (width, height // 3), (120, 110, 100), -1)

    elif scene_type == "indoor":
        # Tiled floor pattern
        for y in range(0, height, 40):
            for x in range(0, width, 40):
                c = 180 + (x // 40 + y // 40) % 2 * 30 + random.randint(-4, 4)
                img[y:y+40, x:x+40] = (c, c - 10, c - 20)

    # Add realistic noise and CCTV camera timestamp overlay
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    timestamp = "CAM-04 | 2026-09-29 23:45:12 UTC"
    cv2.putText(img, timestamp, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def tamper_image(image_np, tamper_type="splice"):
    """
    Applies realistic forensic tampering to an image:
    - 'splice': Pastes a object/texture region from a synthetic second source with different JPEG quality.
    - 'copy_move': Copies a patch within the same frame and pastes it elsewhere with Gaussian blur.
    - 'inpaint': Blurs/erases a region to simulate object removal.
    """
    tampered = image_np.copy()
    h, w, _ = image_np.shape

    # Select random region size (15% to 30% of canvas)
    rw = random.randint(int(w * 0.15), int(w * 0.35))
    rh = random.randint(int(h * 0.15), int(h * 0.35))
    rx = random.randint(50, w - rw - 50)
    ry = random.randint(50, h - rh - 50)

    mask = np.zeros((h, w), dtype=np.uint8)

    if tamper_type == "splice":
        # Generate foreign texture (e.g. synthetic car or person object)
        foreign_img = generate_synthetic_surveillance_frame(w, h, scene_type="indoor")
        # Save foreign patch with different JPEG compression to create ELA anomaly
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.choice([50, 60, 75])]
        _, encoded = cv2.imencode('.jpg', foreign_img, encode_param)
        foreign_compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        tampered[ry:ry+rh, rx:rx+rw] = foreign_compressed[ry:ry+rh, rx:rx+rw]
        mask[ry:ry+rh, rx:rx+rw] = 255

    elif tamper_type == "copy_move":
        # Copy from source patch
        src_x = (rx + rw + 50) % (w - rw - 10)
        src_y = ry
        patch = tampered[src_y:src_y+rh, src_x:src_x+rw].copy()
        patch = cv2.GaussianBlur(patch, (5, 5), 0)

        tampered[ry:ry+rh, rx:rx+rw] = patch
        mask[ry:ry+rh, rx:rx+rw] = 255

    elif tamper_type == "inpaint":
        # Simulate object removal via heavy spatial smoothing / inpainting
        roi = tampered[ry:ry+rh, rx:rx+rw]
        inpainted_roi = cv2.GaussianBlur(roi, (21, 21), 0)
        tampered[ry:ry+rh, rx:rx+rw] = inpainted_roi
        mask[ry:ry+rh, rx:rx+rw] = 255

    # Post-process resave at secondary compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    _, final_encoded = cv2.imencode('.jpg', tampered, encode_param)
    final_tampered = cv2.imdecode(final_encoded, cv2.IMREAD_COLOR)

    return final_tampered, mask, [rx, ry, rw, rh]


def create_mini_dataset(output_dir, num_authentic=50, num_tampered=50, seed=42):
    """
    Creates a reproducible mini benchmark dataset of surveillance frames.
    """
    random.seed(seed)
    np.random.seed(seed)

    auth_dir = os.path.join(output_dir, "authentic")
    tamp_dir = os.path.join(output_dir, "tampered")
    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(tamp_dir, exist_ok=True)

    print(f"[Dataset] Generating {num_authentic} authentic and {num_tampered} tampered surveillance samples in '{output_dir}'...")

    for i in range(num_authentic):
        frame = generate_synthetic_surveillance_frame(640, 480, scene_type=random.choice(["street", "indoor"]))
        # Save clean JPEG
        quality = random.choice([88, 90, 92, 95])
        filepath = os.path.join(auth_dir, f"authentic_{i+1:03d}.jpg")
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    tamper_types = ["splice", "copy_move", "inpaint"]
    for i in range(num_tampered):
        frame = generate_synthetic_surveillance_frame(640, 480, scene_type=random.choice(["street", "indoor"]))
        ttype = tamper_types[i % len(tamper_types)]
        tamp_frame, mask, bbox = tamper_image(frame, tamper_type=ttype)
        filepath = os.path.join(tamp_dir, f"tampered_{i+1:03d}_{ttype}.jpg")
        cv2.imwrite(filepath, tamp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print("[Dataset] Dataset generation complete.")


class ForensicDataset(Dataset):
    """
    PyTorch Dataset for forensic image classification.
    Extracts RGB, ELA, and Noise Residual on the fly or pre-calculated.
    """

    def __init__(self, samples_list, in_channels=3, quality=90, transform=None):
        """
        samples_list: list of tuples (img_path, label: 0 or 1)
        """
        self.samples = samples_list
        self.in_channels = in_channels
        self.quality = quality
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
