"""
Forensic Feature Extraction Module for Netra-One Forensics.

Provides signal extraction methods for image forgery & tamper detection:
1. Error Level Analysis (ELA)
2. Noise Residual Analysis & Local Variance Heatmap
3. FFT Frequency Spectrum Analysis
4. Robust Z-score Anomaly Score Calculation (MAD)
5. Signal Fusion Bounding Box Localization
"""

import io
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def compute_ela(
    img_rgb: np.ndarray, quality: int = 90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Error Level Analysis (ELA) by re-compressing the image at a target JPEG quality.

    Args:
        img_rgb: Input image in RGB uint8 format (HxWx3).
        quality: JPEG compression quality (1-100). Default is 90.

    Returns:
        Tuple of:
            - ela_scaled (np.ndarray): Per-channel 0-255 scaled ELA difference (RGB uint8).
            - ela_gray (np.ndarray): Grayscale magnitude of ELA (uint8).
            - ela_vis (np.ndarray): JET colormap visualization of ELA (RGB uint8).
    """
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer).convert("RGB"))

    # Compute absolute difference
    diff = np.abs(img_rgb.astype(np.float32) - recompressed.astype(np.float32))

    # Scale per channel to 0-255
    ela_scaled = np.zeros_like(diff, dtype=np.uint8)
    for ch in range(3):
        ch_max = diff[:, :, ch].max()
        if ch_max > 0:
            ela_scaled[:, :, ch] = np.clip(
                diff[:, :, ch] * (255.0 / ch_max), 0, 255
            ).astype(np.uint8)
        else:
            ela_scaled[:, :, ch] = 0

    ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
    ela_vis_bgr = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    ela_vis = cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)

    return ela_scaled, ela_gray, ela_vis


def compute_noise_residual(
    img_rgb: np.ndarray, block_size: int = 32, ksize: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes median filter noise residual and local block variance heatmap.

    Args:
        img_rgb: Input image in RGB uint8 format.
        block_size: Size of non-overlapping blocks for variance computation.
        ksize: Kernel size for median filter (must be odd integer).

    Returns:
        Tuple of:
            - residual (np.ndarray): Noise residual image (float32).
            - var_map (np.ndarray): Raw block variance heatmap (float32, same size as img).
            - var_map_vis (np.ndarray): HOT colormap visualization of noise variance (RGB uint8).
    """
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    gray_f = gray.astype(np.float32)
    denoised = cv2.medianBlur(gray, ksize).astype(np.float32)
    residual = gray_f - denoised

    h, w = gray.shape
    var_map = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = residual[i : i + block_size, j : j + block_size]
            if block.size > 0:
                var_val = float(np.var(block))
                var_map[i : i + block_size, j : j + block_size] = var_val

    # Normalize variance map to 0-255 for visualization
    max_var = var_map.max()
    if max_var > 0:
        var_norm = np.clip((var_map / max_var) * 255.0, 0, 255).astype(np.uint8)
    else:
        var_norm = np.zeros((h, w), dtype=np.uint8)

    var_vis_bgr = cv2.applyColorMap(var_norm, cv2.COLORMAP_HOT)
    var_map_vis = cv2.cvtColor(var_vis_bgr, cv2.COLOR_BGR2RGB)

    return residual, var_map, var_map_vis


def compute_fft_spectrum(
    img_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes 2D FFT log-magnitude frequency spectrum heatmap.

    Args:
        img_rgb: Input image in RGB uint8 format.

    Returns:
        Tuple of:
            - magnitude_spectrum (np.ndarray): Log-magnitude spectrum (float32).
            - spectrum_norm (np.ndarray): Normalized spectrum (uint8 0-255).
            - spectrum_vis (np.ndarray): VIRIDIS colormap visualization (RGB uint8).
    """
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1.0 + np.abs(fshift))

    mag_max = magnitude_spectrum.max()
    mag_min = magnitude_spectrum.min()

    if mag_max > mag_min:
        spectrum_norm = (
            (magnitude_spectrum - mag_min) / (mag_max - mag_min) * 255.0
        ).astype(np.uint8)
    else:
        spectrum_norm = np.zeros_like(gray, dtype=np.uint8)

    spectrum_vis_bgr = cv2.applyColorMap(spectrum_norm, cv2.COLORMAP_VIRIDIS)
    spectrum_vis = cv2.cvtColor(spectrum_vis_bgr, cv2.COLOR_BGR2RGB)

    return magnitude_spectrum, spectrum_norm, spectrum_vis


def compute_anomaly_score(
    ela_scaled: np.ndarray, block_size: int = 32
) -> float:
    """Computes robust Z-score anomaly score based on block-wise mean ELA magnitude.

    Uses Median Absolute Deviation (MAD):
        MAD = median(|x - median(x)|)
        z_score = (x - median(x)) / (MAD + 1e-6)
        anomaly_score = max(z_scores)

    Args:
        ela_scaled: ELA scaled image array (RGB or Grayscale uint8).
        block_size: Non-overlapping block size (default: 32).

    Returns:
        anomaly_score (float): Maximum robust Z-score across blocks.
    """
    if ela_scaled.ndim == 3:
        ela_mag = ela_scaled.astype(np.float32).mean(axis=2)
    else:
        ela_mag = ela_scaled.astype(np.float32)

    h, w = ela_mag.shape
    block_means: List[float] = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = ela_mag[i : i + block_size, j : j + block_size]
            if block.size > 0:
                block_means.append(float(np.mean(block)))

    if not block_means:
        return 0.0

    means_arr = np.array(block_means, dtype=np.float32)
    med = float(np.median(means_arr))
    mad = float(np.median(np.abs(means_arr - med)))

    if mad < 1e-6:
        # Fallback to standard deviation if MAD is extremely close to 0
        std = float(np.std(means_arr))
        if std < 1e-6:
            return 0.0
        z_scores = (means_arr - med) / std
    else:
        z_scores = (means_arr - med) / (mad + 1e-6)

    anomaly_score = float(np.max(z_scores))
    return float(round(anomaly_score, 4))


def extract_localization_bbox(
    ela_gray: np.ndarray,
    noise_var_map: np.ndarray,
    min_area: int = 100,
) -> Optional[List[int]]:
    """Fuses ELA and Noise Variance signals to detect and localize tampered bounding box.

    Args:
        ela_gray: Grayscale ELA magnitude (uint8).
        noise_var_map: Raw block variance map (float32).
        min_area: Minimum bounding box area in pixels to count as valid anomaly.

    Returns:
        bbox (Optional[List[int]]): Bounding box as [x, y, w, h] or None if no region found.
    """
    h, w = ela_gray.shape

    # Normalize ELA map to [0, 1]
    ela_max = ela_gray.max()
    ela_norm = (
        ela_gray.astype(np.float32) / ela_max if ela_max > 0 else np.zeros((h, w), dtype=np.float32)
    )

    # Normalize Noise Variance map to [0, 1]
    var_max = noise_var_map.max()
    var_norm = (
        noise_var_map / var_max if var_max > 0 else np.zeros((h, w), dtype=np.float32)
    )

    # Signal Fusion (equal weights)
    fusion_map = 0.5 * ela_norm + 0.5 * var_norm
    fusion_uint8 = np.clip(fusion_map * 255.0, 0, 255).astype(np.uint8)

    # Thresholding using Otsu's method (fallback to 95th percentile if flat map)
    if fusion_uint8.std() < 1e-3:
        threshold_val = np.percentile(fusion_uint8, 95)
        _, binary = cv2.threshold(
            fusion_uint8, threshold_val, 255, cv2.THRESH_BINARY
        )
    else:
        _, binary = cv2.threshold(
            fusion_uint8, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )

    # Morphological Open then Close to reduce noise & fill contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find connected components
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Filter by area and pick largest contour
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    return [int(x), int(y), int(bw), int(bh)]


def extract_all_features(
    img_rgb: np.ndarray, quality: int = 90
) -> Dict[str, Any]:
    """Runs end-to-end Part A forensic signal extraction pipeline on an image.

    Args:
        img_rgb: Input image in RGB format (HxWx3 uint8).
        quality: ELA re-compression quality. Default: 90.

    Returns:
        Dict containing all feature maps, visual heatmaps, anomaly score, and bounding box.
    """
    ela_scaled, ela_gray, ela_vis = compute_ela(img_rgb, quality=quality)
    residual, var_map, var_map_vis = compute_noise_residual(img_rgb)
    mag_spec, spec_norm, spec_vis = compute_fft_spectrum(img_rgb)
    anomaly_score = compute_anomaly_score(ela_scaled)
    bbox = extract_localization_bbox(ela_gray, var_map)

    return {
        "ela_scaled": ela_scaled,
        "ela_gray": ela_gray,
        "ela_vis": ela_vis,
        "noise_residual": residual,
        "noise_var_map": var_map,
        "noise_var_vis": var_map_vis,
        "fft_mag_spectrum": mag_spec,
        "fft_spectrum_norm": spec_norm,
        "fft_spectrum_vis": spec_vis,
        "anomaly_score": anomaly_score,
        "suspected_region_bbox": bbox,
    }
