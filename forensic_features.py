import cv2
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy.fftpack import dct


def extract_ela(image_input, quality=90, scale=15.0):
    if isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(pil_img, recompressed)
    ela_diff_raw = np.array(diff, dtype=np.float32)

    ela_scaled = ela_diff_raw * scale
    ela_scaled = np.clip(ela_scaled, 0, 255).astype(np.uint8)

    ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
    ela_heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

    return ela_scaled, ela_heatmap, ela_diff_raw


def extract_noise_residual(image_input, ksize=5, block_size=32):
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    else:
        img_np = image_input.copy()

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    blurred = cv2.medianBlur(gray, ksize)
    residual = cv2.absdiff(gray, blurred)

    h, w = gray.shape
    variance_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = residual[y : y + block_size, x : x + block_size]
            var_val = np.var(block)
            variance_map[y : y + block_size, x : x + block_size] = var_val

    if h % block_size != 0:
        variance_map[h - (h % block_size) :, :] = variance_map[
            h - block_size : h, :
        ].mean()
    if w % block_size != 0:
        variance_map[:, w - (w % block_size) :] = variance_map[
            :, w - block_size : w
        ].mean()

    max_v = variance_map.max()
    if max_v > 0:
        variance_map_norm = variance_map / max_v
    else:
        variance_map_norm = variance_map

    var_uint8 = (variance_map_norm * 255).astype(np.uint8)
    variance_heatmap = cv2.applyColorMap(var_uint8, cv2.COLORMAP_HOT)

    return residual, variance_map_norm, variance_heatmap


def extract_fft_spectrum(image_input):
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    else:
        img_np = image_input.copy()

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = np.log(1 + np.abs(fshift))

    mag_min = magnitude_spectrum.min()
    mag_max = magnitude_spectrum.max()
    if mag_max > mag_min:
        fft_norm = (magnitude_spectrum - mag_min) / (mag_max - mag_min) * 255.0
    else:
        fft_norm = magnitude_spectrum
    fft_visualization = fft_norm.astype(np.uint8)

    return magnitude_spectrum, fft_visualization


def compute_anomaly_score(ela_diff_raw, block_size=32):
    gray_diff = ela_diff_raw.mean(axis=2)
    h, w = gray_diff.shape

    block_means = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray_diff[y : y + block_size, x : x + block_size]
            block_means.append(np.mean(block))

    if not block_means:
        return 0.0

    block_means = np.array(block_means)
    med = np.median(block_means)
    mad = np.median(np.abs(block_means - med))

    if mad > 1e-6:
        z_scores = (block_means - med) / (1.4826 * mad)
        anomaly_score = float(np.max(z_scores))
    else:
        anomaly_score = float(np.max(block_means) - med)

    return round(float(np.clip(anomaly_score, 0.0, 99.99)), 2)


def locate_suspicious_region(ela_rgb, noise_var_map, min_area_ratio=0.01):
    h, w = noise_var_map.shape[:2]
    ela_gray = cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    fused_signal = 0.5 * ela_gray + 0.5 * noise_var_map
    fused_uint8 = (fused_signal * 255).astype(np.uint8)

    thresh_val = max(np.percentile(fused_uint8, 88), 60)
    _, binary = cv2.threshold(fused_uint8, int(thresh_val), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_bbox = None
    max_area = 0
    min_area_threshold = h * w * min_area_ratio

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y < 40 and bh < 45:
            continue
        if area > max_area and area >= min_area_threshold:
            max_area = area
            best_bbox = [int(x), int(y), int(bw), int(bh)]

    if best_bbox is None and contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area >= min_area_threshold:
                max_area = area
                x, y, bw, bh = cv2.boundingRect(cnt)
                best_bbox = [int(x), int(y), int(bw), int(bh)]

    return best_bbox, binary
