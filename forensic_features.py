import cv2
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy.fftpack import dct


def extract_ela(image_input, quality=90, scale=15.0):
    """
    Error Level Analysis (ELA).
    Re-saves the image at a target JPEG compression quality and computes the
    absolute pixel difference between original and recompressed versions.

    Args:
        image_input: PIL Image or numpy array (RGB).
        quality: JPEG compression quality (1-100, default 90).
        scale: Multiplier for scaling ELA pixel differences.

    Returns:
        ela_img_rgb: RGB uint8 numpy array of ELA map.
        ela_heatmap: Colorized ELA visualization (BGR uint8 array for OpenCV display/saving).
        ela_diff_raw: Raw absolute difference float array (unscaled).
    """
    if isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    # Save to in-memory JPEG buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Compute absolute difference
    diff = ImageChops.difference(pil_img, recompressed)
    ela_diff_raw = np.array(diff, dtype=np.float32)

    # Scale pixel differences
    ela_scaled = ela_diff_raw * scale
    ela_scaled = np.clip(ela_scaled, 0, 255).astype(np.uint8)

    # Generate colorized heatmap (OpenCV COLORMAP_JET)
    ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
    ela_heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

    return ela_scaled, ela_heatmap, ela_diff_raw


def extract_noise_residual(image_input, ksize=5, block_size=32):
    """
    Extracts high-pass spatial noise residual (PRNU proxy) and block-wise variance map.

    Args:
        image_input: PIL Image or numpy array (RGB).
        ksize: Kernel size for median filter blur.
        block_size: Window size for local variance heatmap.

    Returns:
        noise_residual: Spatial high-pass noise residual uint8 array.
        variance_map: Block-wise noise variance heatmap (2D float array normalized 0..1).
        variance_heatmap: Colorized heatmap (BGR uint8 array).
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    else:
        img_np = image_input.copy()

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # High-pass filter: subtract median filtered image
    blurred = cv2.medianBlur(gray, ksize)
    residual = cv2.absdiff(gray, blurred)

    # Block-wise variance calculation
    h, w = gray.shape
    variance_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = residual[y : y + block_size, x : x + block_size]
            var_val = np.var(block)
            variance_map[y : y + block_size, x : x + block_size] = var_val

    # Handle right/bottom edges if image dimensions aren't exact multiples of block_size
    if h % block_size != 0:
        variance_map[h - (h % block_size) :, :] = variance_map[
            h - block_size : h, :
        ].mean()
    if w % block_size != 0:
        variance_map[:, w - (w % block_size) :] = variance_map[
            :, w - block_size : w
        ].mean()

    # Normalize variance map to 0..1
    max_v = variance_map.max()
    if max_v > 0:
        variance_map_norm = variance_map / max_v
    else:
        variance_map_norm = variance_map

    var_uint8 = (variance_map_norm * 255).astype(np.uint8)
    variance_heatmap = cv2.applyColorMap(var_uint8, cv2.COLORMAP_HOT)

    return residual, variance_map_norm, variance_heatmap


def extract_fft_spectrum(image_input):
    """
    Computes 2D Fast Fourier Transform (FFT) spectrum magnitude to identify frequency domain artifacts.

    Args:
        image_input: PIL Image or numpy array (RGB).

    Returns:
        fft_magnitude: 2D float array log-scaled spectrum magnitude.
        fft_visualization: Normalized uint8 heatmap array.
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    else:
        img_np = image_input.copy()

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2D FFT and zero-frequency shift
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Log scale magnitude spectrum
    magnitude_spectrum = np.log(1 + np.abs(fshift))

    # Normalize to 0..255 for visualization
    mag_min = magnitude_spectrum.min()
    mag_max = magnitude_spectrum.max()
    if mag_max > mag_min:
        fft_norm = (magnitude_spectrum - mag_min) / (mag_max - mag_min) * 255.0
    else:
        fft_norm = magnitude_spectrum
    fft_visualization = fft_norm.astype(np.uint8)

    return magnitude_spectrum, fft_visualization


def compute_anomaly_score(ela_diff_raw, block_size=32):
    """
    Computes compression artifact anomaly score based on block-wise ELA statistics.
    Uses robust z-score: z = (max_block_mean - median) / MAD.

    Args:
        ela_diff_raw: Unscaled float ELA difference map.
        block_size: Grid block size.

    Returns:
        anomaly_score: Float score indicating magnitude of artifact anomaly.
    """
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
    """
    Fuses ELA heatmap and noise residual map to detect and localize suspect tampered regions.

    Args:
        ela_rgb: Scaled uint8 ELA RGB array.
        noise_var_map: Normalized 2D float noise variance map (0..1).
        min_area_ratio: Minimum region area ratio to be reported as a bounding box.

    Returns:
        bbox: List [x, y, w, h] of bounding box or None if no significant region found.
        fusion_mask: Binary uint8 mask of detected region.
    """
    h, w = noise_var_map.shape[:2]
    ela_gray = cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Signal fusion: equal weighting of ELA intensity and noise variance anomaly
    fused_signal = 0.5 * ela_gray + 0.5 * noise_var_map
    fused_uint8 = (fused_signal * 255).astype(np.uint8)

    # Thresholding: 90th percentile + Otsu hybrid
    thresh_val = max(np.percentile(fused_uint8, 88), 60)
    _, binary = cv2.threshold(fused_uint8, int(thresh_val), 255, cv2.THRESH_BINARY)

    # Morphological operations: remove isolated noise points and close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find connected component contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_bbox = None
    max_area = 0
    min_area_threshold = h * w * min_area_ratio

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Skip top timestamp overlay strip (y < 40 and bh < 45)
        if y < 40 and bh < 45:
            continue
        if area > max_area and area >= min_area_threshold:
            max_area = area
            best_bbox = [int(x), int(y), int(bw), int(bh)]

    # Fallback to largest contour if no non-timestamp contour found
    if best_bbox is None and contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area >= min_area_threshold:
                max_area = area
                x, y, bw, bh = cv2.boundingRect(cnt)
                best_bbox = [int(x), int(y), int(bw), int(bh)]

    return best_bbox, binary
