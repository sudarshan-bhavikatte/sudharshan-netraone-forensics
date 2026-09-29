"""
Unit tests for Part A: forensic_features.py
Verifies ELA, Noise Residual, FFT Spectrum, Anomaly Score, and Localization Bounding Box.
"""

import numpy as np
import pytest
import cv2

from oxastra_assignment.forensic_features import (
    compute_ela,
    compute_noise_residual,
    compute_fft_spectrum,
    compute_anomaly_score,
    extract_localization_bbox,
    extract_all_features,
)


@pytest.fixture
def sample_synthetic_image():
    """Generates a 256x256 RGB image with a spliced high-variance patch for testing."""
    np.random.seed(42)
    # Background texture
    img = np.random.randint(80, 120, (256, 256, 3), dtype=np.uint8)

    # Spliced region with high-contrast sharp edge patch at [50:110, 50:110]
    patch = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
    img[50:110, 50:110] = patch

    return img


def test_compute_ela(sample_synthetic_image):
    ela_scaled, ela_gray, ela_vis = compute_ela(sample_synthetic_image, quality=90)
    assert ela_scaled.shape == sample_synthetic_image.shape
    assert ela_gray.shape == (256, 256)
    assert ela_vis.shape == sample_synthetic_image.shape
    assert ela_scaled.dtype == np.uint8
    assert ela_vis.dtype == np.uint8


def test_compute_noise_residual(sample_synthetic_image):
    residual, var_map, var_map_vis = compute_noise_residual(sample_synthetic_image, block_size=32)
    assert residual.shape == (256, 256)
    assert var_map.shape == (256, 256)
    assert var_map_vis.shape == sample_synthetic_image.shape
    assert var_map.max() >= 0.0


def test_compute_fft_spectrum(sample_synthetic_image):
    mag_spec, spec_norm, spec_vis = compute_fft_spectrum(sample_synthetic_image)
    assert mag_spec.shape == (256, 256)
    assert spec_norm.shape == (256, 256)
    assert spec_vis.shape == sample_synthetic_image.shape
    assert spec_norm.max() <= 255


def test_compute_anomaly_score(sample_synthetic_image):
    ela_scaled, _, _ = compute_ela(sample_synthetic_image, quality=90)
    score = compute_anomaly_score(ela_scaled, block_size=32)
    assert isinstance(score, float)
    assert score >= 0.0


def test_extract_localization_bbox(sample_synthetic_image):
    ela_scaled, ela_gray, _ = compute_ela(sample_synthetic_image, quality=90)
    _, var_map, _ = compute_noise_residual(sample_synthetic_image, block_size=32)
    bbox = extract_localization_bbox(ela_gray, var_map, min_area=50)

    # Bbox should return [x, y, w, h] list of integers if detected, or None
    if bbox is not None:
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert all(isinstance(val, int) for val in bbox)
        assert w > 0 and h > 0


def test_extract_all_features(sample_synthetic_image):
    results = extract_all_features(sample_synthetic_image, quality=90)
    required_keys = [
        "ela_scaled",
        "ela_gray",
        "ela_vis",
        "noise_residual",
        "noise_var_map",
        "noise_var_vis",
        "fft_mag_spectrum",
        "fft_spectrum_norm",
        "fft_spectrum_vis",
        "anomaly_score",
        "suspected_region_bbox",
    ]
    for key in required_keys:
        assert key in results, f"Missing key {key} in extract_all_features output"
