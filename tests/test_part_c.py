"""
Unit & Integration Tests for Part C: Forensic Audit CLI & Evidence Reporting System.
"""

import hashlib
import json
import os
import tempfile
import numpy as np
import pytest
import torch
from PIL import Image

from oxastra_assignment.forensic_audit import (
    compute_file_sha256,
    run_forensic_audit,
    main as cli_main,
)
from oxastra_assignment.model import MobileNetV2Forensic


@pytest.fixture
def sample_jpeg(tmp_path):
    """Generates a temporary valid JPEG image (128x128)."""
    img_path = str(tmp_path / "test_sample.jpg")
    arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path, quality=90)
    return img_path


@pytest.fixture
def sample_png(tmp_path):
    """Generates a temporary valid PNG image (128x128)."""
    img_path = str(tmp_path / "test_sample.png")
    arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


@pytest.fixture
def sample_grayscale(tmp_path):
    """Generates a temporary 2D grayscale JPEG image (128x128)."""
    img_path = str(tmp_path / "test_gray.jpg")
    arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    Image.fromarray(arr).save(img_path, quality=90)
    return img_path


@pytest.fixture
def dummy_weights(tmp_path):
    """Generates a temporary PyTorch 9-channel MobileNetV2 state_dict checkpoint."""
    weights_path = str(tmp_path / "test_weights.pth")
    model = MobileNetV2Forensic(num_channels=9)
    torch.save(model.state_dict(), weights_path)
    return weights_path


def test_sha256_streaming(sample_jpeg):
    """Tests streaming SHA-256 byte hashing against standard library hashlib."""
    hash_func = compute_file_sha256(sample_jpeg)
    with open(sample_jpeg, "rb") as f:
        expected = hashlib.sha256(f.read()).hexdigest()
    assert hash_func == expected
    assert len(hash_func) == 64


def test_forensic_audit_heuristic(sample_jpeg, tmp_path):
    """Tests end-to-end audit execution in Heuristic Baseline mode."""
    out_dir = str(tmp_path / "output_heuristic")
    report, fig_path, json_path = run_forensic_audit(
        input_path=sample_jpeg,
        output_dir=out_dir,
        weights_path=None,
        quality=90,
    )

    assert os.path.exists(fig_path)
    assert os.path.exists(json_path)
    assert report["file_name"] == "test_sample.jpg"
    assert report["sha256"] == compute_file_sha256(sample_jpeg)
    assert report["tamper_verdict"] in ["TAMPERED", "AUTHENTIC", "INCONCLUSIVE"]
    assert 0.0 <= report["manipulation_probability"] <= 1.0
    assert isinstance(report["compression_artifact_anomaly_score"], float)
    assert report["tool_version"] == "1.0.0"
    assert report["model_weights_sha256"] is None
    assert report["ela_quality"] == 90
    assert "BSA 2023 §63" in report["disclaimer"]

    # Verify written JSON file content matches return dict
    with open(json_path, "r", encoding="utf-8") as f:
        data_from_file = json.load(f)
    assert data_from_file["sha256"] == report["sha256"]


def test_forensic_audit_with_model_weights(sample_jpeg, dummy_weights, tmp_path):
    """Tests end-to-end audit execution with PyTorch trained weights."""
    out_dir = str(tmp_path / "output_weights")
    report, fig_path, json_path = run_forensic_audit(
        input_path=sample_jpeg,
        output_dir=out_dir,
        weights_path=dummy_weights,
        quality=90,
    )

    assert os.path.exists(fig_path)
    assert os.path.exists(json_path)
    assert report["model_weights_sha256"] == compute_file_sha256(dummy_weights)
    assert any("MobileNetV2" in note for note in report["analysis_notes"])


def test_forensic_audit_png_warning(sample_png, tmp_path):
    """Tests non-JPEG input format detection and warning logging."""
    out_dir = str(tmp_path / "output_png")
    report, _, _ = run_forensic_audit(
        input_path=sample_png,
        output_dir=out_dir,
    )

    assert any("Non-JPEG" in note for note in report["analysis_notes"])


def test_forensic_audit_grayscale(sample_grayscale, tmp_path):
    """Tests 2D grayscale image handling and automatic conversion."""
    out_dir = str(tmp_path / "output_gray")
    report, _, _ = run_forensic_audit(
        input_path=sample_grayscale,
        output_dir=out_dir,
    )

    assert any("grayscale" in note for note in report["analysis_notes"])


def test_forensic_audit_small_image_error(tmp_path):
    """Tests minimum image dimension validation (< 64x64)."""
    small_path = str(tmp_path / "small.jpg")
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(small_path)

    with pytest.raises(ValueError, match="smaller than minimum required 64x64"):
        run_forensic_audit(input_path=small_path, output_dir=str(tmp_path / "out"))


def test_forensic_audit_corrupt_file_error(tmp_path):
    """Tests error handling when encountering a corrupt or unreadable image file."""
    corrupt_path = str(tmp_path / "corrupt.jpg")
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_DATA_CORRUPT")

    with pytest.raises(ValueError, match="Failed to decode image file"):
        run_forensic_audit(input_path=corrupt_path, output_dir=str(tmp_path / "out"))


def test_forensic_audit_missing_input_error(tmp_path):
    """Tests FileNotFoundError when target image file does not exist."""
    missing_path = str(tmp_path / "non_existent.jpg")
    with pytest.raises(FileNotFoundError, match="Input image file not found"):
        run_forensic_audit(input_path=missing_path, output_dir=str(tmp_path / "out"))


def test_cli_entrypoint(sample_jpeg, tmp_path, monkeypatch):
    """Tests CLI parser and main execution flow via sys.argv."""
    out_dir = str(tmp_path / "output_cli")
    monkeypatch.setattr(
        "sys.argv",
        [
            "forensic-audit",
            "--input",
            sample_jpeg,
            "--output_dir",
            out_dir,
            "--quality",
            "85",
        ],
    )

    cli_main()

    assert os.path.exists(out_dir)
    report_file = os.path.join(out_dir, "test_sample_report.json")
    fig_file = os.path.join(out_dir, "test_sample_forensic.png")
    assert os.path.exists(report_file)
    assert os.path.exists(fig_file)
