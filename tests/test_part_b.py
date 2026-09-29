"""
Unit tests for Part B: Synthetic Data, Model Architectures, and Training Pipeline.
"""

import os
import shutil
import numpy as np
import pytest
import torch

from oxastra_assignment.synthetic_data import (
    generate_synthetic_tamper,
    create_synthetic_dataset,
    ForensicDataset,
)
from oxastra_assignment.model import (
    HeuristicForensicModel,
    MobileNetV2Forensic,
    build_model,
)
from oxastra_assignment.train_or_eval_model import (
    compute_metrics,
    compute_roc_auc,
    sweep_optimal_threshold,
    run_pipeline,
)


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Creates a temporary dataset directory for unit testing."""
    dataset_dir = str(tmp_path / "test_dataset")
    create_synthetic_dataset(output_dir=dataset_dir, num_samples=8, val_ratio=0.25, seed=42)
    yield dataset_dir
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)


def test_generate_synthetic_tamper():
    src_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    sec_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    for tamper_type in ["splice", "copy_move", "inpaint"]:
        tamp_img, mask, chosen_type = generate_synthetic_tamper(src_img, sec_img, tamper_type=tamper_type)
        assert tamp_img.shape == src_img.shape
        assert mask.shape == (128, 128)
        assert chosen_type == tamper_type
        assert mask.max() == 255


def test_forensic_dataset_3ch(temp_dataset_dir):
    ds = ForensicDataset(root_dir=temp_dataset_dir, split="train", num_channels=3, target_size=(224, 224))
    assert len(ds) > 0
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label in (0, 1)


def test_forensic_dataset_9ch(temp_dataset_dir):
    ds = ForensicDataset(root_dir=temp_dataset_dir, split="train", num_channels=9, target_size=(224, 224))
    assert len(ds) > 0
    tensor, label = ds[0]
    assert tensor.shape == (9, 224, 224)
    assert label in (0, 1)


def test_mobilenet_v2_9ch_weight_patch():
    model = MobileNetV2Forensic(num_channels=9)
    first_conv = model.model.features[0][0]
    assert first_conv.in_channels == 9

    # Verify channels 3..9 are initialized to zero
    zero_weights = first_conv.weight.data[:, 3:, :, :]
    assert torch.all(zero_weights == 0.0)

    # Test forward pass with dummy 9-channel tensor
    dummy_input = torch.randn(2, 9, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 2)

    probs = model.predict_probability(dummy_input)
    assert probs.shape == (2,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_heuristic_model():
    model = build_model("heuristic")
    assert isinstance(model, HeuristicForensicModel)
    ela_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    pred, score = model.predict(ela_img)
    assert pred in (0, 1)
    assert isinstance(score, float)


def test_compute_metrics_and_roc_auc():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

    auc = compute_roc_auc(y_true, y_prob)
    assert auc == 1.0


def test_threshold_sweep():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.85, 0.6, 0.4, 0.1])
    best_thresh, best_f1, metrics = sweep_optimal_threshold(y_true, y_prob)
    assert 0.1 <= best_thresh <= 0.9
    assert best_f1 > 0.0


def test_end_to_end_training_pipeline(tmp_path):
    dataset_dir = str(tmp_path / "ds")
    output_dir = str(tmp_path / "out")
    weights_dir = str(tmp_path / "w")

    create_synthetic_dataset(output_dir=dataset_dir, num_samples=8, val_ratio=0.25)

    metrics = run_pipeline(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        weights_dir=weights_dir,
        epochs=1,
        batch_size=4,
        variant="multichannel",
    )

    assert os.path.exists(os.path.join(weights_dir, "best_model.pth"))
    assert os.path.exists(os.path.join(output_dir, "roc_curve.png"))
    assert os.path.exists(os.path.join(output_dir, "confusion_matrix.png"))
    assert "accuracy" in metrics
