"""
Unit Tests for CASIA v2.0 Dataset Processing & Importer Module.
"""

import os
import numpy as np
from PIL import Image
from oxastra_assignment.prepare_casia_dataset import (
    extract_base_image_id,
    find_casia_dirs,
    prepare_casia_dataset,
)


def test_extract_base_image_id():
    """Tests base image ID extraction for strict source split."""
    fn1 = "Tp_S_NRN_S_O_sec00032_sec00031_0025.tif"
    fn2 = "Au_sec_00032.jpg"
    assert extract_base_image_id(fn1) == "sec00032"
    assert extract_base_image_id(fn2) == "sec"


def test_prepare_casia_dataset(tmp_path):
    """Tests end-to-end CASIA dataset reorganization and splitting."""
    casia_root = tmp_path / "CASIA2"
    au_dir = casia_root / "Au"
    tp_dir = casia_root / "Tp"
    au_dir.mkdir(parents=True)
    tp_dir.mkdir(parents=True)

    # Create dummy authentic and tampered images
    for i in range(5):
        img_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(str(au_dir / f"Au_sec000{i}.jpg"))
        img.save(str(tp_dir / f"Tp_S_NRN_S_O_sec000{i}_sec0000_000{i}.tif"))

    out_dataset = tmp_path / "processed_dataset"
    stats = prepare_casia_dataset(
        source_dir=str(casia_root),
        output_dir=str(out_dataset),
        val_ratio=0.4,
    )

    assert stats["train_authentic"] + stats["val_authentic"] == 5
    assert stats["train_tampered"] + stats["val_tampered"] == 5
    assert (out_dataset / "train" / "authentic").exists()
    assert (out_dataset / "val" / "tampered").exists()
