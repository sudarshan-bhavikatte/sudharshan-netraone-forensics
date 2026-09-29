"""
CASIA v2.0 Dataset Processing & Strict-Split Preparation Script.

Processes the CASIA v2.0 Image Tampering Detection Database:
1. Downloads via kagglehub (or uses a provided local path).
2. Parses Authentic ('Au') and Tampered ('Tp') image collections.
3. Groups images by base source photo to guarantee ZERO data leakage between train and val splits.
4. Converts images to standard RGB JPEGs and organizes them into:
   dataset/train/authentic, dataset/train/tampered
   dataset/val/authentic, dataset/val/tampered
"""

import argparse
import os
import random
import shutil
import sys
from typing import Dict, List, Tuple
from PIL import Image


def unpack_if_archive(source_path: str) -> str:
    """If source_path is an archive file (.zip, .rar, .7z, .tar.gz), extracts it to a temporary directory."""
    if not os.path.isfile(source_path):
        return source_path

    ext = source_path.lower()
    extract_dir = os.path.join(os.path.dirname(source_path), "extracted_casia")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting archive '{source_path}' to '{extract_dir}'...")
    if ext.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(source_path, "r") as z:
            z.extractall(extract_dir)
    elif ext.endswith(".7z"):
        import py7zr
        with py7zr.SevenZipFile(source_path, mode="r") as z:
            z.extractall(extract_dir)
    elif ext.endswith(".rar"):
        import rarfile
        with rarfile.RarFile(source_path) as r:
            r.extractall(extract_dir)
    else:
        shutil.unpack_archive(source_path, extract_dir)

    return extract_dir


def find_casia_dirs(root_path: str) -> Tuple[str, str]:
    """Finds authentic ('Au') and tampered ('Tp') image directories inside CASIA root."""
    root_path = unpack_if_archive(root_path)
    au_dir = None
    tp_dir = None

    for r, dirs, files in os.walk(root_path):
        for d in dirs:
            d_lower = d.lower()
            if d_lower in ["au", "casia2.0_au", "casia2_au"]:
                au_dir = os.path.join(r, d)
            elif d_lower in ["tp", "casia2.0_tp", "casia2_tp"]:
                tp_dir = os.path.join(r, d)

    if not au_dir or not tp_dir:
        # Check direct children if walk failed
        items = os.listdir(root_path)
        for item in items:
            full = os.path.join(root_path, item)
            if os.path.isdir(full):
                if "au" in item.lower():
                    au_dir = full
                elif "tp" in item.lower():
                    tp_dir = full

    if not au_dir or not tp_dir:
        raise FileNotFoundError(
            f"Could not locate 'Au' (Authentic) and 'Tp' (Tampered) subdirectories under '{root_path}'."
        )

    return au_dir, tp_dir


def extract_base_image_id(filename: str) -> str:
    """Extracts base source image ID from CASIA filename to enforce strict source split.

    Example CASIA tampered filename: 'Tp_S_NRN_S_O_sec00032_sec00031_0025.tif'
    Base ID extracted: 'sec00032' or prefix.
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split("_")
    if len(parts) >= 6:
        # Common CASIA 2.0 naming convention
        return parts[5]
    elif len(parts) >= 2:
        return parts[1]
    return basename


def prepare_casia_dataset(
    source_dir: str,
    output_dir: str = "dataset",
    val_ratio: float = 0.2,
    max_samples: int = 0,
    seed: int = 42,
) -> Dict[str, int]:
    """Processes CASIA v2.0 dataset into train/val split structure.

    Args:
        source_dir: Path to extracted CASIA 2.0 dataset directory.
        output_dir: Target output dataset root directory (default: 'dataset').
        val_ratio: Fraction of source images allocated to validation set (default: 0.2).
        max_samples: Optional limit on total images to copy (0 for all).
        seed: Random seed for reproducible splitting.

    Returns:
        Dict with sample count statistics per split and class.
    """
    random.seed(seed)
    au_dir, tp_dir = find_casia_dirs(source_dir)

    print(f"Located Authentic directory: {au_dir}")
    print(f"Located Tampered directory : {tp_dir}")

    # Collect valid images
    valid_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")

    au_files = [
        f for f in os.listdir(au_dir) if f.lower().endswith(valid_exts)
    ]
    tp_files = [
        f for f in os.listdir(tp_dir) if f.lower().endswith(valid_exts)
    ]

    print(f"Found {len(au_files)} authentic images and {len(tp_files)} tampered images.")

    # Group tampered images by base source ID
    groups: Dict[str, List[Tuple[str, str, int]]] = {}

    for fname in au_files:
        base_id = extract_base_image_id(fname)
        full_path = os.path.join(au_dir, fname)
        if base_id not in groups:
            groups[base_id] = []
        groups[base_id].append((full_path, fname, 0))

    for fname in tp_files:
        base_id = extract_base_image_id(fname)
        full_path = os.path.join(tp_dir, fname)
        if base_id not in groups:
            groups[base_id] = []
        groups[base_id].append((full_path, fname, 1))

    # Split groups strictly into train and val
    base_ids = list(groups.keys())
    random.shuffle(base_ids)

    num_val_groups = int(len(base_ids) * val_ratio)
    val_group_set = set(base_ids[:num_val_groups])
    train_group_set = set(base_ids[num_val_groups:])

    # Output directory paths
    target_dirs = {
        ("train", 0): os.path.join(output_dir, "train", "authentic"),
        ("train", 1): os.path.join(output_dir, "train", "tampered"),
        ("val", 0): os.path.join(output_dir, "val", "authentic"),
        ("val", 1): os.path.join(output_dir, "val", "tampered"),
    }

    for d in target_dirs.values():
        os.makedirs(d, exist_ok=True)

    stats = {
        "train_authentic": 0,
        "train_tampered": 0,
        "val_authentic": 0,
        "val_tampered": 0,
    }

    total_copied = 0

    for base_id, item_list in groups.items():
        split_name = "val" if base_id in val_group_set else "train"

        for src_path, fname, label in item_list:
            if max_samples > 0 and total_copied >= max_samples:
                break

            dest_dir = target_dirs[(split_name, label)]
            base_out = os.path.splitext(fname)[0] + ".jpg"
            dest_path = os.path.join(dest_dir, base_out)

            # Convert and save as standardized JPEG RGB
            try:
                with Image.open(src_path) as img:
                    img_rgb = img.convert("RGB")
                    img_rgb.save(dest_path, "JPEG", quality=95)

                key = f"{split_name}_{'tampered' if label == 1 else 'authentic'}"
                stats[key] += 1
                total_copied += 1
            except Exception as e:
                print(f"Skipping unreadable file '{fname}': {e}", file=sys.stderr)

        if max_samples > 0 and total_copied >= max_samples:
            break

    print("\n--- CASIA v2.0 Dataset Processing Summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v} images")
    print(f"Total processed images: {total_copied}")
    print(f"Dataset ready in '{output_dir}/'")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="CASIA v2.0 Dataset Importer & Splitter")
    parser.add_argument(
        "--source_dir",
        "-s",
        type=str,
        default=None,
        help="Path to CASIA 2.0 extracted directory. If omitted, downloads automatically via kagglehub.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="dataset",
        help="Target dataset output directory. Default: 'dataset'.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio. Default: 0.2.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit total number of processed images (0 for full dataset). Default: 0.",
    )

    args = parser.parse_args()

    source = args.source_dir
    if source is None:
        print("Downloading / locating CASIA v2.0 dataset via kagglehub...")
        import kagglehub
        handles = ["sophatvathana/casia-dataset"]
        downloaded = False
        for handle in handles:
            try:
                source = kagglehub.dataset_download(handle)
                print(f"Dataset downloaded successfully to: {source}")
                downloaded = True
                break
            except Exception as err:
                print(f"Notice: Download failed for '{handle}': {err}", file=sys.stderr)

        if not downloaded or not source:
            print("\n" + "=" * 68, file=sys.stderr)
            print("AUTOMATIC CASIA v2.0 DOWNLOAD FAILED / INTERRUPTED", file=sys.stderr)
            print("=" * 68, file=sys.stderr)
            print("Cause: Kaggle download of the 5.2 GB dataset was interrupted.", file=sys.stderr)
            print("Solution Options:\n", file=sys.stderr)
            print("1. Re-run this script to retry download:", file=sys.stderr)
            print("   uv run python -m oxastra_assignment.prepare_casia_dataset\n", file=sys.stderr)
            print("2. Or download CASIA v2.0 manually from:", file=sys.stderr)
            print("   https://www.kaggle.com/datasets/sophatvathana/casia-dataset", file=sys.stderr)
            print("   and specify your local folder path:", file=sys.stderr)
            print("   uv run python -m oxastra_assignment.prepare_casia_dataset --source_dir /path/to/CASIA2.0\n", file=sys.stderr)
            print("3. Or run with synthetic procedural data:", file=sys.stderr)
            print("   uv run python -m oxastra_assignment.train_or_eval_model --epochs 8\n", file=sys.stderr)
            print("=" * 68 + "\n", file=sys.stderr)
            sys.exit(1)

    prepare_casia_dataset(
        source_dir=source,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
