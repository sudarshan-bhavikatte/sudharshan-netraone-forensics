"""
Forensic Audit CLI & Evidence Report Generator for Netra-One Forensics.

Provides CLI tool to analyze images for digital forgery/manipulation:
1. Streaming SHA-256 evidence integrity hashing (before image decoding).
2. Forensic signal extraction (ELA, Noise Residual, FFT, MAD Z-score Anomaly).
3. PyTorch MobileNetV2 (9-ch/3-ch) or Heuristic Baseline inference.
4. BSA 2023 §63 compliant JSON evidence report export.
5. High-DPI (150+ DPI) 3-panel diagnostic visualization figure.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image, ImageOps

from .forensic_features import extract_all_features
from .model import HeuristicForensicModel, MobileNetV2Forensic, build_model


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash by streaming raw file bytes before image parsing."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def preprocess_image_for_model(
    img_rgb: np.ndarray,
    quality: int = 90,
    num_channels: int = 9,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Preprocesses input RGB image into model input tensor.

    Args:
        img_rgb: Input image in RGB uint8 format (HxWx3).
        quality: ELA re-compression quality.
        num_channels: Model input channel count (3 or 9).
        target_size: Resize dimensions (224, 224).

    Returns:
        torch.Tensor of shape (1, num_channels, 224, 224).
    """
    from .forensic_features import compute_ela, compute_noise_residual

    rgb_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))

    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    ela_scaled, _, _ = compute_ela(rgb_resized, quality=quality)
    residual, _, _ = compute_noise_residual(rgb_resized)

    res_abs = np.abs(residual)
    res_max = res_abs.max()
    if res_max > 0:
        res_uint8 = np.clip((res_abs / res_max) * 255.0, 0, 255).astype(np.uint8)
    else:
        res_uint8 = np.zeros_like(res_abs, dtype=np.uint8)

    res_3ch = cv2.cvtColor(res_uint8, cv2.COLOR_GRAY2RGB)

    if num_channels == 3:
        t_ela = torch.from_numpy(ela_scaled.transpose(2, 0, 1)).float() / 255.0
        t_norm = (t_ela - norm_mean) / norm_std
        return t_norm.unsqueeze(0)

    elif num_channels == 9:
        t_rgb = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float() / 255.0
        t_ela = torch.from_numpy(ela_scaled.transpose(2, 0, 1)).float() / 255.0
        t_res = torch.from_numpy(res_3ch.transpose(2, 0, 1)).float() / 255.0

        t_rgb_norm = (t_rgb - norm_mean) / norm_std
        t_ela_norm = (t_ela - norm_mean) / norm_std
        t_res_norm = (t_res - norm_mean) / norm_std

        tensor_9ch = torch.cat([t_rgb_norm, t_ela_norm, t_res_norm], dim=0)
        return tensor_9ch.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported num_channels={num_channels}")


def generate_diagnostic_figure(
    img_rgb: np.ndarray,
    ela_vis: np.ndarray,
    noise_var_vis: np.ndarray,
    bbox: Optional[List[int]],
    tamper_verdict: str,
    probability: float,
    anomaly_score: float,
    save_path: str,
) -> None:
    """Generates and saves a 3-panel high-DPI (150+ DPI) forensic diagnostic plot.

    Panels:
        1. Original RGB image with suspected bounding box (if detected).
        2. ELA Heatmap (JET colormap).
        3. Noise Residual Variance Heatmap (HOT colormap).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Panel 1: Original Image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    if bbox is not None and len(bbox) == 4:
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="-",
            label="Suspected Patch",
        )
        axes[0].add_patch(rect)
        axes[0].legend(loc="upper right", fontsize=9, framealpha=0.8)

    # Panel 2: ELA Heatmap
    axes[1].imshow(ela_vis)
    axes[1].set_title("Error Level Analysis (ELA)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Noise Residual Variance
    axes[2].imshow(noise_var_vis)
    axes[2].set_title("Noise Residual Variance", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Overall Header & Verdict Styling
    verdict_color = (
        "#d62728"
        if tamper_verdict == "TAMPERED"
        else "#ff7f0e"
        if tamper_verdict == "INCONCLUSIVE"
        else "#2ca02c"
    )

    fig.suptitle(
        f"Verdict: {tamper_verdict}   |   P(tamper) = {probability:.4f}   |   Anomaly Score = {anomaly_score:.2f}",
        fontsize=14,
        fontweight="bold",
        color=verdict_color,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_forensic_audit(
    input_path: str,
    output_dir: str,
    weights_path: Optional[str] = None,
    quality: int = 90,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], str, str]:
    """Executes full Part C Forensic Evidence Audit workflow on an image.

    Args:
        input_path: Path to target image file.
        output_dir: Output directory path for evidence report & diagnostic figure.
        weights_path: Optional path to PyTorch model checkpoint (.pth).
        quality: JPEG re-save quality for ELA signal extraction (1-100).
        threshold: Classification decision threshold (0.0-1.0).

    Returns:
        Dict containing full evidence report data structure.
    """
    analysis_notes: List[str] = []

    # 1. Check Input File Existence
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image file not found: {input_path}")

    # 2. SHA-256 Hash Computation (Stream bytes BEFORE image decoding)
    image_sha256 = compute_file_sha256(input_path)

    # 3. Format & Integrity Edge Case Handling
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in [".jpg", ".jpeg"]:
        warning_msg = (
            f"Non-JPEG input format detected ({ext}). ELA signal relies on JPEG lossy "
            f"quantization artifacts and may be less reliable on uncompressed/lossless formats."
        )
        print(f"WARNING: {warning_msg}", file=sys.stderr)
        analysis_notes.append(warning_msg)

    # Decode image using PIL & OpenCV safely
    try:
        pil_img = Image.open(input_path)
        pil_img = ImageOps.exif_transpose(pil_img)  # Handle EXIF orientation
        img_np = np.array(pil_img)
    except Exception as err:
        raise ValueError(f"Failed to decode image file '{input_path}': {err}")

    if img_np is None or img_np.size == 0:
        raise ValueError(f"Image array loaded from '{input_path}' is empty or corrupt.")

    # Convert grayscale / RGBA to 3-channel RGB
    if img_np.ndim == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        analysis_notes.append("Converted single-channel grayscale input to 3-channel RGB.")
    elif img_np.ndim == 3 and img_np.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        analysis_notes.append("Converted 4-channel RGBA input to 3-channel RGB.")
    elif img_np.ndim == 3 and img_np.shape[2] == 3:
        img_rgb = img_np
    else:
        raise ValueError(f"Unsupported image shape: {img_np.shape}")

    # Check minimum dimensions (64x64)
    h, w = img_rgb.shape[:2]
    if h < 64 or w < 64:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are smaller than minimum required 64x64."
        )

    # 4. Extract Forensic Feature Maps & Signals
    features = extract_all_features(img_rgb, quality=quality)
    anomaly_score = features["anomaly_score"]
    bbox = features["suspected_region_bbox"]

    # 5. Model Inference / Heuristic Baseline Selection
    weights_sha256: Optional[str] = None
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")

        weights_sha256 = compute_file_sha256(weights_path)
        print(f"Loading trained forensic model from '{weights_path}'...")
        
        # Build multi-channel MobileNetV2 model
        try:
            model = build_model("multichannel", weights_path=weights_path)
            num_channels = 9
        except Exception as e:
            # Fallback to 3-channel model if weights match 3-ch
            print(f"Loading as 3-channel model due to: {e}")
            model = build_model("ela_only", weights_path=weights_path)
            num_channels = 3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        input_tensor = preprocess_image_for_model(
            img_rgb, quality=quality, num_channels=num_channels
        ).to(device)

        with torch.no_grad():
            if isinstance(model, MobileNetV2Forensic):
                prob = float(model.predict_probability(input_tensor).cpu().item())
            else:
                prob = 0.5
        
        analysis_notes.append(
            f"Evaluated with PyTorch MobileNetV2 ({num_channels}-channel) model."
        )
    else:
        # Heuristic Baseline Mode
        heuristic_model = HeuristicForensicModel(threshold=3.0)
        pred, score = heuristic_model.predict(features["ela_scaled"])
        # Map MAD z-score to probability scale roughly
        prob = float(np.clip(score / 6.0, 0.0, 1.0))
        analysis_notes.append(
            "No model weights provided. Operating in Heuristic Baseline mode using MAD Z-score thresholding."
        )

    # 6. Determine Tamper Verdict
    # Range [threshold-0.15, threshold+0.15] (e.g., 0.35 to 0.65 when threshold=0.5) is INCONCLUSIVE band
    lower_inconclusive = max(0.1, threshold - 0.15)
    upper_inconclusive = min(0.9, threshold + 0.15)

    if prob >= upper_inconclusive:
        tamper_verdict = "TAMPERED"
    elif prob <= lower_inconclusive:
        tamper_verdict = "AUTHENTIC"
    else:
        tamper_verdict = "INCONCLUSIVE"

    # 7. Generate Output Paths & Files
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    figure_path = os.path.join(output_dir, f"{base_name}_forensic.png")
    json_path = os.path.join(output_dir, f"{base_name}_report.json")

    # Generate 3-panel figure
    generate_diagnostic_figure(
        img_rgb=img_rgb,
        ela_vis=features["ela_vis"],
        noise_var_vis=features["noise_var_vis"],
        bbox=bbox,
        tamper_verdict=tamper_verdict,
        probability=prob,
        anomaly_score=anomaly_score,
        save_path=figure_path,
    )

    # ISO 8601 UTC timestamp
    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # BSA 2023 §63 Compliant Report Data Structure
    report_data: Dict[str, Any] = {
        "file_name": os.path.basename(input_path),
        "sha256": image_sha256,
        "tamper_verdict": tamper_verdict,
        "manipulation_probability": float(round(prob, 4)),
        "suspected_region_bbox": bbox,
        "compression_artifact_anomaly_score": float(round(anomaly_score, 4)),
        "timestamp": timestamp_iso,
        "tool_version": "1.0.0",
        "model_weights_sha256": weights_sha256,
        "ela_quality": quality,
        "analysis_notes": analysis_notes,
        "disclaimer": (
            "This report is an automated forensic indicator, not legal proof. "
            "Human expert review is required. Pursuant to BSA 2023 §63, electronic "
            "evidence requires hash verification and chain of custody documentation."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data, figure_path, json_path


def main() -> None:
    """CLI Entry Point for Netra-One Forensic Audit."""
    parser = argparse.ArgumentParser(
        description="Netra-One Forensics — Digital Image Tamper & Deepfake Evidence Audit CLI"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input image file to audit.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        type=str,
        help="Directory path to save forensic report JSON and diagnostic figure.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to trained PyTorch model checkpoint (.pth). Runs heuristic baseline if omitted.",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=90,
        help="JPEG re-save quality for ELA signal extraction (1-100). Default: 90.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Classification decision threshold (0.0 - 1.0). Default: 0.5.",
    )

    args = parser.parse_args()

    try:
        report, fig_path, json_path = run_forensic_audit(
            input_path=args.input,
            output_dir=args.output_dir,
            weights_path=args.weights,
            quality=args.quality,
            threshold=args.threshold,
        )

        print("\n" + "=" * 64)
        print("NETRA-ONE FORENSIC EVIDENCE AUDIT REPORT")
        print("=" * 64)
        print(f"File Name        : {report['file_name']}")
        print(f"SHA-256 Hash     : {report['sha256']}")
        print(f"Verdict          : {report['tamper_verdict']}")
        print(f"Probability      : {report['manipulation_probability']:.4f}")
        print(f"Anomaly Score    : {report['compression_artifact_anomaly_score']:.4f}")
        print(f"Suspected BBox   : {report['suspected_region_bbox']}")
        print(f"Diagnostic Plot  : {fig_path}")
        print(f"JSON Evidence    : {json_path}")
        print("=" * 64 + "\n")

    except Exception as err:
        print(f"Error during forensic audit: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
