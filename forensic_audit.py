import argparse
import datetime
import hashlib
import json
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from forensic_features import (
    compute_anomaly_score,
    extract_ela,
    extract_fft_spectrum,
    extract_noise_residual,
    locate_suspicious_region,
)
from model import NetraForensicClassifier, build_model


def compute_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_forensic_audit(
    input_path, output_dir="./reports", weights_path=None, quality=90
):
    if not os.path.exists(input_path):
        print(f"[Error] Input image file not found: {input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    file_stem = os.path.splitext(base_name)[0]

    sha256_val = compute_file_sha256(input_path)

    ext = os.path.splitext(input_path)[1].lower()
    is_jpeg = ext in [".jpg", ".jpeg"]
    if not is_jpeg:
        print(
            f"[Forensic Notice] File '{base_name}' is non-JPEG format ({ext}). ELA compression analysis will recompress into memory."
        )

    try:
        pil_img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"[Error] Failed to open image: {e}")
        sys.exit(1)

    w, h = pil_img.size
    if w < 64 or h < 64:
        print(
            f"[Error] Image dimensions ({w}x{h}) are too small for forensic signal extraction (minimum 64x64 required)."
        )
        sys.exit(1)

    img_rgb = np.array(pil_img)

    ela_scaled, ela_heatmap, ela_diff_raw = extract_ela(
        img_rgb, quality=quality
    )
    noise_residual, noise_var_map, noise_heatmap = extract_noise_residual(
        img_rgb
    )
    _, fft_vis = extract_fft_spectrum(img_rgb)

    anomaly_score = compute_anomaly_score(ela_diff_raw)

    bbox, fusion_mask = locate_suspicious_region(ela_scaled, noise_var_map)

    prob = 0.0
    verdict = "AUTHENTIC"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if weights_path and os.path.exists(weights_path):
        model = build_model(
            weights_path=weights_path, in_channels=3, device=device
        )
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = (
            transform(ela_scaled).unsqueeze(0).to(device)
        )
        prob = model.predict_probability(input_tensor)
    else:
        heuristic_score = min(
            1.0, (anomaly_score / 6.0) * 0.7 + (noise_var_map.max() * 0.3)
        )
        prob = round(float(heuristic_score), 2)

    if prob >= 0.50:
        verdict = "FLAGGED_TAMPERED"
    elif prob >= 0.35:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "AUTHENTIC"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    img_bbox = img_rgb.copy()
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(
            img_bbox, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3
        )
        cv2.putText(
            img_bbox,
            "SUSPECT REGION",
            (bx, max(20, by - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    axes[0].imshow(img_bbox)
    axes[0].set_title("Original Surveillance Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    ela_rgb_disp = cv2.cvtColor(ela_heatmap, cv2.COLOR_BGR2RGB)
    axes[1].imshow(ela_rgb_disp)
    axes[1].set_title(
        f"Error Level Analysis (Quality={quality})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")

    noise_rgb_disp = cv2.cvtColor(noise_heatmap, cv2.COLOR_BGR2RGB)
    axes[2].imshow(noise_rgb_disp)
    axes[2].set_title("Noise Residual Variance", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    verdict_color = "red" if verdict == "FLAGGED_TAMPERED" else ("orange" if verdict == "INCONCLUSIVE" else "green")
    plt.suptitle(
        f"Netra-One Forensic Audit | File: {base_name}\n"
        f"Verdict: {verdict} | Manipulation Probability: {prob:.2f} | Anomaly Score: {anomaly_score}",
        fontsize=13,
        fontweight="bold",
        color=verdict_color,
        y=1.02,
    )

    plt.tight_layout()
    fig_filename = os.path.join(output_dir, f"{file_stem}_forensic.png")
    plt.savefig(fig_filename, bbox_inches="tight")
    plt.close()

    timestamp_iso = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    evidence_report = {
        "file_name": base_name,
        "sha256": sha256_val,
        "tamper_verdict": verdict,
        "manipulation_probability": round(float(prob), 4),
        "suspected_region_bbox": bbox if bbox else None,
        "compression_artifact_anomaly_score": anomaly_score,
        "timestamp": timestamp_iso,
        "tool_version": "1.0.0 (Netra-One BSA 2023 §63 Compliant)",
        "ela_quality": quality,
        "legal_disclaimer": "Automated forensic diagnostic output generated under Bharatiya Sakshya Adhiniyam 2023 Section 63 standards. Expert forensic verification required for court submission.",
    }

    json_filename = os.path.join(output_dir, f"{file_stem}_evidence.json")
    with open(json_filename, "w") as jf:
        json.dump(evidence_report, jf, indent=2)

    print("\n" + "=" * 60)
    print(" FORENSIC EVIDENCE REPORT GENERATED")
    print("=" * 60)
    print(f" Target File        : {base_name}")
    print(f" SHA-256 Checksum   : {sha256_val}")
    print(f" Tamper Verdict     : {verdict}")
    print(f" Manipulation Prob  : {prob:.4f}")
    print(f" Suspected Bounding Box : {bbox}")
    print(f" ELA Anomaly Score  : {anomaly_score}")
    print(f" Diagnostic Image   : {fig_filename}")
    print(f" Evidence JSON      : {json_filename}")
    print("=" * 60 + "\n")

    return evidence_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Netra-One Digital Forensic Image Inspection CLI"
    )
    parser.add_argument(
        "--input", required=True, type=str, help="Path to suspect input image"
    )
    parser.add_argument(
        "--output_dir",
        default="./reports",
        type=str,
        help="Output directory for reports and figures",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Path to trained PyTorch model weights (.pth)",
    )
    parser.add_argument(
        "--quality",
        default=90,
        type=int,
        help="JPEG quality factor for ELA (default 90)",
    )

    args = parser.parse_args()

    run_forensic_audit(
        input_path=args.input,
        output_dir=args.output_dir,
        weights_path=args.weights,
        quality=args.quality,
    )
