# Netra-One Forensics — Implementation Plan

**Goal:** Image tamper and deepfake forensic tool: ELA + noise/frequency analysis, lightweight CNN classifier, CLI that outputs a visual diagnostic and a SHA-256 evidence JSON.

**Rubric weights:** Signal extraction 30 | Model quality 25 | CLI/report 25 | README/defense 20

---

## Part A: `forensic_features.py` — Signal Extraction (30 pts)

### ELA (Error Level Analysis)
- Load image as RGB.
- Re-save to an in-memory buffer at JPEG quality = 90 (configurable via `--quality`).
- Compute per-pixel absolute difference: `ela = |original - recompressed|`.
- Scale to 0–255: `ela = ela * (255 / ela.max())` — per channel.
- Apply `cv2.COLORMAP_JET` for visualization.
- **Why it works:** Authentic regions that have already been compressed converge to a stable error level; spliced regions imported from a different compression history show higher or inconsistent error levels.

### Noise Residual
- Compute: `R = I - medianBlur(I, ksize=5)`.
- Divide the residual into non-overlapping 32×32 blocks.
- Compute local variance per block → variance heatmap.
- **Why it works:** Authentic cameras have a characteristic noise texture. Spliced regions from a different source or that were blurred/inpainted have anomalously low or mismatched local variance.

### FFT Spectrum
- Convert to grayscale, compute `np.fft.fft2`, shift zero-frequency to center.
- Visualize `log(1 + |F|)` as a heatmap.
- **Why it works:** Resampling artifacts (scaling, rotation) leave periodic peaks in the frequency domain. GAN-generated images often show grid artifacts in the spectrum.

### Anomaly Score
- Compute block-wise mean ELA magnitude (same 32×32 grid).
- Robust z-score per block: `z = (x - median) / MAD`, where MAD = median absolute deviation.
- `anomaly_score = max(block z-scores)` — a single float returned in the JSON.
- Threshold ≥ 3.0 → flag as suspicious (tunable).

### Localization / Bounding Box
- Fuse signals: `fusion_map = normalize(ELA_gray) + normalize(noise_variance_map)`.
- Threshold with Otsu's method (or 95th percentile if Otsu fails on flat maps).
- Apply morphological open then close (kernel 5×5) to remove noise and fill gaps.
- Find the largest connected component.
- Return its bounding box `[x, y, w, h]` or `null` if nothing exceeds the threshold.

---

## Part B: Data & Model (25 pts)

### Synthetic Data Generator
- Take 2–3k clean JPEG images as source.
- For each image, generate one tampered version using one of:
  - **Splice:** paste a region from a different source image (random position, random size ~10–30% of canvas).
  - **Copy-move:** copy a region within the same image and paste it elsewhere with slight blur.
  - **Inpaint-like:** flood-fill or Gaussian-smear a rectangular region.
- Apply random post-processing: re-save at JPEG quality 70–95, optional mild resize.
- Save ground-truth binary masks alongside each tampered image.
- **Split strictly by source image** — the same photo must not appear in both train and test splits.

### Models

**Heuristic baseline (no training)**
- `predict = 1 if anomaly_score >= threshold else 0`
- Used for ablation comparison.

**MobileNetV2 — ELA input (3 channels)**
- Load ImageNet-pretrained MobileNetV2.
- Replace classifier head: `Dropout(0.3) → Linear(1280, 2)`.
- Input: ELA map resized to 224×224, normalized with ImageNet stats.

**MobileNetV2 — Multi-channel variant (9 channels)**
- Input: concat([RGB, ELA, noise_residual]), each 3 channels → 9 channels total.
- Patch the first conv layer: `new_conv = Conv2d(9, 32, ...)`, copy ImageNet weights for first 3 channels, initialize remaining 6 to zero.
- Same classifier head as above.
- This variant is the target submission model.

### Training Config
- Optimizer: Adam, lr = 1e-4, weight decay = 1e-5.
- Input size: 224×224. Augmentation: horizontal flip only (no resize/recompression — they destroy ELA signal).
- 8–10 epochs, early stopping on validation AUC (patience = 3).
- Save checkpoint each epoch as `weights/epoch_{n}.pth`, keep best as `weights/best_model.pth`.
- Threshold selection: sweep 0.1–0.9 on val set, pick F1-maximizing threshold. Optional `INCONCLUSIVE` band (0.35–0.65).

### Metrics
- Accuracy, Precision, Recall, F1, ROC-AUC.
- ROC curve plot saved to `outputs/roc_curve.png`.
- Confusion matrix saved to `outputs/confusion_matrix.png`.
- Ablation table: heuristic vs ELA-only model vs ELA+noise model.
- Optional: localization IoU on synthetic data (if ground-truth masks available).

---

## Part C: `forensic_audit.py` — CLI & Report (25 pts)

### Arguments
```
--input          path to image (required)
--output_dir     directory for figure + JSON (required)
--weights        path to .pth file (optional; heuristic-only if omitted)
--quality        JPEG re-save quality for ELA (default: 90)
```

### SHA-256 Hash
- Computed via **streaming `hashlib.sha256`** on the raw file bytes **before any image processing**.
- Also compute `sha256` of the weights file (if provided) and include as `model_weights_sha256`.

### JSON Evidence Report
Required keys:
```json
{
  "file_name": "sample.jpg",
  "sha256": "abc123...",
  "tamper_verdict": "TAMPERED | AUTHENTIC | INCONCLUSIVE",
  "manipulation_probability": 0.87,
  "suspected_region_bbox": [x, y, w, h],
  "compression_artifact_anomaly_score": 4.21,
  "timestamp": "2026-09-29T10:00:00Z"
}
```
Extra fields:
```json
{
  "tool_version": "1.0.0",
  "model_weights_sha256": "...",
  "ela_quality": 90,
  "analysis_notes": "...",
  "disclaimer": "This report is an automated indicator only, not legal proof. Human expert review required. BSA 2023 §63 requires a certificate and chain of custody."
}
```

### Diagnostic Figure
- Three-panel matplotlib figure at 150+ dpi.
- Panel 1: Original image with bounding box drawn (red rectangle, 2px).
- Panel 2: ELA heatmap (COLORMAP_JET).
- Panel 3: Noise variance heatmap (COLORMAP_HOT).
- Suptitle: `"Verdict: TAMPERED  |  P(tamper) = 0.87"`.
- Saved as `outputs/<basename>_forensic.png`.

### Edge Cases
| Condition | Behavior |
|---|---|
| PNG / non-JPEG input | Warn user that ELA is less reliable; continue with re-save to JPEG in memory |
| Image smaller than 64×64 | Raise error with clear message |
| Grayscale image | Convert to 3-channel RGB before processing |
| Corrupt or unreadable file | Catch PIL/cv2 exception, print clear error, exit code 1 |
| No weights provided | Run heuristic-only mode, note in `analysis_notes` |

---

## Part D: README (20 pts)

- Problem statement and ASCII/Mermaid pipeline diagram.
- Math: ELA derivation, noise residual, DCT/FFT rationale.
- Dataset construction, strict split policy, CASIA v2 bias caveat (compression uniformity).
- Results table, ROC curve image embed, ablation table.
- 4–6 case studies with figure embeds (2 authentic, 2 spliced, 1 copy-move, 1 honest failure).
- **Failure modes:** double JPEG, resizing, screenshots/social-media recompression, high-contrast edge false positives, GAN/diffusion deepfakes with no splice signal, adversarial laundering.
- **Legal & evidence integrity:**
  - Hash before analysis, work only on copies.
  - Deterministic, reproducible pipeline.
  - Tool output is an indicator, not proof.
  - BSA 2023 §63 requires a certificate and chain of custody; a model score alone is insufficient.
  - Human expert review required before any legal use.
- Limitations and future work: PRNU with camera fingerprint, pixel-level segmentation network, diffusion-specific detectors.
- Colab notebook link, one-minute demo with `samples/`.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Overclaiming deepfake detection (ELA is weak on fully synthetic images) | FFT feature + explicit limitation in README |
| Data leakage inflating metrics | Strict source-image split policy, document it |
| ELA signal destroyed by resize | No resize augmentation during training |
| Weights too large for GitHub | MobileNetV2 ≈ 10 MB — commit directly or use a GitHub Release |
| Reviewer can't reproduce results | Pinned `requirements.txt`, sample images in `samples/`, include example outputs |
| PNG input degrading ELA | Warn the user and note it in `analysis_notes` |

---

## Final Checklist

- [ ] `forensic_features.py` — all five signals working and unit-tested on a sample image
- [ ] `model.py` — both variants load without error; multi-channel weights init verified
- [ ] `train_or_eval_model.py` — training run completes, metrics printed and saved
- [ ] `forensic_audit.py` — runs end-to-end: `python forensic_audit.py --input samples/tampered.jpg --output_dir ./outputs/`
- [ ] JSON report contains all required keys and valid SHA-256
- [ ] Diagnostic figure saved at ≥150 dpi with all three panels
- [ ] `requirements.txt` is complete and pinned
- [ ] README metrics match actual results
- [ ] Repo is public and link works
