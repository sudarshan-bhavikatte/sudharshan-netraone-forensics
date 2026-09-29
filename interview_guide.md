# Netra-One Forensics — Interview & Defense Guide

This guide tracks implementation progress across all parts and subparts defined in [`PLAN.md`](file:///c:/Users/zeus/projects/oxastra_assignment/PLAN.md) and provides technical depth, interview questions, and defense rationale for each component.

---

## Progress Overview

| Part | Subpart | Status | Focus Area |
|---|---|---|---|
| **Part A** | ELA (Error Level Analysis) | ✅ Completed | Re-compression difference, per-channel scaling, JET heatmap |
| **Part A** | Noise Residual | ✅ Completed | Median filter residual, 32×32 block variance HOT heatmap |
| **Part A** | FFT Spectrum | ✅ Completed | 2D FFT log-magnitude frequency VIRIDIS heatmap |
| **Part A** | Anomaly Score | ✅ Completed | Block-wise MAD robust z-score calculation |
| **Part A** | Localization / Bounding Box | ✅ Completed | Signal fusion, Otsu thresholding, morphology, contour bbox |
| **Part B** | Synthetic Data Generator | ✅ Completed | Splice, copy-move, inpaint generator, strict source-split |
| **Part B** | Models (Heuristic & MobileNetV2 9-ch) | ✅ Completed | Pretrained backbone, 9-ch weight patch, Heuristic MAD model |
| **Part B** | Training Config & Sweep | ✅ Completed | Adam optimizer, lr=1e-4, F1 threshold sweep (0.1–0.9) |
| **Part B** | Metrics & Ablation | ✅ Completed | Accuracy, Precision, Recall, F1, ROC-AUC, ROC & Confusion plots |
| **Part C** | Arguments & CLI Workflow | ✅ Completed | Argparse interface, error handling |
| **Part C** | SHA-256 Evidence Integrity | ✅ Completed | Streaming raw byte hashing before load |
| **Part C** | JSON Evidence Report | ✅ Completed | Standardized schema, BSA 2023 disclaimer |
| **Part C** | 3-Panel Diagnostic Figure | ✅ Completed | Matplotlib visualization (150+ DPI) |
| **Part C** | Edge Case Handling | ✅ Completed | Small images, PNG warnings, missing weights |
| **Part D** | Technical README & Defense | ✅ Completed | Math derivations, failure modes, legal framework |

---

## Detailed Part Breakdown & Defense Notes

### Part A: Signal Extraction ([`forensic_features.py`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_features.py))

#### 1. ELA (Error Level Analysis)
- **Status:** ✅ Completed ([`compute_ela`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_features.py#L18-L57))
- **Implementation:** Re-saves RGB image in-memory via `io.BytesIO` at JPEG quality=90, computes absolute per-pixel difference `|I_orig - I_recompressed|`, scales each channel independently to [0, 255], converts to grayscale magnitude, and generates a JET heatmap (`COLORMAP_JET`).
- **Key Equation:**  
  $$\text{ELA} = \text{clip}\left(|I_{\text{orig}} - I_{\text{recompressed}}| \times \frac{255}{\max(\text{diff})}, 0, 255\right)$$
- **Interview Q&A:**
  - *Q: Why does ELA fail on PNG or uncompressed inputs?*  
    *A: ELA relies on JPEG lossy quantization grid history. PNG is lossless, so re-saving a raw PNG to JPEG introduces uniform quantization error across the entire image.*
  - *Q: Why do we scale per-channel instead of globally?*  
    *A: Per-channel scaling prevents dominant primary colors (e.g. bright red objects) from masking subtle ELA anomalies in the green/blue channels.*

#### 2. Noise Residual
- **Status:** ✅ Completed ([`compute_noise_residual`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_features.py#L60-L107))
- **Implementation:** Applies a 5×5 median filter (`cv2.medianBlur`) to extract image noise residual $R = I - \text{denoised}$. Divides $R$ into non-overlapping 32×32 blocks, computes local variance per block to form a spatial variance map, and renders a HOT colormap visualization (`COLORMAP_HOT`).
- **Interview Q&A:**
  - *Q: How does noise residual detect splicing or inpainting?*  
    *A: Camera sensors leave uniform photon shot noise (PRNU). Splice insertions or AI inpainting alter or destroy this local noise variance.*

#### 3. FFT Spectrum
- **Status:** ✅ Completed ([`compute_fft_spectrum`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_features.py#L110-L147))
- **Implementation:** Computes 2D Fast Fourier Transform (`np.fft.fft2`) on grayscale image, shifts DC (zero-frequency) component to center (`np.fft.fftshift`), computes log-magnitude spectrum $\log(1 + |F|)$, normalizes to [0, 255], and outputs a VIRIDIS heatmap (`COLORMAP_VIRIDIS`).
- **Interview Q&A:**
  - *Q: What features in FFT point to deepfakes vs manual splices?*  
    *A: Resampling/rotation produces periodic peak grids in spatial frequency; GAN generator transposed convolutions leave high-frequency checkerboard grid spikes.*

#### 4. Anomaly Score
- **Status:** ✅ Completed ([`compute_anomaly_score`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_features.py#L150-L198))
- **Implementation:** Divides mean ELA magnitude into non-overlapping 32×32 blocks, computes block means $x$, calculates median $M$ and Median Absolute Deviation $\text{MAD} = \text{median}(|x - M|)$, and calculates robust Z-scores $Z = \frac{x - M}{\text{MAD} + 1e-6}$. Returns $\max(Z)$ rounded to 4 decimals.
- **Interview Q&A:**
  - *Q: Why use MAD instead of standard mean and standard deviation?*  
    *A: Standard mean and std dev are sensitive to extreme outliers. If a small spliced region has a huge ELA spike, it inflates the standard deviation and lowers normal Z-scores. MAD is robust against up to 50% outlier corruption.*

#### 5. Bounding Box Localization
- **Status:** ✅ Completed ([`extract_localization_bbox`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_features.py#L201-L266))
- **Implementation:** Fuses normalized ELA gray map and noise variance map ($0.5 \times \text{ELA}_{\text{norm}} + 0.5 \times \text{Noise}_{\text{norm}}$). Binarizes using Otsu's thresholding (`cv2.THRESH_OTSU`), applies 5×5 morphological OPEN then CLOSE to eliminate speckle noise and bridge gaps, and extracts the largest contour's bounding box `[x, y, w, h]`.
- **Interview Q&A:**
  - *Q: Why fuse ELA and Noise Residual before thresholding?*  
    *A: Spliced patches often leave both re-compression ELA anomalies AND noise variance discrepancies. Fusing both signals yields higher signal-to-noise ratio and reduces false positives from high-contrast edges.*

---

### Part B: Data & Model ([`synthetic_data.py`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/synthetic_data.py), [`model.py`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/model.py), [`train_or_eval_model.py`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/train_or_eval_model.py))

#### 1. Synthetic Data Generator & Strict Split
- **Status:** ✅ Completed ([`generate_synthetic_tamper`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/synthetic_data.py#L22-L99), [`ForensicDataset`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/synthetic_data.py#L173-L245))
- **Implementation:** Generates procedurally tampered images (Splice, Copy-Move, and Gaussian Inpainting) along with binary ground-truth masks. Applies random JPEG quality re-compression (70–95). Splits source images strictly into `train/` and `val/` directories before patch generation to ensure zero data leakage.
- **Interview Q&A:**
  - *Q: Why is strict source-image splitting critical in forensic AI?*  
    *A: If patches from the same background photo appear in both train and validation sets, the neural network memorizes background scene features rather than learning compression/noise forgery artifacts (data leakage).*

#### 2. Model Architectures & 9-Channel Weight Patching
- **Status:** ✅ Completed ([`MobileNetV2Forensic`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/model.py#L40-L101), [`HeuristicForensicModel`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/model.py#L18-L37))
- **Implementation:**
  - **Heuristic Baseline:** Non-trainable MAD Z-score thresholding classifier.
  - **MobileNetV2 (9-Channel Multi-Input):** Concatenates RGB (3 ch) + ELA (3 ch) + Noise Residual (3 ch). Patches first convolutional layer `Conv2d(9, 32)`: copies pretrained ImageNet weights to channels 0..2 (`[:, :3, :, :]`), and initializes channels 3..8 (`[:, 3:, :, :]`) to `0.0`.
- **Interview Q&A:**
  - *Q: Why initialize the extra 6 channels to zero instead of random Gaussian?*  
    *A: Zero initialization guarantees that on epoch 0, step 0, the 9-channel network outputs the exact same logits as standard ImageNet pretrained MobileNetV2, preserving feature representations without early gradient destruction.*

#### 3. Training Config, Early Stopping & Threshold Sweeping
- **Status:** ✅ Completed ([`train_model`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/train_or_eval_model.py#L240-L315), [`sweep_optimal_threshold`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/train_or_eval_model.py#L107-L127))
- **Implementation:** Uses Adam optimizer (`lr=1e-4`, `weight_decay=1e-5`), horizontal flip augmentation only (no resize/recompression during training to preserve ELA grid). Monitors validation ROC-AUC every epoch with early stopping patience=3. Sweeps decision thresholds from 0.1 to 0.9 on validation set probabilities to pick F1-maximizing threshold.
- **Interview Q&A:**
  - *Q: Why avoid random resize or re-compression augmentations during training?*  
    *A: Resize and JPEG re-compression erase fine DCT quantization grids and ELA signals. Applying them as random augmentations would corrupt the physical forensic signal the model is being trained to detect.*

#### 4. Metrics, Diagnostics & Ablation Analysis
- **Status:** ✅ Completed ([`compute_metrics`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/train_or_eval_model.py#L22-L68), [`plot_roc_curve`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/train_or_eval_model.py#L130-L167), [`plot_confusion_matrix`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/train_or_eval_model.py#L170-L203))
- **Implementation:** Evaluates Accuracy, Precision, Recall, F1 score, and ROC-AUC (using NumPy 2.0 trapezoidal integration). Automatically renders high-DPI (150+ DPI) diagnostic plots for `outputs/roc_curve.png` and `outputs/confusion_matrix.png`. Supports ablation comparison across Heuristic vs ELA-only vs Multi-channel models.

---

### Part C: Forensic Audit CLI ([`forensic_audit.py`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_audit.py))

#### 1. SHA-256 Evidence Integrity
- **Status:** ✅ Completed ([`compute_file_sha256`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_audit.py#L25-L31))
- **Implementation:** Streams raw file bytes in 64KB chunks using `hashlib.sha256()` *before* any image decoding to preserve forensic chain of custody. Also hashes model weights `.pth` file if provided (`model_weights_sha256`).

#### 2. JSON Evidence Report & BSA 2023 §63 Compliance
- **Status:** ✅ Completed ([`run_forensic_audit`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_audit.py#L165-L270))
- **Implementation:** Exports standardized evidence JSON containing image SHA-256 hash, model weights SHA-256, tamper verdict (`TAMPERED | AUTHENTIC | INCONCLUSIVE`), manipulation probability, suspected region bbox, anomaly score, UTC timestamp, ELA quality, analysis notes, and BSA 2023 §63 statutory disclaimer.

#### 3. 3-Panel Diagnostic Figure & Edge Case Handling
- **Status:** ✅ Completed ([`generate_diagnostic_figure`](file:///c:/Users/zeus/projects/oxastra_assignment/src/oxastra_assignment/forensic_audit.py#L90-L162))
- **Implementation:** Renders high-DPI (150+ DPI) 3-panel figure (Original with bounding box, ELA heatmap, and Noise Variance heatmap). Handles non-JPEG inputs (emits warning), 2D grayscale images (converts to RGB), small images (<64x64, raises error), corrupted files (catches exception), and omitted weights (runs Heuristic Baseline mode).

---

### Part D: README & Final Defense ([`README.md`](file:///c:/Users/zeus/projects/oxastra_assignment/README.md))

- **Status:** ✅ Completed ([`README.md`](file:///c:/Users/zeus/projects/oxastra_assignment/README.md))
- **Implementation:** Comprehensive technical README containing system architecture Mermaid flowchart, mathematical derivations (ELA, Noise Residual, FFT, MAD Z-score), strict source-split data policy, 9-channel weight patching strategy, empirical ablation table, embedded ROC and confusion matrix diagnostic plots, sample audit figure embed, legal framework analysis (BSA 2023 §63), failure modes & vulnerabilities, and quickstart reproduction instructions.
