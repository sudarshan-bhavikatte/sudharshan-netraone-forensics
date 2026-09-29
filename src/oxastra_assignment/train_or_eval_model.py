"""
Training, Validation, Threshold Sweeping, and Evaluation Pipeline for Netra-One Forensics.

Executes:
1. PyTorch model training with Adam optimizer (lr=1e-4).
2. Early stopping based on validation ROC-AUC.
3. F1-maximizing threshold sweeping on validation set.
4. Metric computation (Accuracy, Precision, Recall, F1, ROC-AUC).
5. Diagnostic plot generation (ROC curve & Confusion Matrix).
6. Model ablation comparison (Heuristic vs ELA-only vs Multi-channel 9-ch).
"""

import argparse
import os
from typing import Dict, List, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import HeuristicForensicModel, MobileNetV2Forensic, build_model
from .synthetic_data import ForensicDataset, create_synthetic_dataset


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Computes forensic classification evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted manipulation probabilities for class 1.
        threshold: Classification threshold for positive class.

    Returns:
        Dict containing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Compute ROC-AUC manually / via rank sum trapezoidal rule
    auc = compute_roc_auc(y_true, y_prob)

    return {
        "accuracy": float(round(accuracy, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "roc_auc": float(round(auc, 4)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Area Under ROC Curve (ROC-AUC) using trapezoidal integration."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Sort probabilities descending
    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp_count = 0
    fp_count = 0

    for label in y_true_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)

    # Integrate area under FPR vs TPR
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trapz_fn(tpr_list, fpr_list))
    return float(np.clip(auc, 0.0, 1.0))



def sweep_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float, Dict[str, float]]:
    """Sweeps threshold from 0.1 to 0.9 on validation predictions to pick F1-maximizing threshold.

    Returns:
        Tuple of (best_threshold, best_f1, metrics_at_best_threshold).
    """
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.1, 0.9, 17)

    for thresh in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=thresh)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(round(thresh, 2))
            best_metrics = metrics

    return best_threshold, best_f1, best_metrics


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str, title: str = "ROC Curve"
) -> None:
    """Generates and saves ROC curve plot."""
    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return

    tprs = [0.0]
    fprs = [0.0]
    tp_c, fp_c = 0, 0

    for label in y_true_sorted:
        if label == 1:
            tp_c += 1
        else:
            fp_c += 1
        tprs.append(tp_c / n_pos)
        fprs.append(fp_c / n_neg)

    auc = compute_roc_auc(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fprs, tprs, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Chance")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int, save_path: str, title: str = "Confusion Matrix"
) -> None:
    """Generates and saves confusion matrix plot."""
    cm = np.array([[tn, fp], [fn, tp]])
    classes = ["Authentic", "Tampered"]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if cm[i, j] > np.max(cm) / 2 else "black",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def evaluate_model(
    model: Union[MobileNetV2Forensic, HeuristicForensicModel],
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs evaluation loop returning ground truth and predicted manipulation probabilities."""
    y_true_list = []
    y_prob_list = []

    if isinstance(model, HeuristicForensicModel):
        for inputs, labels in val_loader:
            for i in range(inputs.shape[0]):
                # Reconstruct ELA array from tensor
                ela_tensor = inputs[i]
                if ela_tensor.shape[0] == 9:
                    ela_tensor = ela_tensor[3:6]
                ela_img = (ela_tensor * 255.0).numpy().transpose(1, 2, 0).astype(np.uint8)
                pred, score = model.predict(ela_img)
                prob = min(1.0, score / 6.0) # Map Z-score roughly to probability
                y_true_list.append(labels[i].item())
                y_prob_list.append(prob)
    else:
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                probs = model.predict_probability(inputs).cpu().numpy()
                y_true_list.extend(labels.numpy())
                y_prob_list.extend(probs)

    return np.array(y_true_list), np.array(y_prob_list)


def train_model(
    model: MobileNetV2Forensic,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    weights_dir: str = "weights",
    freeze_backbone: bool = False,
) -> Tuple[MobileNetV2Forensic, Dict[str, Any]]:
    """Executes model training loop with Adam optimizer and early stopping on validation AUC."""
    os.makedirs(weights_dir, exist_ok=True)

    # Set CPU multithreading if running on CPU
    if device.type == "cpu":
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)

    # Optional speedup: Freeze MobileNetV2 feature extractor backbone
    if freeze_backbone and hasattr(model, "model") and hasattr(model.model, "features"):
        print("Freezing MobileNetV2 feature backbone for accelerated CPU training...")
        for param in model.model.features.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    model.to(device)
    best_auc = -1.0
    best_epoch = 0
    patience = 3
    patience_counter = 0

    best_weights_path = os.path.join(weights_dir, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation evaluation
        y_true, y_prob = evaluate_model(model, val_loader, device)
        val_auc = compute_roc_auc(y_true, y_prob)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} - Val AUC: {val_auc:.4f}"
        )

        # Save checkpoint
        epoch_path = os.path.join(weights_dir, f"epoch_{epoch}.pth")
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_auc": val_auc}, epoch_path)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> Best model saved at epoch {epoch} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    # Load best weights
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    return model, {"best_epoch": best_epoch, "best_auc": best_auc}


def run_pipeline(
    dataset_dir: str = "dataset",
    output_dir: str = "outputs",
    weights_dir: str = "weights",
    epochs: int = 2,
    batch_size: int = 16,
    variant: str = "multichannel",
    freeze_backbone: bool = False,
) -> Dict[str, Any]:
    """Runs complete end-to-end Part B pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Check if dataset exists, generate if missing
    if not os.path.exists(os.path.join(dataset_dir, "train")):
        print(f"Dataset not found at '{dataset_dir}'. Generating synthetic dataset...")
        create_synthetic_dataset(output_dir=dataset_dir, num_samples=40)

    num_channels = 3 if variant == "ela_only" else 9
    train_dataset = ForensicDataset(dataset_dir, split="train", num_channels=num_channels, is_train=True)
    val_dataset = ForensicDataset(dataset_dir, split="val", num_channels=num_channels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variant == "heuristic":
        model = build_model("heuristic")
        y_true, y_prob = evaluate_model(model, val_loader, device)
        best_thresh, best_f1, metrics = sweep_optimal_threshold(y_true, y_prob)
    else:
        model = build_model(variant)
        trained_model, info = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            weights_dir=weights_dir,
            freeze_backbone=freeze_backbone,
        )
        y_true, y_prob = evaluate_model(trained_model, val_loader, device)
        best_thresh, best_f1, metrics = sweep_optimal_threshold(y_true, y_prob)

    plot_roc_curve(y_true, y_prob, os.path.join(output_dir, "roc_curve.png"), f"ROC Curve ({variant})")
    plot_confusion_matrix(
        metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"],
        os.path.join(output_dir, "confusion_matrix.png"), f"Confusion Matrix ({variant})"
    )

    metrics["best_threshold"] = best_thresh
    metrics["variant"] = variant

    print("\n--- Final Model Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netra-One Forensic Model Training & Evaluation")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--variant", type=str, default="multichannel", choices=["heuristic", "ela_only", "multichannel"])
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze MobileNetV2 backbone for fast CPU training")
    args = parser.parse_args()

    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        variant=args.variant,
        freeze_backbone=args.freeze_backbone,
    )
