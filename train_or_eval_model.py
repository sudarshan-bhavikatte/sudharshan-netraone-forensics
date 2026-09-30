import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ForensicDataset, create_mini_dataset
from model import NetraForensicClassifier, build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    data_dir,
    output_dir="./outputs",
    weights_dir="./weights",
    epochs=8,
    batch_size=16,
    lr=1e-4,
    in_channels=3,
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    auth_dir = os.path.join(data_dir, "authentic")
    tamp_dir = os.path.join(data_dir, "tampered")

    if not os.path.exists(auth_dir) or not os.path.exists(tamp_dir):
        print(f"[Training] Dataset directory '{data_dir}' not found. Generating synthetic dataset...")
        create_mini_dataset(data_dir, num_authentic=100, num_tampered=100)

    samples = []
    for fname in os.listdir(auth_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((os.path.join(auth_dir, fname), 0))

    for fname in os.listdir(tamp_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((os.path.join(tamp_dir, fname), 1))

    print(f"[Training] Loaded {len(samples)} samples from {data_dir}")

    labels = [s[1] for s in samples]
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, random_state=42, stratify=labels, shuffle=True
    )

    print(f"[Training] Stratified Split: {len(train_samples)} Train | {len(val_samples)} Validation")

    train_ds = ForensicDataset(train_samples, in_channels=in_channels, is_train=True)
    val_ds = ForensicDataset(val_samples, in_channels=in_channels, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = NetraForensicClassifier(in_channels=in_channels, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_score = -1.0
    best_weights_path = os.path.join(weights_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_ds)

        val_loss, y_true, y_probs, y_preds = evaluate_performance(model, val_loader, criterion, device)
        val_acc = accuracy_score(y_true, y_preds)
        val_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
        )

        score = val_auc + 0.1 * val_acc
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_weights_path)
            print(f" Saved new best checkpoint to {best_weights_path} (Val AUC: {val_auc:.4f})")

    model.load_state_dict(torch.load(best_weights_path))
    _, y_true, y_probs, y_preds = evaluate_performance(model, val_loader, criterion, device)

    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("=" * 50)

    save_roc_curve(y_true, y_probs, auc, os.path.join(output_dir, "roc_curve.png"))

    save_confusion_matrix(y_true, y_preds, os.path.join(output_dir, "confusion_matrix.png"))

    return best_weights_path


def evaluate_performance(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_probs = []
    y_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(targets.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())

    val_loss = val_loss / len(dataloader.dataset)
    return val_loss, np.array(y_true), np.array(y_probs), np.array(y_preds)


def save_roc_curve(y_true, y_probs, auc_score, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'Netra-One MobileNetV2 (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Metrics] Saved ROC curve to {output_path}")


def save_confusion_matrix(y_true, y_preds, output_path):
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(5, 4), dpi=150)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Forensic Detection Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Authentic', 'Tampered'])
    plt.yticks(tick_marks, ['Authentic', 'Tampered'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Verdict')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Metrics] Saved Confusion Matrix to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Netra-One Forensic Classifier")
    parser.add_argument("--data_dir", type=str, default="./data/surveillance_mini", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for plots")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels (3 for ELA, 9 for Multi-modal)")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
    )

