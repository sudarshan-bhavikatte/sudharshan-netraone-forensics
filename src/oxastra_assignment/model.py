"""
Model Architecture Module for Netra-One Forensics.

Provides:
1. Heuristic baseline classifier (rule-based MAD z-score).
2. MobileNetV2 ELA-only classifier (3-channel input).
3. MobileNetV2 Multi-channel classifier (9-channel input with 3-ch weight copy + 6-ch zero initialization patch).
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .forensic_features import compute_anomaly_score, compute_ela


class HeuristicForensicModel:
    """Non-trainable rule-based baseline model using robust MAD ELA Z-score thresholding."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def predict(self, ela_scaled: np.ndarray) -> Tuple[int, float]:
        """Predicts binary tamper verdict (1=TAMPERED, 0=AUTHENTIC) based on anomaly score threshold.

        Args:
            ela_scaled: Scaled ELA image map (RGB uint8).

        Returns:
            Tuple of (prediction: int, anomaly_score: float).
        """
        score = compute_anomaly_score(ela_scaled)
        pred = 1 if score >= self.threshold else 0
        return pred, score


class MobileNetV2Forensic(nn.Module):
    """MobileNetV2 classifier modified for image forensic tamper detection.

    Supports:
        - num_channels=3: Standard ELA input.
        - num_channels=9: Concatenated [RGB, ELA, Noise Residual] multi-channel input.
    """

    def __init__(
        self, num_channels: int = 9, num_classes: int = 2, dropout: float = 0.3
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        # Load ImageNet pretrained backbone
        weights = MobileNet_V2_Weights.DEFAULT
        backbone = mobilenet_v2(weights=weights)


        if num_channels != 3:
            old_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            # Copy ImageNet pretrained weights for first 3 channels
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                # Initialize remaining channels (4..9) to zero
                new_conv.weight[:, 3:, :, :] = 0.0

            backbone.features[0][0] = new_conv

        # Replace classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (N, 2)."""
        return self.model(x)

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning manipulation probability (class 1 probability)."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]


def build_model(
    variant: str = "multichannel", weights_path: Optional[str] = None
) -> Union[MobileNetV2Forensic, HeuristicForensicModel]:
    """Factory function to build forensic classifier model.

    Args:
        variant: Model variant ('heuristic', 'ela_only', or 'multichannel').
        weights_path: Optional path to pre-trained .pth checkpoint weights file.

    Returns:
        Instantiated PyTorch model or Heuristic baseline model.
    """
    variant = variant.lower()

    if variant == "heuristic":
        return HeuristicForensicModel(threshold=3.0)

    elif variant == "ela_only":
        model = MobileNetV2Forensic(num_channels=3)
    elif variant == "multichannel":
        model = MobileNetV2Forensic(num_channels=9)
    else:
        raise ValueError(
            f"Unknown model variant: {variant}. Choice of 'heuristic', 'ela_only', 'multichannel'."
        )

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Support loading state_dict directly or wrapped inside checkpoint dict
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    return model
