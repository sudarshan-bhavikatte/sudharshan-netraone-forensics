import os
import torch
import torch.nn as nn
import torchvision.models as models


class NetraForensicClassifier(nn.Module):
    """
    Lightweight Forensic Image Classifier based on MobileNetV2.
    Supports standard 3-channel (RGB or ELA) and 9-channel multi-modal inputs:
    [RGB (3-ch) | ELA (3-ch) | Noise Residual (3-ch)].
    """

    def __init__(self, in_channels=3, pretrained=True, dropout_rate=0.3):
        super(NetraForensicClassifier, self).__init__()
        self.in_channels = in_channels

        # Load MobileNetV2 pretrained backbone
        weights = (
            models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        base_model = models.mobilenet_v2(weights=weights)

        # Adapt first convolutional layer if in_channels != 3
        if in_channels != 3:
            orig_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            with torch.no_grad():
                # Copy original pretrained weights to first 3 channels
                new_conv.weight[:, :3, :, :] = orig_conv.weight
                # Initialize additional channels with zeros or small random weights
                nn.init.kaiming_normal_(
                    new_conv.weight[:, 3:, :, :], nonlinearity="relu"
                )
                if orig_conv.bias is not None:
                    new_conv.bias = orig_conv.bias
            base_model.features[0][0] = new_conv

        self.features = base_model.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
        in_features = base_model.last_channel
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(64, 2),  # 2 classes: 0 = Authentic, 1 = Tampered
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.pooling(features)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def predict_probability(self, x):
        """Returns tampered class probability (0.0 to 1.0)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            # Probability of Class 1 (Tampered)
            return probs[:, 1].item() if probs.shape[0] == 1 else probs[:, 1]


def build_model(weights_path=None, in_channels=3, device="cpu"):
    """
    Factory function to construct the classifier and optionally load checkpoint weights.

    Args:
        weights_path: Path to PyTorch model .pth checkpoint.
        in_channels: 3 or 9.
        device: 'cpu' or 'cuda'.

    Returns:
        model: NetraForensicClassifier instance loaded onto target device.
    """
    model = NetraForensicClassifier(in_channels=in_channels, pretrained=True)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Model] Successfully loaded weights from {weights_path}")
    else:
        if weights_path:
            print(
                f"[Model Warning] Weights file '{weights_path}' not found. Using pretrained backbone baseline."
            )
        else:
            print(
                "[Model] Initialized pretrained MobileNetV2 backbone (no fine-tuned weights supplied)."
            )

    model.to(device)
    model.eval()
    return model
