"""Step 3: Modular TinySleepNet Architecture in PyTorch (`src/models/tinysleepnet.py`).

References:
- Supratak & Guo (2020), "TinySleepNet: An Efficient Deep Learning Model for Sleep Stage
  Scoring Based on Raw Single-Channel EEG" (EMBC 2020).
- van der Aar et al. (2024), Section 2.2 (p. 4) & van der Aar et al. (2025), p. 316.
- Exact layer specifications in `PROJECT_SPEC.md`.

Key Design Features:
1. Parameterized `in_channels` (`C >= 1`) so the exact same architecture runs on single-lead
   wearable simulations (`C = 1`), dual-lead (`C = 2`), and multi-channel `full_psg` (`C = 5`).
2. Clean separation between `feature_extractor` (`RepresentationLearning` CNN) and
   `classifier` (`ContextBlindHead` for `use_context=False` vs. `SequentialLSTMHead` for
   `use_context=True`), enabling seamless checkpoint transfer and Feature Matching in Phase 2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    """1D Convolution with TensorFlow-style `'SAME'` padding for any stride `>= 1`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        out_length = math.ceil(length / self.stride)
        pad_total = max((out_length - 1) * self.stride + self.kernel_size - length, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ConvBlock1D(nn.Module):
    """Single TinySleepNet 1D Convolutional block: `SamePadConv1d -> BatchNorm1d -> ReLU`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = SamePadConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RepresentationLearningCNN(nn.Module):
    """Part 1: TinySleepNet CNN Feature Extractor (`RepresentationLearning`).

    Follows `PROJECT_SPEC.md`:
    - Conv1: 128 filters, kernel_size = 50 (Fs/2), stride = 6 (Fs/16), padding = 'same' + BN + ReLU
    - MaxPool1: kernel_size = 8, stride = 8
    - Dropout1: p = 0.5
    - Conv2: 128 filters, kernel_size = 8, stride = 1, padding = 'same' + BN + ReLU
    - Conv3: 128 filters, kernel_size = 8, stride = 1, padding = 'same' + BN + ReLU
    - Conv4: 128 filters, kernel_size = 8, stride = 1, padding = 'same' + BN + ReLU
    - MaxPool2: kernel_size = 4, stride = 4
    - Dropout2: p = 0.5
    - Flatten -> 1D feature vector `f_t` per 30-second epoch.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_filters: int = 128,
        conv1_kernel: int = 50,
        conv1_stride: int = 6,
        conv_kernel: int = 8,
        pool1: int = 8,
        pool2: int = 4,
        dropout: float = 0.5,
        epoch_samples: int = 3000,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.epoch_samples = int(epoch_samples)

        self.conv1 = ConvBlock1D(
            in_channels=self.in_channels,
            out_channels=conv_filters,
            kernel_size=conv1_kernel,
            stride=conv1_stride,
        )
        self.pool1 = nn.MaxPool1d(kernel_size=pool1, stride=pool1)
        self.drop1 = nn.Dropout(p=dropout)

        self.conv2 = ConvBlock1D(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            stride=1,
        )
        self.conv3 = ConvBlock1D(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            stride=1,
        )
        self.conv4 = ConvBlock1D(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            stride=1,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=pool2, stride=pool2)
        self.drop2 = nn.Dropout(p=dropout)

        # Dynamically determine flattened feature dimension `feat_dim`
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.epoch_samples)
            feat = self._forward_features(dummy)
            self.feat_dim = int(feat.shape[1])

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1D feature vector `f_t` of shape `(Batch, feat_dim)` from `(Batch, C, 3000)`."""
        return self._forward_features(x)


class ContextBlindHead(nn.Module):
    """Part 2: Model A Context-Blind Classification Head (`use_context = False`).

    Maps single-epoch CNN feature vector `f_t` `(Batch, feat_dim)` -> `(Batch, n_classes)`.
    """

    def __init__(self, feat_dim: int, n_classes: int = 5, dropout: float = 0.5) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(features))


class SequentialLSTMHead(nn.Module):
    """Part 3: Model B Context-Aware Sequential LSTM Head (`use_context = True`).

    Takes a sequence of `M` epoch feature vectors `(Batch, M, feat_dim)`, passes them
    through a single unidirectional LSTM layer (`hidden_size = 128`) + `Dropout(0.5)`,
    and predicts sleep stages `(Batch, M, n_classes)`.
    """

    def __init__(
        self,
        feat_dim: int,
        lstm_hidden: int = 128,
        n_classes: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(seq_features)  # (Batch, M, lstm_hidden)
        out = self.drop(lstm_out)
        return self.fc(out)  # (Batch, M, n_classes)


class TinySleepNet(nn.Module):
    """Modular TinySleepNet supporting both Model A (`use_context=False`) and Model B (`use_context=True`)."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 5,
        use_context: bool = False,
        conv_filters: int = 128,
        conv1_kernel: int = 50,
        conv1_stride: int = 6,
        conv_kernel: int = 8,
        pool1: int = 8,
        pool2: int = 4,
        lstm_hidden: int = 128,
        dropout: float = 0.5,
        epoch_samples: int = 3000,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_classes = int(n_classes)
        self.use_context = bool(use_context)
        self.epoch_samples = int(epoch_samples)

        self.feature_extractor = RepresentationLearningCNN(
            in_channels=self.in_channels,
            conv_filters=conv_filters,
            conv1_kernel=conv1_kernel,
            conv1_stride=conv1_stride,
            conv_kernel=conv_kernel,
            pool1=pool1,
            pool2=pool2,
            dropout=dropout,
            epoch_samples=self.epoch_samples,
        )
        self.feat_dim = self.feature_extractor.feat_dim

        if not self.use_context:
            self.classifier: nn.Module = ContextBlindHead(
                feat_dim=self.feat_dim,
                n_classes=self.n_classes,
                dropout=dropout,
            )
        else:
            self.classifier = SequentialLSTMHead(
                feat_dim=self.feat_dim,
                lstm_hidden=lstm_hidden,
                n_classes=self.n_classes,
                dropout=dropout,
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN representations `f_t`.

        - If `x` has shape `(Batch, C, T)`, returns `(Batch, feat_dim)`.
        - If `x` has shape `(Batch, Seq_Len, C, T)`, returns `(Batch, Seq_Len, feat_dim)`.
        """
        if x.ndim == 3:
            return self.feature_extractor(x)
        elif x.ndim == 4:
            b, s, c, t = x.shape
            flat_x = x.reshape(b * s, c, t)
            flat_feat = self.feature_extractor(flat_x)
            return flat_feat.reshape(b, s, self.feat_dim)
        else:
            raise ValueError(f"Expected 3D (B, C, T) or 4D (B, M, C, T) input tensor, got {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN feature extractor and classification head.

        - When `use_context == False`:
          Accepts `(Batch, C, T)` -> returns logits `(Batch, n_classes)`.
        - When `use_context == True`:
          Accepts `(Batch, M, C, T)` -> returns logits `(Batch, M, n_classes)`.
          (If given 3D `(Batch, C, T)`, treats `Batch` as a single sequence `(1, Batch, C, T)`
           and returns `(Batch, n_classes)` for convenience during full-night inference).
        """
        if not self.use_context:
            if x.ndim == 4:
                b, s, c, t = x.shape
                feats = self.feature_extractor(x.reshape(b * s, c, t))
                logits = self.classifier(feats)
                return logits.reshape(b, s, self.n_classes)
            feats = self.feature_extractor(x)
            return self.classifier(feats)
        else:
            if x.ndim == 3:
                # Treat (N_epochs, C, T) as a single contiguous sequence (1, N_epochs, C, T)
                feats = self.feature_extractor(x).unsqueeze(0)  # (1, N, feat_dim)
                logits = self.classifier(feats)  # (1, N, n_classes)
                return logits.squeeze(0)  # (N, n_classes)
            elif x.ndim == 4:
                seq_feats = self.extract_features(x)  # (Batch, M, feat_dim)
                return self.classifier(seq_feats)  # (Batch, M, n_classes)
            else:
                raise ValueError(f"Unexpected input tensor shape: {x.shape}")

    def get_checkpoint_payload(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Export model weights with separated CPU `feature_extractor` and `classifier` state dicts."""
        payload: dict[str, Any] = {
            "in_channels": self.in_channels,
            "n_classes": self.n_classes,
            "use_context": self.use_context,
            "feat_dim": self.feat_dim,
            "epoch_samples": self.epoch_samples,
            "feature_extractor_state_dict": {
                k: v.detach().cpu() for k, v in self.feature_extractor.state_dict().items()
            },
            "classifier_state_dict": {
                k: v.detach().cpu() for k, v in self.classifier.state_dict().items()
            },
            "model_state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }
        if extra_metadata:
            payload.update(extra_metadata)
        return payload

    def load_from_checkpoint_payload(
        self,
        payload: dict[str, Any],
        load_classifier: bool = True,
        strict: bool = True,
    ) -> None:
        """Load weights from a saved checkpoint payload (supports partial loading for Transfer Learning)."""
        self.feature_extractor.load_state_dict(
            payload["feature_extractor_state_dict"], strict=strict
        )
        if load_classifier and "classifier_state_dict" in payload:
            self.classifier.load_state_dict(payload["classifier_state_dict"], strict=strict)


def build_tinysleepnet_from_config(
    config: dict[str, Any],
    in_channels: int = 1,
    use_context_override: bool | None = None,
) -> TinySleepNet:
    """Instantiate `TinySleepNet` from the `'model'` section of `base_config.yaml`."""
    m_cfg = config.get("model", {})
    prep_cfg = config.get("preprocessing", {})
    target_fs = int(prep_cfg.get("target_fs", 100))
    epoch_sec = int(prep_cfg.get("epoch_sec", 30))
    epoch_samples = target_fs * epoch_sec

    use_context = (
        bool(m_cfg.get("use_context", False))
        if use_context_override is None
        else bool(use_context_override)
    )

    return TinySleepNet(
        in_channels=in_channels,
        n_classes=int(m_cfg.get("n_classes", 5)),
        use_context=use_context,
        conv_filters=int(m_cfg.get("conv_filters", 128)),
        conv1_kernel=int(m_cfg.get("conv1_kernel", 50)),
        conv1_stride=int(m_cfg.get("conv1_stride", 6)),
        conv_kernel=int(m_cfg.get("conv_kernel", 8)),
        pool1=int(m_cfg.get("pool1", 8)),
        pool2=int(m_cfg.get("pool2", 4)),
        lstm_hidden=int(m_cfg.get("lstm_hidden", 128)),
        dropout=float(m_cfg.get("dropout", 0.5)),
        epoch_samples=epoch_samples,
    )
