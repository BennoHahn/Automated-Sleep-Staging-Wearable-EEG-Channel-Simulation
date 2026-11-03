"""Phase 2 Secondary Model Interfaces (`src/models/future_models.py`).

Stubs/interfaces for secondary architectures referenced in `PROJECT_INSTRUCTIONS.md`:
- `DeepSleepNet`: Dual-branch CNN (small + large filter sizes) + 2-layer BiLSTM
  (Supratak et al., 2017; Waters & Clifford, 2022).
- `SeqSleepNet`: Filterbank + Epoch-level BiRNN + Sequence-level BiRNN
  (Phan et al., 2019; Heremans et al., 2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseSleepModel(nn.Module):
    """Abstract interface shared by all sleep staging architectures across Phase 1 and Phase 2."""

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DeepSleepNetStub(BaseSleepModel):
    """Interface stub for DeepSleepNet (Supratak et al., 2017; Waters & Clifford, 2022)."""

    def __init__(self, in_channels: int = 1, n_classes: int = 5, use_context: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.use_context = use_context


class SeqSleepNetStub(BaseSleepModel):
    """Interface stub for SeqSleepNet (Phan et al., 2019; Heremans et al., 2022)."""

    def __init__(self, in_channels: int = 1, n_classes: int = 5, seq_len: int = 20) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.seq_len = seq_len
