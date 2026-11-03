"""Phase 2 Transfer Learning Interfaces & Loss Functions (`src/training/transfer.py`).

Implements the clean interfaces and mathematical loss functions for Phase 2 Transfer Learning:
1. Direct Transfer (zero-shot evaluation of pre-trained PSG checkpoint on wearable channels,
   van der Aar et al., 2025).
2. Training From Scratch (randomly initialized TinySleepNet on wearable channels).
3. Fine-Tuning:
   - `'full'`: All CNN + classifier parameters trainable at reduced `lr = 1e-5`
     (van der Aar et al., 2024; Waters & Clifford, 2022).
   - `'head_only'`: Freeze `feature_extractor` CNN layers, train only `classifier` head
     (Waters & Clifford, 2022).
4. Paired Feature Matching (Heremans et al., 2022, Eq. 1, p. 5):
   `L = L_CE(y_hat_S, y_S) + L_CE(y_hat_T, y_T) + lambda_1 * MSE(f_S, f_T) + lambda_2 * ||W_T - W_S0||_2^2`
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tinysleepnet import TinySleepNet


def configure_finetuning_mode(model: TinySleepNet, mode: str = "full") -> None:
    """Configure parameter `requires_grad` flags for `'full'` vs `'head_only'` fine-tuning."""
    mode_lower = mode.lower()
    if mode_lower == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode_lower == "head_only":
        for p in model.feature_extractor.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported fine-tuning mode '{mode}'. Use 'full' or 'head_only'.")


class FeatureMatchingLoss(nn.Module):
    """Paired Feature Matching loss from Heremans et al. (2022), Eq. 1 (p. 5).

    Total Loss:
        L = L_C(y_hat_S, y_S) + L_C(y_hat_T, y_T) + lambda_1 * MSE(f_S, f_T) + lambda_2 * L2(W_feat_T, W_feat_init)
    """

    def __init__(
        self,
        source_model: TinySleepNet,
        lambda_1: float = 1.0,
        lambda_2: float = 1e-3,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.initial_feat_params = [
            p.detach().clone() for p in source_model.feature_extractor.parameters()
        ]

    def forward(
        self,
        logits_source: torch.Tensor,
        y_source: torch.Tensor,
        logits_target: torch.Tensor,
        y_target: torch.Tensor,
        feat_source: torch.Tensor,
        feat_target: torch.Tensor,
        target_model: TinySleepNet,
    ) -> torch.Tensor:
        loss_c_s = self.ce(logits_source.reshape(-1, logits_source.shape[-1]), y_source.reshape(-1))
        loss_c_t = self.ce(logits_target.reshape(-1, logits_target.shape[-1]), y_target.reshape(-1))
        loss_fm = F.mse_loss(feat_target, feat_source.detach())

        l2_reg = torch.tensor(0.0, device=feat_target.device)
        for p_curr, p_init in zip(
            target_model.feature_extractor.parameters(), self.initial_feat_params
        ):
            l2_reg = l2_reg + torch.sum((p_curr - p_init.to(p_curr.device)) ** 2)

        return loss_c_s + loss_c_t + self.lambda_1 * loss_fm + self.lambda_2 * l2_reg
