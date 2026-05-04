"""
BirdCLEF+ 2026 — GeM-pooled backbone, multilabel head, focal BCE loss helpers.

No training loop; only modules and factory utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import config, create_experiment_dirs


class GeMPooling(nn.Module):
    """
    Generalized mean pooling over spatial dimensions (height, width).

    Uses a learnable exponent ``p`` (initialized from ``config.model.gem_p``).
    For feature maps ``(N, C, H, W)``, computes a per-channel pooled vector
    ``(N, C)`` via the GeM formula: ``mean(clamp(x, min=eps)^p)^(1/p)``.
    """

    def __init__(self, p: float | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        p_init = float(config.model.gem_p if p is None else p)
        self.p = nn.Parameter(torch.tensor([p_init], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x :
            Tensor shaped ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Tensor shaped ``(batch, channels)``.
        """
        p = self.p
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=(-2, -1))
        return x.pow(1.0 / p)


class BirdCLEFModel(nn.Module):
    """
    Multilabel classifier: 1-channel timm backbone, GeM pooling, MLP head.

    Outputs **raw logits** (no sigmoid). Use ``BCEWithLogits``-style losses.
    """

    def __init__(self, pretrained: bool | None = None) -> None:
        super().__init__()
        m = config.model
        pt = m.pretrained if pretrained is None else pretrained
        self.backbone = timm.create_model(
            m.backbone,
            pretrained=pt,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )
        feat_dim = int(self.backbone.num_features)
        self.pool = GeMPooling()
        d = m.dropout
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=d),
            nn.Linear(512, m.num_classes),
        )

    def get_feature_dim(self) -> int:
        """Return the backbone channel dimension before pooling / head."""
        return int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x :
            Input tensor ``(N, 1, H, W)`` (e.g. single-channel spectrogram).

        Returns
        -------
        torch.Tensor
            Logits ``(N, num_classes)``.
        """
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class FocalLoss(nn.Module):
    """
    Focal loss on top of per-element binary cross-entropy with logits.

    Supports **soft** targets in ``[0, 1]`` (multilabel). ``pt`` is defined as
    ``exp(-bce)`` per user specification.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits :
            Raw logits, same shape as ``targets``.
        targets :
            Soft float targets in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Scalar loss if ``reduction`` is ``'mean'`` or ``'sum'``, else
            tensor matching element-wise layout of ``logits``.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Linearly combine two supervision targets under mixup (scalar loss).

    ``lam * loss(logits, labels_a) + (1 - lam) * loss(logits, labels_b)``
    """
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def build_model(pretrained: bool | None = None) -> BirdCLEFModel:
    """
    Construct :class:`BirdCLEFModel` from ``config.model``.

    Ensures experiment output directories exist via
    :func:`config.create_experiment_dirs`.

    Parameters
    ----------
    pretrained :
        If ``None``, uses ``config.model.pretrained``. Otherwise overrides
        the timm ``pretrained`` flag for the backbone weights.
    """
    create_experiment_dirs()
    return BirdCLEFModel(pretrained=pretrained)
