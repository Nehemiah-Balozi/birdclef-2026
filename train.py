"""
BirdCLEF+ 2026 — single-GPU training loop with mixed precision and CSV logging.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import GradScaler, autocast

from config import config, create_experiment_dirs
from dataset import get_dataloaders, mixup_batch
from model import FocalLoss, build_model, mixup_criterion

import torch.backends.cudnn as cudnn

cudnn.enabled = False


def set_seed(seed: int) -> None:
    """
    Fix RNG seeds and cuDNN flags for reproducible runs (best effort on GPU).

    Sets Python ``random``, NumPy, PyTorch CPU/CUDA, and enables deterministic
    algorithms where supported.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Macro-averaged ROC-AUC over classes that have at least one positive and
    one negative label in ``labels`` (other classes are skipped).
    """
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    n_c = labels.shape[1]
    scores: list[float] = []
    for j in range(n_c):
        y = labels[:, j]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos == 0 or neg == 0:
            continue
        if np.isnan(preds[:, j]).any():
            continue
        scores.append(float(roc_auc_score(y, preds[:, j])))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def compute_map(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Macro-averaged mean average precision; skips classes with no positive
    samples in ``labels``.
    """
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    n_c = labels.shape[1]
    scores: list[float] = []
    for j in range(n_c):
        y = labels[:, j]
        if int(y.sum()) == 0:
            continue
        if np.isnan(preds[:, j]).any():
            continue
        scores.append(float(average_precision_score(y, preds[:, j])))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


class CSVLogger:
    """Append one CSV row per epoch (creates file with header if missing)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._header = ["epoch", "train_loss", "val_loss", "val_auc", "val_map", "lr"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(self._header)

    def log_row(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_auc: float,
        val_map: float,
        lr: float,
    ) -> None:
        """Append a single training epoch summary row."""
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_auc:.6f}",
                    f"{val_map:.6f}",
                    f"{lr:.8f}",
                ]
            )


class Trainer:
    """
    One-fold trainer: AdamW + cosine schedule, AMP, focal BCE, mixup (optional).

    Parameters
    ----------
    fold :
        Validation fold index passed to :func:`dataset.get_dataloaders`.
    """

    def __init__(self, fold: int) -> None:
        self.fold = fold
        self.device = torch.device("cuda")
        self.model = build_model().to(self.device)
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=1e-6,
        )
        self.train_loader, self.val_loader = get_dataloaders(fold)
        self.scaler = GradScaler("cuda")
        self.best_auc = 0.0
        self.checkpoint_path = Path(config.paths.checkpoints_dir) / f"fold{fold}_best.pth"
        self._csv = CSVLogger(Path(config.paths.logs_dir) / f"fold{fold}_log.csv")

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """
        Run one training epoch with optional mixup (50% when enabled) and AMP.

        Returns
        -------
        dict
            ``train_loss`` (mean batch loss) and ``lr`` (optimizer param-group LR).
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        alpha = config.augmentation.mixup_alpha
        use_mixup = config.augmentation.use_mixup

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            specs = batch["melspec"].to(self.device)
            labels = batch["labels"].to(self.device)

            apply_mixup = (
                use_mixup and alpha > 0 and random.random() < 0.5
            )

            with autocast("cuda"):
                if apply_mixup:
                    mixed_specs, la, lb, lam = mixup_batch(
                        specs, labels, alpha, dual=True
                    )
                    logits = self.model(mixed_specs)
                    loss = mixup_criterion(self.criterion, logits, la, lb, lam)
                else:
                    logits = self.model(specs)
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        lr = float(self.optimizer.param_groups[0]["lr"])
        return {"train_loss": avg_loss, "lr": lr}

    def validate(self, epoch: int) -> dict[str, float]:
        """
        Evaluate on the validation loader; update best checkpoint by macro ROC-AUC.

        Returns
        -------
        dict
            ``val_loss``, ``val_auc``, ``val_map`` (NaNs possible if metrics undefined).
        """
        self.model.eval()
        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                specs = batch["melspec"].to(self.device)
                labels = batch["labels"].to(self.device)
                if labels.sum() == 0:
                    continue
                with autocast("cuda"):
                    logits = self.model(specs)
                    loss = self.criterion(logits, labels)
                losses.append(float(loss.detach()))
                all_logits.append(logits.float().cpu())
                all_labels.append(labels.float().cpu())

        val_loss = float(np.mean(losses)) if losses else float("nan")
        logits_cat = torch.cat(all_logits, dim=0)
        labels_np = torch.cat(all_labels, dim=0).numpy()
        preds = torch.sigmoid(logits_cat).numpy()
        auc = compute_auc(labels_np, preds)
        map_score = compute_map(labels_np, preds)

        if not np.isnan(auc) and auc > self.best_auc:
            self.best_auc = float(auc)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "auc": float(auc),
                    "map": float(map_score),
                    "config": config,
                },
                self.checkpoint_path,
            )

        return {"val_loss": val_loss, "val_auc": auc, "val_map": map_score}

    def fit(self) -> None:
        """Train for ``config.training.num_epochs``; log CSV rows and print summaries."""
        n_ep = config.training.num_epochs
        for epoch in range(n_ep):
            tr = self.train_one_epoch(epoch)
            va = self.validate(epoch)
            self.scheduler.step()

            self._csv.log_row(
                epoch=epoch + 1,
                train_loss=tr["train_loss"],
                val_loss=va["val_loss"],
                val_auc=va["val_auc"],
                val_map=va["val_map"],
                lr=tr["lr"],
            )

            print(
                f"Epoch {epoch + 1}/{n_ep} | "
                f"train_loss={tr['train_loss']:.4f} | "
                f"val_loss={va['val_loss']:.4f} | "
                f"val_auc={va['val_auc']:.4f} | "
                f"val_map={va['val_map']:.4f} | "
                f"lr={tr['lr']:.2e} | "
                f"best_auc={self.best_auc:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 training")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index")
    args = parser.parse_args()

    set_seed(config.training.seed)
    create_experiment_dirs()

    trainer = Trainer(fold=args.fold)
    trainer.fit()
    print(f"Best validation ROC-AUC (macro, fold {args.fold}): {trainer.best_auc:.6f}")


if __name__ == "__main__":
    main()
