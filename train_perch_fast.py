"""
Train only the classification head on pre-extracted PERCH embeddings (.npz).

No audio I/O — same splits / focal loss / AUC / mAP / cosine LR as train.py pattern.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from config import config, create_experiment_dirs
from dataset import (
    _add_filepath_column,
    _apply_xc_low_rating_filter,
    _parse_soundscape_time_to_seconds,
    _site_group_key,
    _stratified_split_indices,
    mixup_batch,
)
from model import FocalLoss, mixup_criterion

import torch.backends.cudnn as cudnn

cudnn.enabled = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
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


class EmbeddingHead(nn.Module):
    """Linear(1536→512) → BN → ReLU → Dropout → Linear(512, num_classes)."""

    def __init__(self) -> None:
        super().__init__()
        n_cls = config.model.num_classes
        d = config.model.dropout
        self.net = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=d),
            nn.Linear(512, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_embedding_splits(
    fold: int,
    xc_path: Path,
    sc_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rebuild train/val tensors matching ``get_dataloaders`` / ``get_dataloaders_perch`` splits.
    """
    z_xc = np.load(xc_path, allow_pickle=True)
    z_sc = np.load(sc_path, allow_pickle=True)

    if not np.all(z_xc["completed"]):
        raise RuntimeError("XC embeddings incomplete — finish extract_perch_embeddings.py first.")
    if not np.all(z_sc["completed"]):
        raise RuntimeError(
            "Soundscape embeddings incomplete — finish extract_perch_embeddings.py first."
        )

    df = pd.read_csv(config.paths.train_csv)
    df = _add_filepath_column(df)
    df = _apply_xc_low_rating_filter(df).reset_index(drop=True)
    df["site_group"] = [_site_group_key(a, b) for a, b in zip(df["latitude"], df["longitude"])]

    fp_np = z_xc["filepaths"].astype(str)
    df_fp = df["filepath"].astype(str).values
    if len(fp_np) != len(df_fp) or not np.all(fp_np == df_fp):
        raise RuntimeError("xc_embeddings.npz does not match current train.csv + filter ordering.")

    train_idx, val_idx = _stratified_split_indices(
        df,
        n_folds=config.training.n_folds,
        val_fold=fold,
        seed=config.training.seed,
    )
    tr_xc = np.isin(np.arange(len(df)), train_idx)
    va_xc = np.isin(np.arange(len(df)), val_idx)

    scape = pd.read_csv(config.paths.soundscape_labels)
    scape = scape.drop_duplicates(subset=["filename", "start", "end"], keep="first").reset_index(
        drop=True
    )
    m = len(scape)
    if z_sc["embeddings"].shape[0] != m:
        raise RuntimeError("soundscape_embeddings.npz row count does not match deduped labels CSV.")

    for i in range(m):
        row = scape.iloc[i]
        if str(z_sc["filename"][i]) != str(row["filename"]):
            raise RuntimeError("soundscape npz row order mismatch (filename).")
        t0 = float(_parse_soundscape_time_to_seconds(row["start"]))
        t1 = float(_parse_soundscape_time_to_seconds(row["end"]))
        if not np.isclose(float(z_sc["start_sec"][i]), t0, rtol=0, atol=1e-2):
            raise RuntimeError("soundscape npz row order mismatch (start).")
        if not np.isclose(float(z_sc["end_sec"][i]), t1, rtol=0, atol=1e-2):
            raise RuntimeError("soundscape npz row order mismatch (end).")

    scape_files = sorted(scape["filename"].astype(str).unique().tolist())
    random.seed(config.training.seed)
    random.shuffle(scape_files)
    n_train_files = int(len(scape_files) * 0.8)
    train_sc_files = set(scape_files[:n_train_files])
    val_sc_files = set(scape_files[n_train_files:])

    fn = z_sc["filename"].astype(str)
    tr_sc = np.array([fn[i] in train_sc_files for i in range(m)], dtype=bool)
    va_sc = np.array([fn[i] in val_sc_files for i in range(m)], dtype=bool)

    emb_xc = np.asarray(z_xc["embeddings"], dtype=np.float32)
    lab_xc = np.asarray(z_xc["labels"], dtype=np.float32)
    emb_sc = np.asarray(z_sc["embeddings"], dtype=np.float32)
    lab_sc = np.asarray(z_sc["labels"], dtype=np.float32)

    x_train = np.concatenate([emb_xc[tr_xc], emb_sc[tr_sc]], axis=0)
    y_train = np.concatenate([lab_xc[tr_xc], lab_sc[tr_sc]], axis=0)
    x_val = np.concatenate([emb_xc[va_xc], emb_sc[va_sc]], axis=0)
    y_val = np.concatenate([lab_xc[va_xc], lab_sc[va_sc]], axis=0)

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
    )


class TrainerEmbeddingFast:
    def __init__(self, fold: int, batch_size: int, xc_path: Path, sc_path: Path) -> None:
        self.fold = fold
        self.device = torch.device("cuda")
        self.batch_size = batch_size

        x_tr, y_tr, x_va, y_va = load_embedding_splits(fold, xc_path, sc_path)
        self.train_ds = TensorDataset(x_tr, y_tr)
        self.val_ds = TensorDataset(x_va, y_va)

        self.model = EmbeddingHead().to(self.device)
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

        nw = min(4, config.training.num_workers)
        pw = nw > 0
        g = torch.Generator()
        g.manual_seed(config.training.seed)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=pw,
            generator=g,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=pw,
            drop_last=False,
        )

        self.scaler = GradScaler("cuda")
        self.best_auc = 0.0
        self.checkpoint_path = Path(config.paths.checkpoints_dir) / f"fold{fold}_best_fast.pth"
        self._csv = CSVLogger(Path(config.paths.logs_dir) / f"fold{fold}_log_fast.csv")

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        alpha = config.augmentation.mixup_alpha
        use_mixup = config.augmentation.use_mixup

        for xb, yb in self.train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            apply_mixup = use_mixup and alpha > 0 and random.random() < 0.5

            with autocast("cuda"):
                if apply_mixup:
                    mixed_x, la, lb, lam = mixup_batch(xb, yb, alpha, dual=True)
                    logits = self.model(mixed_x)
                    loss = mixup_criterion(self.criterion, logits, la, lb, lam)
                else:
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)

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
        self.model.eval()
        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                if yb.sum() == 0:
                    continue
                with autocast("cuda"):
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)
                losses.append(float(loss.detach()))
                all_logits.append(logits.float().cpu())
                all_labels.append(yb.float().cpu())

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
    parser = argparse.ArgumentParser(description="Fast PERCH head training on .npz embeddings")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Large batch for in-memory embeddings (tune for VRAM)",
    )
    args = parser.parse_args()

    root = Path(config.paths.data_root)
    xc_path = root / "perch_embeddings" / "xc_embeddings.npz"
    sc_path = root / "perch_embeddings" / "soundscape_embeddings.npz"

    set_seed(config.training.seed)
    create_experiment_dirs()

    trainer = TrainerEmbeddingFast(args.fold, args.batch_size, xc_path, sc_path)
    trainer.fit()
    print(f"Best validation ROC-AUC (macro, fold {args.fold}): {trainer.best_auc:.6f}")


if __name__ == "__main__":
    main()
