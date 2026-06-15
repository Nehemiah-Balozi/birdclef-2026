"""
Distill frozen PERCH ONNX + frozen PERCH head into EfficientNet-B5 (BirdCLEFModel).

Loss = 0.5 * MSE(sigmoid(student), clamp(sigmoid(teacher))) + 0.5 * focal(student, hard_labels).

Dataloaders: :func:`dataset_perch.get_dataloaders_distill` (same XC + segment split as
``dataset.get_dataloaders`` — configure augmentation in ``config`` for exp020-style noise).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import GradScaler, autocast

import config as cfg
from dataset import apply_spec_augment, build_mel_transform, mel_to_db, mixup_batch, normalize_melspec
from dataset_perch import get_dataloaders_distill
from model import FocalLoss, build_model, mixup_criterion
from model_perch import PerchTeacher

import torch.backends.cudnn as cudnn

cudnn.enabled = False

MSE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distill_mse_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """MSE between student and teacher probabilities; teacher targets clamped for stability."""
    with torch.no_grad():
        teacher_probs = torch.sigmoid(teacher_logits.float()).clamp(0.01, 0.99)
    student_probs = torch.sigmoid(student_logits.float())
    return F.mse_loss(student_probs, teacher_probs)


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


class TrainerPerchDistill:
    def __init__(
        self,
        fold: int,
        teacher_ckpt: Path,
    ) -> None:
        self.fold = fold
        self.device = torch.device("cuda")
        self.teacher = PerchTeacher(head_checkpoint=teacher_ckpt, head_device=self.device)
        self.student = build_model(pretrained=True).to(self.device)
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")

        self.mel_xfm = build_mel_transform().to(self.device)
        self.db_xfm = mel_to_db().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.config.training.learning_rate,
            weight_decay=cfg.config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.config.training.num_epochs,
            eta_min=1e-6,
        )
        self.train_loader, self.val_loader = get_dataloaders_distill(fold)
        self.scaler = GradScaler("cuda")
        self.best_auc = 0.0
        self.checkpoint_path = Path(cfg.config.paths.checkpoints_dir) / f"fold{fold}_best.pth"
        self._csv = CSVLogger(Path(cfg.config.paths.logs_dir) / f"fold{fold}_log.csv")
        self._debug_first_train_batch_done = False

    def _wav_to_mel(self, wav: torch.Tensor, *, training: bool) -> torch.Tensor:
        """``wav`` (B, n_samples) on device → normalized log-mel ``(B,1,f,t)``."""
        x = wav.unsqueeze(1).to(self.device, non_blocking=True)
        mel = self.mel_xfm(x)
        if not hasattr(self, "_mel_shape_printed"):
            print(f"[debug] x shape before mel: {tuple(x.shape)}")
            print(f"[debug] mel_xfm type: {type(self.mel_xfm)}")
            print(f"[debug] mel shape after transform: {tuple(mel.shape)}")
            print(f"[debug] mel has nan: {bool(mel.isnan().any().item())}")
            self._mel_shape_printed = True
        mel = mel.clamp(min=1e-10)
        mel = self.db_xfm(mel)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
        mel = normalize_melspec(mel)
        if training and cfg.config.augmentation.use_specaugment:
            mel = apply_spec_augment(mel)
        return mel

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.student.train()
        total_loss = 0.0
        n_batches = 0
        alpha = cfg.config.augmentation.mixup_alpha
        use_mixup = cfg.config.augmentation.use_mixup

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            if "melspec" in batch:
                melspec = batch["melspec"]
                if torch.isnan(melspec).any():
                    batch["melspec"] = torch.nan_to_num(melspec, nan=0.0)
            waveforms = batch["waveform"].to(dtype=torch.float32)  # CPU (B, n)
            labels = batch["labels"].to(self.device, non_blocking=True)

            apply_mixup = use_mixup and alpha > 0 and random.random() < 0.5

            if apply_mixup:
                mixed_wav, la, lb, lam = mixup_batch(
                    waveforms,
                    labels.cpu(),
                    alpha,
                    dual=True,
                )
                la = la.to(self.device, non_blocking=True)
                lb = lb.to(self.device, non_blocking=True)
                with torch.no_grad():
                    teacher_logits = self.teacher(mixed_wav)
                with autocast("cuda"):
                    specs = self._wav_to_mel(mixed_wav.to(self.device), training=True)
                    student_logits = self.student(specs)
                    focal = mixup_criterion(self.criterion, student_logits, la, lb, lam)
                    mse = distill_mse_loss(student_logits, teacher_logits)
                    loss = MSE_WEIGHT * mse + FOCAL_WEIGHT * focal
            else:
                with torch.no_grad():
                    teacher_logits = self.teacher(waveforms)
                with autocast("cuda"):
                    specs = self._wav_to_mel(waveforms.to(self.device), training=True)
                    student_logits = self.student(specs)
                    focal = self.criterion(student_logits, labels)
                    mse = distill_mse_loss(student_logits, teacher_logits)
                    loss = MSE_WEIGHT * mse + FOCAL_WEIGHT * focal

            if epoch == 0 and not self._debug_first_train_batch_done:
                wav_dbg = mixed_wav if apply_mixup else waveforms
                w = wav_dbg.detach().float()
                s = specs.detach().float()
                tl = teacher_logits.detach().float()
                stu = student_logits.detach().float()
                print("[distill-debug] first training batch (epoch 0)")
                print(
                    f"  waveform: min={w.min().item():.6f} max={w.max().item():.6f} "
                    f"isnan_any={bool(torch.isnan(w).any().item())}"
                )
                print(
                    f"  melspec: min={s.min().item():.6f} max={s.max().item():.6f} "
                    f"isnan_any={bool(torch.isnan(s).any().item())}"
                )
                print(
                    f"  teacher_logits: min={tl.min().item():.6f} max={tl.max().item():.6f} "
                    f"isnan_any={bool(torch.isnan(tl).any().item())}"
                )
                print(
                    f"  student_logits: min={stu.min().item():.6f} max={stu.max().item():.6f} "
                    f"isnan_any={bool(torch.isnan(stu).any().item())}"
                )
                print(
                    f"  mse_loss={float(mse.detach().float())} "
                    f"focal_loss={float(focal.detach().float())}"
                )
                self._debug_first_train_batch_done = True

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        lr = float(self.optimizer.param_groups[0]["lr"])
        return {"train_loss": avg_loss, "lr": lr}

    def validate(self, epoch: int) -> dict[str, float]:
        self.student.eval()
        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                if "melspec" in batch:
                    melspec = batch["melspec"]
                    if torch.isnan(melspec).any():
                        batch["melspec"] = torch.nan_to_num(melspec, nan=0.0)
                waveforms = batch["waveform"].to(dtype=torch.float32)
                labels = batch["labels"].to(self.device, non_blocking=True)
                if labels.sum() == 0:
                    continue
                with torch.no_grad():
                    teacher_logits = self.teacher(waveforms)
                with autocast("cuda"):
                    specs = self._wav_to_mel(waveforms.to(self.device), training=False)
                    student_logits = self.student(specs)
                    focal = self.criterion(student_logits, labels)
                    mse = distill_mse_loss(student_logits, teacher_logits)
                    loss = MSE_WEIGHT * mse + FOCAL_WEIGHT * focal
                losses.append(float(loss.detach()))
                all_logits.append(student_logits.float().cpu())
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
                    "model_state": self.student.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "auc": float(auc),
                    "map": float(map_score),
                    "config": cfg.config,
                },
                self.checkpoint_path,
            )

        return {"val_loss": val_loss, "val_auc": auc, "val_map": map_score}

    def fit(self) -> None:
        n_ep = cfg.config.training.num_epochs
        patience = cfg.config.training.early_stopping_patience
        epochs_no_improve = 0
        for epoch in range(n_ep):
            tr = self.train_one_epoch(epoch)
            prev_best = self.best_auc
            va = self.validate(epoch)
            self.scheduler.step()

            if self.best_auc > prev_best:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

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

            if patience > 0 and epochs_no_improve >= patience:
                print(
                    f"Early stopping: no val AUC improvement for {patience} epochs "
                    f"(best_auc={self.best_auc:.4f})."
                )
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="PERCH → EfficientNet-B5 distillation")
    parser.add_argument("--fold", type=int, default=0, help="XC / segment split fold index")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp025_perch_distill_fold0",
        help="Experiment name for checkpoints/logs (override config paths)",
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=Path,
        default=None,
        help="PERCH head checkpoint (.pth with model_state head.*). "
        "Default: <data_root>/perch_fold1_clean2.pth",
    )
    args = parser.parse_args()

    if args.exp_name:
        cfg.apply_experiment_name_override(args.exp_name)
        print(
            f"Experiment: {cfg.config.paths.experiment_name!r} -> "
            f"{cfg.config.paths.experiment_dir!r}"
        )

    root = Path(cfg.config.paths.data_root)
    teacher_path = args.teacher_ckpt or (root / "perch_fold1_clean2.pth")
    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_path}. Pass --teacher_ckpt /path/to.pth"
        )

    set_seed(cfg.config.training.seed)
    cfg.create_experiment_dirs()

    trainer = TrainerPerchDistill(fold=args.fold, teacher_ckpt=teacher_path)
    trainer.fit()
    print(f"Best validation ROC-AUC (macro, fold {args.fold}): {trainer.best_auc:.6f}")


if __name__ == "__main__":
    main()
