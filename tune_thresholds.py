"""
Per-species F1-optimal threshold tuning on the honest soundscape validation
split (148 segments, ``config.training.seed`` permutation, 20% of deduped
``train_soundscapes_labels.csv``).

Loads a checkpoint (default: ``exp029_perch_distill_fold4_clean.pth`` at
``data_root``), reproduces the inference mel pipeline (with the NaN-safe
``clamp`` + ``nan_to_num`` fixes used in ``train_perch_distill``), collects
per-segment probabilities, then sweeps thresholds per class via
``precision_recall_curve`` to maximize F1. Classes with no positives in the
val set fall back to ``0.5``.

Output: ``thresholds/per_species_thresholds.json`` — ``{species_code: float}``
in taxonomy order (the same order ``LabelEncoder`` uses).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import f1_score, precision_recall_curve
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

import config as cfg
from dataset import (
    LabelEncoder,
    _parse_soundscape_time_to_seconds,
    build_mel_transform,
    center_crop,
    mel_to_db,
    normalize_melspec,
)
from model import build_model

import torch.backends.cudnn as cudnn

cudnn.enabled = False


DEFAULT_CHECKPOINT = (
    Path(cfg.config.paths.data_root) / "exp029_perch_distill_fold4_clean.pth"
)
DEFAULT_OUTPUT_JSON = (
    Path(cfg.config.paths.data_root) / "thresholds" / "per_species_thresholds.json"
)


# ---------------------------------------------------------------------------
# Val split (matches dataset.get_dataloaders)
# ---------------------------------------------------------------------------


def build_val_segments_df() -> pd.DataFrame:
    """Return the 20% soundscape segment val split, seeded by ``training.seed``."""
    scape = pd.read_csv(cfg.config.paths.soundscape_labels)
    scape = scape.drop_duplicates(
        subset=["filename", "start", "end"], keep="first"
    ).reset_index(drop=True)
    n_seg = len(scape)
    if n_seg == 0:
        raise RuntimeError("No soundscape labels found.")
    rng = np.random.default_rng(cfg.config.training.seed)
    perm = rng.permutation(n_seg)
    n_val = int(round(0.2 * n_seg))
    n_val = max(1, min(n_val, n_seg))
    val_idx = perm[:n_val]
    return scape.iloc[val_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# NaN-safe segment dataset
# ---------------------------------------------------------------------------


class ValSegmentDataset(Dataset):
    """Labeled soundscape segment → ``(1, n_mels, time)`` mel + multihot labels.

    Same audio reading as ``dataset.SoundscapeDataset`` plus the clamp +
    ``nan_to_num`` guards that ``train_perch_distill._wav_to_mel`` uses, so a
    silent / corrupt segment can't propagate NaNs into the model.
    """

    def __init__(self, val_scape_df: pd.DataFrame, label_encoder: LabelEncoder) -> None:
        self.df = val_scape_df.reset_index(drop=True)
        self.enc = label_encoder
        self.sr = cfg.config.audio.sample_rate
        self.n_samples = cfg.config.audio.n_samples
        self.soundscapes_root = Path(cfg.config.paths.train_soundscapes)
        self.mel_xfm = build_mel_transform()
        self.db_xfm = mel_to_db()
        with torch.inference_mode():
            dummy = torch.zeros(1, self.n_samples)
            ref = normalize_melspec(self.db_xfm(self.mel_xfm(dummy).clamp(min=1e-10)))
            self._empty_melspec = torch.zeros_like(ref)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_labels(self, cell: object) -> torch.Tensor:
        parts = [p.strip() for p in str(cell).split(";") if p.strip()]
        return torch.from_numpy(self.enc.encode_labels(parts))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        labels = self._encode_labels(row["primary_label"])
        fp = self.soundscapes_root / str(row["filename"])
        empty = {"melspec": self._empty_melspec.clone(), "labels": labels}
        try:
            t0 = _parse_soundscape_time_to_seconds(row["start"])
            t1 = _parse_soundscape_time_to_seconds(row["end"])
            f0 = int(round(t0 * self.sr))
            f1 = int(round(t1 * self.sr))
            n_frames = max(1, f1 - f0)
            wav, file_sr = torchaudio.load(
                str(fp), frame_offset=f0, num_frames=n_frames
            )
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if file_sr != self.sr:
                wav = torchaudio.functional.resample(wav, file_sr, self.sr)
            wav = center_crop(wav, self.n_samples)
            mel = self.mel_xfm(wav)
            mel = mel.clamp(min=1e-10)
            mel = self.db_xfm(mel)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
            mel = normalize_melspec(mel)
            return {"melspec": mel, "labels": labels}
        except Exception:
            return empty


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    """Build ``BirdCLEFModel`` (no ImageNet download) and load ``model_state``."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model = build_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` arrays with shape ``(N, num_classes)``."""
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            specs = batch["melspec"].to(device, non_blocking=True)
            labels = batch["labels"]
            if use_amp:
                with autocast("cuda"):
                    logits = model(specs)
            else:
                logits = model(specs)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def find_optimal_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    default: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class F1-maximizing threshold via ``precision_recall_curve``.

    Returns
    -------
    thresholds : ``(C,)`` chosen threshold per class (``default`` if no positives).
    best_f1 : ``(C,)`` F1 at the chosen threshold (``NaN`` if no positives).
    has_pos : ``(C,)`` bool mask of classes with at least one positive label.
    """
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    n_classes = labels.shape[1]
    thresholds = np.full(n_classes, default, dtype=np.float64)
    best_f1 = np.full(n_classes, np.nan, dtype=np.float64)
    has_pos = labels.sum(axis=0) > 0

    for j in range(n_classes):
        if not has_pos[j]:
            continue
        y = labels[:, j].astype(np.int32)
        p = probs[:, j]
        precision, recall, ts = precision_recall_curve(y, p)
        if ts.size == 0:
            continue
        prec = precision[:-1]
        rec = recall[:-1]
        denom = prec + rec
        f1 = np.where(denom > 0, 2.0 * prec * rec / np.maximum(denom, 1e-12), 0.0)
        k = int(np.argmax(f1))
        thresholds[j] = float(np.clip(ts[k], 1e-4, 1.0 - 1e-4))
        best_f1[j] = float(f1[k])

    return thresholds, best_f1, has_pos


def macro_f1_at(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> float:
    """Macro F1 across classes that have at least one positive label."""
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    has_pos = labels.sum(axis=0) > 0
    if not has_pos.any():
        return float("nan")
    y_true = labels[:, has_pos].astype(np.int32)
    thr = thresholds[has_pos].reshape(1, -1)
    y_pred = (probs[:, has_pos] >= thr).astype(np.int32)
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-species F1-optimal thresholds on the honest soundscape val split."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint .pth (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg.config.training.batch_size,
        help="Inference batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cfg.config.training.num_workers,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--default_threshold",
        type=float,
        default=0.5,
        help="Threshold used for classes with no positives in val (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    val_df = build_val_segments_df()
    print(f"Val segments: {len(val_df)}")

    enc = LabelEncoder()
    n_classes = enc.num_classes
    print(f"Classes: {n_classes}")

    dataset = ValSegmentDataset(val_df, enc)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = load_model_from_checkpoint(args.checkpoint, device)
    print("Running inference...")
    probs, labels = run_inference(model, loader, device)
    print(f"Probs: {probs.shape}, Labels: {labels.shape}")

    thresholds, best_f1, has_pos = find_optimal_thresholds(
        probs, labels, default=args.default_threshold
    )
    baseline_thr = np.full(n_classes, 0.5, dtype=np.float64)

    macro_f1_baseline = macro_f1_at(probs, labels, baseline_thr)
    macro_f1_tuned = macro_f1_at(probs, labels, thresholds)
    improvement = macro_f1_tuned - macro_f1_baseline

    out_dict = {
        enc.idx2label[j]: float(thresholds[j]) for j in range(n_classes)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out_dict, f, indent=2, sort_keys=True)

    n_with_pos = int(has_pos.sum())
    n_tuned_low = int((thresholds < 0.3).sum())
    n_tuned_high = int((thresholds > 0.7).sum())
    mean_thr = float(thresholds.mean())
    mean_thr_with_pos = float(thresholds[has_pos].mean()) if n_with_pos else float("nan")

    print()
    print("=" * 60)
    print("Per-species threshold tuning summary")
    print("=" * 60)
    print(f"Saved thresholds:                  {args.output}")
    print(f"Classes with >=1 positive in val:  {n_with_pos} / {n_classes}")
    print(f"Mean threshold (all classes):      {mean_thr:.4f}")
    print(f"Mean threshold (classes w/ pos):   {mean_thr_with_pos:.4f}")
    print(f"Classes with threshold < 0.3:      {n_tuned_low}")
    print(f"Classes with threshold > 0.7:      {n_tuned_high}")
    print()
    print(f"Macro F1 @ 0.5 baseline:           {macro_f1_baseline:.6f}")
    print(f"Macro F1 @ tuned thresholds:       {macro_f1_tuned:.6f}")
    print(f"Improvement:                       {improvement:+.6f}")


if __name__ == "__main__":
    main()
