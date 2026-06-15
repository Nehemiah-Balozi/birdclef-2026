"""
Waveform datasets and dataloaders for PERCH ONNX (raw audio [batch, 160000]).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from config import config
from dataset import (
    LabelEncoder,
    _add_filepath_column,
    _apply_xc_low_rating_filter,
    _parse_secondary_labels,
    _parse_soundscape_time_to_seconds,
    _site_group_key,
    _stratified_split_indices,
    apply_spec_augment,
    build_mel_transform,
    center_crop,
    load_audio,
    mel_to_db,
    normalize_melspec,
    random_crop,
)


def mixup_waveform_batch(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    *,
    dual: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
):
    """
    Mixup on waveforms ``(B, n_samples)`` and multilabel targets (CPU tensors).
    Same semantics as :func:`dataset.mixup_batch` for spectrograms.
    """
    if alpha <= 0:
        if dual:
            return waveforms, labels, labels, 1.0
        return waveforms, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    b = waveforms.size(0)
    perm = torch.randperm(b)
    mixed_wav = lam * waveforms + (1.0 - lam) * waveforms[perm]
    if dual:
        return mixed_wav, labels, labels[perm], lam
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_wav, mixed_labels


class BirdCLEFWaveformDataset(Dataset):
    """Reference recordings: returns mono waveform ``(n_samples,)`` float32."""

    def __init__(
        self,
        df: pd.DataFrame,
        label_encoder: LabelEncoder,
        mode: str = "train",
    ) -> None:
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.mode = mode
        self.n_samples = config.audio.n_samples

    def __len__(self) -> int:
        return len(self.df)

    def _encode_row_labels(self, row: pd.Series) -> torch.Tensor:
        primary = [str(row["primary_label"]).strip()]
        secondary = _parse_secondary_labels(row["secondary_labels"])
        vec = self.label_encoder.encode_labels(primary + secondary)
        return torch.from_numpy(vec)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        fp = str(row["filepath"])
        empty = {
            "waveform": torch.zeros(self.n_samples, dtype=torch.float32),
            "labels": self._encode_row_labels(row),
            "filepath": fp,
        }
        try:
            wav = load_audio(fp)
            if self.mode == "train":
                wav = random_crop(wav, self.n_samples)
            else:
                wav = center_crop(wav, self.n_samples)
            w = wav.squeeze(0).to(torch.float32)
            return {
                "waveform": w,
                "labels": self._encode_row_labels(row),
                "filepath": fp,
            }
        except Exception:
            return empty


class SoundscapeWaveformDataset(Dataset):
    """Soundscape segments: mono waveform ``(n_samples,)`` for PERCH."""

    def __init__(
        self,
        labels_df: pd.DataFrame,
        label_encoder: LabelEncoder,
        mode: str = "val",
    ) -> None:
        self.df = labels_df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.mode = mode
        self.soundscapes_root = Path(config.paths.train_soundscapes)
        self.n_samples = config.audio.n_samples
        self.sr = config.audio.sample_rate

    def __len__(self) -> int:
        return len(self.df)

    def _encode_scape_labels(self, cell: object) -> torch.Tensor:
        parts = [p.strip() for p in str(cell).split(";") if p.strip()]
        vec = self.label_encoder.encode_labels(parts)
        return torch.from_numpy(vec)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        fp = str(self.soundscapes_root / filename)
        empty = {
            "waveform": torch.zeros(self.n_samples, dtype=torch.float32),
            "labels": self._encode_scape_labels(row["primary_label"]),
            "filepath": fp,
        }
        try:
            t0 = _parse_soundscape_time_to_seconds(row["start"])
            t1 = _parse_soundscape_time_to_seconds(row["end"])
            f0 = int(round(t0 * self.sr))
            f1 = int(round(t1 * self.sr))
            n_frames = max(1, f1 - f0)
            wav, sr = torchaudio.load(fp, frame_offset=f0, num_frames=n_frames)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            wav = center_crop(wav, self.n_samples)
            w = wav.squeeze(0).to(torch.float32)
            return {
                "waveform": w,
                "labels": self._encode_scape_labels(row["primary_label"]),
                "filepath": fp,
            }
        except Exception:
            return empty


class BirdCLEFMelWaveformDataset(Dataset):
    """
    XC reference: same audio pipeline as ``BirdCLEFDataset`` (noise mix, gain, crop),
    returns both ``waveform`` ``(n_samples,)`` and ``melspec`` for distillation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_encoder: LabelEncoder,
        mode: str = "train",
    ) -> None:
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.mode = mode
        self.n_samples = config.audio.n_samples
        self._noise_files: tuple[str, ...] = ()
        if mode == "train" and config.augmentation.use_noise_mix:
            root = Path(config.paths.train_soundscapes)
            self._noise_files = tuple(sorted(str(p) for p in root.glob("*.ogg")))
        self.mel_xfm = build_mel_transform()
        self.db_xfm = mel_to_db()
        with torch.inference_mode():
            dummy = torch.zeros(1, self.n_samples)
            ref = normalize_melspec(self.db_xfm(self.mel_xfm(dummy)))
            self._empty_melspec = torch.zeros_like(ref)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_row_labels(self, row: pd.Series) -> torch.Tensor:
        primary = [str(row["primary_label"]).strip()]
        secondary = _parse_secondary_labels(row["secondary_labels"])
        vec = self.label_encoder.encode_labels(primary + secondary)
        return torch.from_numpy(vec)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        fp = str(row["filepath"])
        empty = {
            "waveform": torch.zeros(self.n_samples, dtype=torch.float32),
            "melspec": self._empty_melspec.clone(),
            "labels": self._encode_row_labels(row),
            "filepath": fp,
        }
        try:
            wav = load_audio(fp)
            if self.mode == "train":
                wav = random_crop(wav, self.n_samples)
                if (
                    config.augmentation.use_noise_mix
                    and self._noise_files
                    and random.random() < config.augmentation.noise_mix_prob
                ):
                    try:
                        noise_fp = random.choice(self._noise_files)
                        noise_wav = load_audio(noise_fp)
                        noise_wav = random_crop(noise_wav, self.n_samples)
                        a = config.augmentation
                        alpha = random.uniform(a.noise_mix_alpha_min, a.noise_mix_alpha_max)
                        wav = wav * alpha + noise_wav * (1.0 - alpha)
                    except Exception:
                        pass
                if (
                    config.augmentation.use_gain_aug
                    and random.random() < config.augmentation.gain_aug_prob
                ):
                    a = config.augmentation
                    gain = random.uniform(a.gain_min, a.gain_max)
                    wav = wav * gain
            else:
                wav = center_crop(wav, self.n_samples)
            w = wav.squeeze(0).to(torch.float32)
            mel = self.mel_xfm(wav)
            mel = mel.clamp(min=1e-10)
            mel = self.db_xfm(mel)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
            mel = normalize_melspec(mel)
            if self.mode == "train" and config.augmentation.use_specaugment:
                mel = apply_spec_augment(mel)
            return {
                "waveform": w,
                "melspec": mel,
                "labels": self._encode_row_labels(row),
                "filepath": fp,
            }
        except Exception:
            return empty


class SoundscapeMelWaveformDataset(Dataset):
    """Labeled soundscape segment: ``waveform`` + ``melspec`` (same processing as ``SoundscapeDataset``)."""

    def __init__(
        self,
        labels_df: pd.DataFrame,
        label_encoder: LabelEncoder,
        mode: str = "val",
    ) -> None:
        self.df = labels_df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.mode = mode
        self.soundscapes_root = Path(config.paths.train_soundscapes)
        self.n_samples = config.audio.n_samples
        self.sr = config.audio.sample_rate
        self.mel_xfm = build_mel_transform()
        self.db_xfm = mel_to_db()
        with torch.inference_mode():
            dummy = torch.zeros(1, self.n_samples)
            ref = normalize_melspec(self.db_xfm(self.mel_xfm(dummy)))
            self._empty_melspec = torch.zeros_like(ref)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_scape_labels(self, cell: object) -> torch.Tensor:
        parts = [p.strip() for p in str(cell).split(";") if p.strip()]
        vec = self.label_encoder.encode_labels(parts)
        return torch.from_numpy(vec)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        fp = str(self.soundscapes_root / filename)
        empty = {
            "waveform": torch.zeros(self.n_samples, dtype=torch.float32),
            "melspec": self._empty_melspec.clone(),
            "labels": self._encode_scape_labels(row["primary_label"]),
            "filepath": fp,
        }
        try:
            t0 = _parse_soundscape_time_to_seconds(row["start"])
            t1 = _parse_soundscape_time_to_seconds(row["end"])
            f0 = int(round(t0 * self.sr))
            f1 = int(round(t1 * self.sr))
            n_frames = max(1, f1 - f0)
            wav, sr = torchaudio.load(fp, frame_offset=f0, num_frames=n_frames)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            wav = center_crop(wav, self.n_samples)
            w = wav.squeeze(0).to(torch.float32)
            mel = self.mel_xfm(wav)
            mel = mel.clamp(min=1e-10)
            mel = self.db_xfm(mel)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
            mel = normalize_melspec(mel)
            return {
                "waveform": w,
                "melspec": mel,
                "labels": self._encode_scape_labels(row["primary_label"]),
                "filepath": fp,
            }
        except Exception:
            return empty


def get_dataloaders_distill(fold: int) -> tuple[DataLoader, DataLoader]:
    """
    Same splits as :func:`dataset.get_dataloaders`: XC train fold + 80% soundscape
    segments for train; 20% soundscape segments only for val. Each sample includes
    ``waveform``, ``melspec``, and ``labels``.
    """
    if fold < 0 or fold >= config.training.n_folds:
        raise ValueError(f"fold must be in [0, {config.training.n_folds}), got {fold}")

    df = pd.read_csv(config.paths.train_csv)
    df = _add_filepath_column(df)
    df = _apply_xc_low_rating_filter(df)
    df["site_group"] = [_site_group_key(a, b) for a, b in zip(df["latitude"], df["longitude"])]

    train_idx, _ = _stratified_split_indices(
        df,
        n_folds=config.training.n_folds,
        val_fold=fold,
        seed=config.training.seed,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)

    enc = LabelEncoder()

    scape = pd.read_csv(config.paths.soundscape_labels)
    scape = scape.drop_duplicates(subset=["filename", "start", "end"], keep="first").reset_index(
        drop=True
    )
    n_seg = len(scape)
    rng = np.random.default_rng(config.training.seed)
    perm = rng.permutation(n_seg) if n_seg else np.array([], dtype=np.int64)
    n_val = 0
    if n_seg > 0:
        n_val = int(round(0.2 * n_seg))
        if n_val < 1:
            n_val = 1
        if n_val >= n_seg and n_seg > 1:
            n_val = n_seg - 1
        elif n_val >= n_seg:
            n_val = n_seg
    val_idx = perm[:n_val]
    train_idx_seg = perm[n_val:]
    train_scape_df = scape.iloc[train_idx_seg].reset_index(drop=True)
    val_scape_df = scape.iloc[val_idx].reset_index(drop=True)
    print(
        f"[distill] Soundscape segment split — train: {len(train_scape_df)}, val: {len(val_scape_df)}"
    )

    train_ref = BirdCLEFMelWaveformDataset(train_df, enc, mode="train")
    train_scape = SoundscapeMelWaveformDataset(train_scape_df, enc, mode="train")
    val_scape = SoundscapeMelWaveformDataset(val_scape_df, enc, mode="val")
    train_ds: Dataset = ConcatDataset([train_ref, train_scape])
    val_ds: Dataset = val_scape

    nw = config.training.num_workers
    pw = nw > 0

    g = torch.Generator()
    g.manual_seed(config.training.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        drop_last=False,
    )
    return train_loader, val_loader


    """Same splits as :func:`dataset.get_dataloaders`, but waveform batches."""
    if fold < 0 or fold >= config.training.n_folds:
        raise ValueError(f"fold must be in [0, {config.training.n_folds}), got {fold}")

    df = pd.read_csv(config.paths.train_csv)
    df = _add_filepath_column(df)
    df = _apply_xc_low_rating_filter(df)
    df["site_group"] = [_site_group_key(a, b) for a, b in zip(df["latitude"], df["longitude"])]

    train_idx, val_idx = _stratified_split_indices(
        df,
        n_folds=config.training.n_folds,
        val_fold=fold,
        seed=config.training.seed,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    enc = LabelEncoder()

    scape = pd.read_csv(config.paths.soundscape_labels)
    scape = scape.drop_duplicates(subset=["filename", "start", "end"], keep="first").reset_index(
        drop=True
    )
    scape_files = sorted(scape["filename"].astype(str).unique().tolist())
    random.seed(config.training.seed)
    random.shuffle(scape_files)
    n_train_files = int(len(scape_files) * 0.8)
    train_scape_files = set(scape_files[:n_train_files])
    val_scape_files = set(scape_files[n_train_files:])
    train_scape_df = scape[scape["filename"].astype(str).isin(train_scape_files)].reset_index(
        drop=True
    )
    val_scape_df = scape[scape["filename"].astype(str).isin(val_scape_files)].reset_index(drop=True)
    print(
        f"Soundscape split segments - train: {len(train_scape_df)}, val: {len(val_scape_df)}"
    )

    train_ref = BirdCLEFWaveformDataset(train_df, enc, mode="train")
    val_ref = BirdCLEFWaveformDataset(val_df, enc, mode="val")
    train_scape = SoundscapeWaveformDataset(train_scape_df, enc, mode="train")
    val_scape = SoundscapeWaveformDataset(val_scape_df, enc, mode="val")
    train_ds: Dataset = ConcatDataset([train_ref, train_scape])
    val_ds: Dataset = ConcatDataset([val_ref, val_scape])

    nw = config.training.num_workers
    pw = nw > 0

    g = torch.Generator()
    g.manual_seed(config.training.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=False,
        persistent_workers=pw,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
        persistent_workers=pw,
        drop_last=False,
    )
    return train_loader, val_loader
