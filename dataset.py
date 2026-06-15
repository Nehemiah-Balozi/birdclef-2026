"""
BirdCLEF+ 2026 — datasets, audio/mel utilities, and dataloaders.

Pure data pipeline (no optimization / training loop logic).
"""

from __future__ import annotations

import ast
import random
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from config import config


# ---------------------------------------------------------------------------
# 1. Label encoder
# ---------------------------------------------------------------------------


class LabelEncoder:
    """
    Taxonomy-based multilabel encoding for the 234 competition classes.

    Builds a stable sorted label list from ``taxonomy.csv`` and maps each
    ``primary_label`` string to a class index.
    """

    def __init__(self, taxonomy_csv: str | Path | None = None) -> None:
        path = Path(taxonomy_csv or config.paths.taxonomy_csv)
        tax = pd.read_csv(path)
        self.idx2label: list[str] = sorted(tax["primary_label"].astype(str).unique().tolist())
        self.label2idx: dict[str, int] = {lab: i for i, lab in enumerate(self.idx2label)}
        self.num_classes: int = len(self.idx2label)

    def encode_labels(self, labels: list[str]) -> np.ndarray:
        """
        Encode a list of present class IDs into a dense multihot vector.

        Returns
        -------
        np.ndarray
            Shape ``(num_classes,)``, ``float32``, ``1.0`` for each present
            label (known in taxonomy), ``0.0`` elsewhere.
        """
        out = np.zeros(self.num_classes, dtype=np.float32)
        for lab in labels:
            key = str(lab).strip()
            idx = self.label2idx.get(key)
            if idx is not None:
                out[idx] = 1.0
        return out


# ---------------------------------------------------------------------------
# 2. Audio utils
# ---------------------------------------------------------------------------


def load_audio(filepath: str | Path) -> torch.Tensor:
    """
    Load audio as a mono ``torch.Tensor`` at ``config.audio.sample_rate``.

    Stereo inputs are averaged to mono. Resamples when the file sample rate
    differs from the target.
    """
    wav, sr = torchaudio.load(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    target_sr = config.audio.sample_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def random_crop(waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Random temporal crop of a ``(1, n)`` waveform to exactly ``n_samples``.

    If shorter than ``n_samples``, right-zero-pads. If longer, chooses a
    random start index (training).
    """
    n = waveform.shape[-1]
    if n == n_samples:
        return waveform
    if n > n_samples:
        start = random.randint(0, n - n_samples)
        return waveform[:, start : start + n_samples]
    pad = n_samples - n
    return F.pad(waveform, (0, pad))


def center_crop(waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Center crop or symmetric zero-pad to exactly ``n_samples`` (validation).
    """
    n = waveform.shape[-1]
    if n == n_samples:
        return waveform
    if n > n_samples:
        start = max(0, (n - n_samples) // 2)
        return waveform[:, start : start + n_samples]
    pad = n_samples - n
    pad_left = pad // 2
    pad_right = pad - pad_left
    return F.pad(waveform, (pad_left, pad_right))


# ---------------------------------------------------------------------------
# 3. Mel spectrogram
# ---------------------------------------------------------------------------


def build_mel_transform() -> T.MelSpectrogram:
    """Construct ``MelSpectrogram`` from ``config.mel``."""
    m = config.mel
    return T.MelSpectrogram(
        sample_rate=config.audio.sample_rate,
        n_fft=m.n_fft,
        hop_length=m.hop_length,
        n_mels=m.n_mels,
        f_min=m.fmin,
        f_max=m.fmax,
        mel_scale=m.mel_scale,
    )


def mel_to_db() -> T.AmplitudeToDB:
    """Amplitude-to-decibel transform (torchaudio)."""
    return T.AmplitudeToDB()


def normalize_melspec(melspec: torch.Tensor) -> torch.Tensor:
    """
    Per-sample global normalization: ``(x - mean) / (std + 1e-6)`` over all
    elements of the spectrogram tensor. Returns zeros if ``std`` is near zero
    to avoid NaNs.
    """
    mean = melspec.mean()
    std = melspec.std()
    if std < 1e-8:
        return torch.zeros_like(melspec)
    return (melspec - mean) / (std + 1e-6)


def apply_pcen(
    mel: torch.Tensor,
    *,
    sample_rate: int | None = None,
    hop_length: int | None = None,
    time_constant: float | None = None,
    eps: float | None = None,
    gain: float | None = None,
    bias: float | None = None,
    power: float | None = None,
) -> torch.Tensor:
    """Per-Channel Energy Normalization on a magnitude/power mel spectrogram.

    Implements::

        M_t   = (1 - s) * M_{t-1} + s * E_t                   (causal IIR low-pass)
        PCEN_t = (E_t / (eps + M_t)^alpha + delta)^r - delta^r

    where ``s = 1 - exp(-(hop_length / sample_rate) / time_constant)`` and the
    smoother runs per mel bin along the time axis (via
    ``torchaudio.functional.lfilter`` so a 256-bin spectrogram is filtered in
    one C++ call instead of a Python loop).

    PCEN suppresses stationary background energy (cicadas, wind, recorder hum)
    while preserving transients like bird calls — generally more robust than
    log-mel for noisy soundscapes (Wang, Lostanlen, Cella & Bello, 2017
    — *Trainable frontend for robust and far-field keyword spotting*).

    Inputs are clamped to ``>= 0`` for numerical safety; output has the same
    shape as ``mel``. Each ``None`` argument falls back to ``config.mel``.
    """
    import math

    from torchaudio.functional import lfilter

    m = config.mel
    sr = sample_rate if sample_rate is not None else config.audio.sample_rate
    hop = hop_length if hop_length is not None else m.hop_length
    tau = time_constant if time_constant is not None else m.pcen_time_constant
    e_eps = eps if eps is not None else m.pcen_eps
    alpha = gain if gain is not None else m.pcen_gain
    delta = bias if bias is not None else m.pcen_bias
    r = power if power is not None else m.pcen_power

    s = 1.0 - math.exp(-(hop / sr) / tau)

    E = mel.clamp(min=0.0)
    if E.shape[-1] == 0:
        return E

    a_coeffs = torch.tensor([1.0, -(1.0 - s)], dtype=E.dtype, device=E.device)
    b_coeffs = torch.tensor([s, 0.0], dtype=E.dtype, device=E.device)
    M = lfilter(E, a_coeffs, b_coeffs, clamp=False)

    smoother = (e_eps + M).pow(alpha)
    return (E / smoother + delta).pow(r) - (delta ** r)


def _postprocess_mel(
    raw_mel: torch.Tensor, db_transform: T.AmplitudeToDB
) -> torch.Tensor:
    """Magnitude/power mel → model-ready spectrogram (PCEN or log-mel + norm)."""
    if config.mel.use_pcen:
        return apply_pcen(raw_mel)
    return normalize_melspec(db_transform(raw_mel))


# ---------------------------------------------------------------------------
# 4. Augmentations
# ---------------------------------------------------------------------------


def apply_spec_augment(melspec: torch.Tensor) -> torch.Tensor:
    """
    Apply SpecAugment-style frequency and time masking.

    Uses ``config.augmentation.freq_mask_param`` and ``time_mask_param``.
    Expects input shape ``(1, n_mels, time)``.
    """
    if not config.augmentation.use_specaugment:
        return melspec
    a = config.augmentation
    x = melspec.clone()
    if a.freq_mask_param > 0:
        x = T.FrequencyMasking(a.freq_mask_param)(x)
    if a.time_mask_param > 0:
        x = T.TimeMasking(a.time_mask_param)(x)
    return x


def mixup_batch(
    specs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    *,
    dual: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
):
    """
    Batch mixup on spectrograms and soft multilabel targets.

    Draws a single ``λ ~ Beta(α, α)`` and mixes each sample with a shuffled
    partner. If ``α <= 0``, returns inputs unchanged.

    Parameters
    ----------
    dual :
        If ``True``, return ``(mixed_specs, labels_a, labels_b, lam)`` for
        dual-target losses (e.g. :func:`model.mixup_criterion`). Otherwise
        return ``(mixed_specs, mixed_soft_labels)``.
    """
    if alpha <= 0:
        if dual:
            return specs, labels, labels, 1.0
        return specs, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    b = specs.size(0)
    perm = torch.randperm(b, device=specs.device)
    mixed_specs = lam * specs + (1.0 - lam) * specs[perm]
    if dual:
        return mixed_specs, labels, labels[perm], lam
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_specs, mixed_labels


# ---------------------------------------------------------------------------
# Helpers (parsing / filtering / split)
# ---------------------------------------------------------------------------


def _parse_secondary_labels(cell: object) -> list[str]:
    """Parse ``secondary_labels`` cell into a list of string class IDs."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s or s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    return []


def _add_filepath_column(df: pd.DataFrame) -> pd.DataFrame:
    """``train_audio / primary_label / basename(filename)``."""
    out = df.copy()
    root = Path(config.paths.train_audio)
    out["filepath"] = out.apply(
        lambda r: str(root / str(r["primary_label"]) / Path(str(r["filename"])).name),
        axis=1,
    )
    return out


def _apply_xc_low_rating_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all iNat rows. For XC, drop rows with rating < 2 only if the species
    would still have at least 10 rows after dropping all such XC rows.
    """
    d = df.copy()
    rating = pd.to_numeric(d["rating"], errors="coerce")
    coll = d["collection"].astype(str).str.strip().str.lower()
    is_xc = coll.eq("xc")
    low_xc = is_xc & rating.lt(2)

    drop = np.zeros(len(d), dtype=bool)
    for _, g in d.groupby("primary_label"):
        pos = g.index.to_numpy()
        low_mask = low_xc.loc[g.index].to_numpy()
        n_low = int(low_mask.sum())
        n_tot = len(pos)
        n_after = n_tot - n_low
        if n_low > 0 and n_after >= 10:
            drop[pos[low_mask]] = True

    return d.loc[~drop].reset_index(drop=True)


def _site_group_key(lat: object, lon: object) -> str:
    """Coarse geolocation bucket for site-stratified splitting."""
    try:
        if lat is None or lon is None or (isinstance(lat, float) and np.isnan(lat)):
            return "unknown"
        if isinstance(lon, float) and np.isnan(lon):
            return "unknown"
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return "unknown"
    return f"{round(la, 2):.2f}_{round(lo, 2):.2f}"


def _stratified_split_indices(
    df: pd.DataFrame,
    n_folds: int,
    val_fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Site-stratified train/val indices when possible (``StratifiedGroupKFold``),
    else ``StratifiedKFold`` on ``primary_label``.
    """
    y = df["primary_label"].astype(str).to_numpy()
    groups = df["site_group"].astype(str).to_numpy()
    n = len(df)
    X_placeholder = np.zeros(n)

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(sgkf.split(X_placeholder, y, groups))
        train_idx, val_idx = splits[val_fold]
        return train_idx, val_idx
    except Exception:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X_placeholder, y))
        train_idx, val_idx = splits[val_fold]
        return train_idx, val_idx


def _parse_soundscape_time_to_seconds(value: object) -> float:
    """
    Convert a soundscape ``start`` / ``end`` cell to seconds (float).

    Supports:

    - Numeric seconds: ``int``, ``float``, ``numpy`` scalars, or string
      ``\"0\"``, ``\"5.5\"`` (no colons).
    - Clock strings: ``HH:MM:SS`` or ``MM:SS`` (string or Excel-style).
    - ``datetime.time`` (e.g. from pandas after parsing).
    - ``pandas.Timedelta`` / ``datetime.timedelta`` (offset from zero).
    """
    if value is None:
        raise ValueError("missing time value")
    if not isinstance(value, (str, time, timedelta, pd.Timedelta)):
        try:
            if pd.isna(value):
                raise ValueError("missing or NaN time value")
        except TypeError:
            pass

    if isinstance(value, timedelta):
        return float(value.total_seconds())
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, time):
        return (
            value.hour * 3600.0
            + value.minute * 60.0
            + value.second
            + value.microsecond * 1e-6
        )
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        raise ValueError("empty time string")

    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600.0 + int(m) * 60.0 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60.0 + float(sec)
        raise ValueError(f"Unrecognized time format (too many ':'): {value!r}")

    return float(s)


# ---------------------------------------------------------------------------
# 5. BirdCLEFDataset
# ---------------------------------------------------------------------------


class BirdCLEFDataset(Dataset):
    """
    Reference recordings from ``train_audio`` (rows from ``train.csv``).

    Parameters
    ----------
    df :
        Must include ``filepath``, ``primary_label``, ``secondary_labels``,
        ``rating``, ``collection`` (``filepath`` is typically added upstream).
    mode :
        ``'train'`` → random crop, optional soundscape noise mix, optional gain scale,
        optional SpecAugment;
        ``'val'`` → center crop, no augmentation.
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
            ref = _postprocess_mel(self.mel_xfm(dummy), self.db_xfm)
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
            mel = _postprocess_mel(self.mel_xfm(wav), self.db_xfm)
            if self.mode == "train" and config.augmentation.use_specaugment:
                mel = apply_spec_augment(mel)
            return {
                "melspec": mel,
                "labels": self._encode_row_labels(row),
                "filepath": fp,
            }
        except Exception:
            return empty


# ---------------------------------------------------------------------------
# 6. SoundscapeDataset
# ---------------------------------------------------------------------------


class SoundscapeDataset(Dataset):
    """
    Pantanal soundscape segments with expert multilabel annotations.

    Parameters
    ----------
    labels_df :
        Columns ``filename``, ``start``, ``end``, ``primary_label`` (semicolon
        separated class IDs). ``start`` / ``end`` may be seconds as integers /
        floats, ``HH:MM:SS`` (or ``MM:SS``) strings, ``datetime.time``, or
        ``Timedelta`` — see ``_parse_soundscape_time_to_seconds``.
    mode :
        Reserved for parity with ``BirdCLEFDataset``; processing matches
        validation-style centering (fixed 5 s window).
    """

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
            ref = _postprocess_mel(self.mel_xfm(dummy), self.db_xfm)
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
            mel = _postprocess_mel(self.mel_xfm(wav), self.db_xfm)
            return {
                "melspec": mel,
                "labels": self._encode_scape_labels(row["primary_label"]),
                "filepath": fp,
            }
        except Exception:
            return empty


# ---------------------------------------------------------------------------
# 7. PseudoLabelDataset
# ---------------------------------------------------------------------------


class PseudoLabelDataset(Dataset):
    """
    Soundscape segments supervised by pseudo-label probabilities.

    Expects rows with ``filename``, ``start_sec``, ``end_sec`` and one column
    per species label (sorted taxonomy order).
    """

    def __init__(
        self,
        pseudo_df: pd.DataFrame,
        label_encoder: LabelEncoder,
        mode: str = "train",
    ) -> None:
        self.df = pseudo_df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.mode = mode
        self.soundscapes_root = Path(config.paths.train_soundscapes)
        self.n_samples = config.audio.n_samples
        self.sr = config.audio.sample_rate
        self.species_cols = label_encoder.idx2label
        self.mel_xfm = build_mel_transform()
        self.db_xfm = mel_to_db()
        with torch.inference_mode():
            dummy = torch.zeros(1, self.n_samples)
            ref = _postprocess_mel(self.mel_xfm(dummy), self.db_xfm)
            self._empty_melspec = torch.zeros_like(ref)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_pseudo_labels(self, row: pd.Series) -> torch.Tensor:
        probs = row[self.species_cols].to_numpy(dtype=np.float32, copy=False)
        binary = (probs >= 0.5).astype(np.float32)
        return torch.tensor(binary, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        fp = str(self.soundscapes_root / filename)
        empty = {
            "melspec": self._empty_melspec.clone(),
            "labels": self._encode_pseudo_labels(row),
            "filepath": fp,
        }
        try:
            t0 = float(row["start_sec"])
            t1 = float(row["end_sec"])
            f0 = int(round(t0 * self.sr))
            f1 = int(round(t1 * self.sr))
            n_frames = max(1, f1 - f0)
            wav, sr = torchaudio.load(fp, frame_offset=f0, num_frames=n_frames)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            wav = center_crop(wav, self.n_samples)
            mel = _postprocess_mel(self.mel_xfm(wav), self.db_xfm)
            if self.mode == "train" and config.augmentation.use_specaugment:
                mel = apply_spec_augment(mel)
            return {
                "melspec": mel,
                "labels": self._encode_pseudo_labels(row),
                "filepath": fp,
            }
        except Exception:
            return empty


# ---------------------------------------------------------------------------
# 8. DataLoaders
# ---------------------------------------------------------------------------


def get_dataloaders(fold: int) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation ``DataLoader``s for a given fold index.

    - Loads ``train.csv``, adds ``filepath``, applies XC low-rating filter,
      and assigns ``site_group`` from rounded lat/lon (fallback to
      ``StratifiedKFold`` if group stratification fails).
    - Training set: XC reference audio (non-val fold indices) plus **80%** of
      labeled soundscape **segments** (all 66 files represented; no file-level
      holdout — random segment split at ``config.training.seed``).
    - Validation set: **only** the held-out **20%** of soundscape segments
      (honest val — no XC).

    Parameters
    ----------
    fold :
        Validation fold index in ``0 .. config.training.n_folds - 1``.
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
        f"Soundscape segment split (20% val, all files in pool) — "
        f"train segments: {len(train_scape_df)}, val segments: {len(val_scape_df)}"
    )

    train_ref = BirdCLEFDataset(train_df, enc, mode="train")
    train_scape = SoundscapeDataset(train_scape_df, enc, mode="train")
    val_scape = SoundscapeDataset(val_scape_df, enc, mode="val")
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
