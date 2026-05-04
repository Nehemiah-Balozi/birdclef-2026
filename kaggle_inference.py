from __future__ import annotations

"""
BirdCLEF+ 2026 — self-contained Kaggle inference script (no local imports).

Loads checkpoints, runs overlap soundscape inference, writes submission CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta
from torch.amp import autocast

# ---------------------------------------------------------------------------
# 2. Kaggle environment config (inline)
# ---------------------------------------------------------------------------

DATA_ROOT = '/kaggle/input/competitions/birdclef-2026'
SAMPLE_RATE = 32000
AUDIO_DURATION_SEC = 5
N_SAMPLES = SAMPLE_RATE * AUDIO_DURATION_SEC

N_MELS = 256
FMIN = 250
FMAX = 16000
HOP_LENGTH = 512
N_FFT = 2048
MEL_SCALE = "htk"

BACKBONE = "tf_efficientnet_b4_ns"
NUM_CLASSES = 234
GEM_P = 3.0
DROPOUT = 0.2

OVERLAP_STRIDE_SEC = 2.5

# ---------------------------------------------------------------------------
# 3. GeMPooling and BirdCLEFModel
# ---------------------------------------------------------------------------


class GeMPooling(nn.Module):
    """Generalized mean pooling over spatial (H, W) with learnable exponent ``p``."""

    def __init__(self, p: float = GEM_P, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([float(p)], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=(-2, -1))
        return x.pow(1.0 / p)


class BirdCLEFModel(nn.Module):
    """1-channel backbone + GeM + MLP head; outputs raw logits."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )
        feat_dim = int(self.backbone.num_features)
        self.pool = GeMPooling()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# ---------------------------------------------------------------------------
# 4. Audio utils
# ---------------------------------------------------------------------------


def load_audio(filepath: str | Path) -> torch.Tensor:
    """Mono waveform at ``SAMPLE_RATE``; stereo averaged; resampled if needed."""
    wav, sr = torchaudio.load(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def center_crop(waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Center crop or symmetric zero-pad to ``n_samples``."""
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


def build_mel_transform() -> Ta.MelSpectrogram:
    """Mel spectrogram matching training config."""
    return Ta.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        mel_scale=MEL_SCALE,
    )


def mel_to_db() -> Ta.AmplitudeToDB:
    return Ta.AmplitudeToDB()


def normalize_melspec(melspec: torch.Tensor) -> torch.Tensor:
    mean = melspec.mean()
    std = melspec.std()
    return (melspec - mean) / (std + 1e-6)


# ---------------------------------------------------------------------------
# 5. LabelEncoder
# ---------------------------------------------------------------------------


class LabelEncoder:
    """234-class order from ``taxonomy.csv`` (sorted ``primary_label``)."""

    def __init__(self, taxonomy_csv: str | Path | None = None) -> None:
        path = Path(taxonomy_csv or (Path(DATA_ROOT) / "taxonomy.csv"))
        tax = pd.read_csv(path)
        self.idx2label: list[str] = sorted(tax["primary_label"].astype(str).unique().tolist())
        self.label2idx: dict[str, int] = {lab: i for i, lab in enumerate(self.idx2label)}
        self.num_classes: int = len(self.idx2label)

    def encode_labels(self, labels: list[str]) -> np.ndarray:
        out = np.zeros(self.num_classes, dtype=np.float32)
        for lab in labels:
            key = str(lab).strip()
            idx = self.label2idx.get(key)
            if idx is not None:
                out[idx] = 1.0
        return out


# ---------------------------------------------------------------------------
# Helpers for overlap inference
# ---------------------------------------------------------------------------


def _prepare_waveform_min_duration(wav: torch.Tensor, min_samples: int) -> torch.Tensor:
    n = wav.shape[-1]
    if n >= min_samples:
        return wav
    return F.pad(wav, (0, min_samples - n))


def _chunk_to_melspec(
    chunk: torch.Tensor,
    mel_transform: nn.Module,
    db_transform: nn.Module,
) -> torch.Tensor:
    mel = mel_transform(chunk)
    mel = db_transform(mel)
    return normalize_melspec(mel)


def _windows_overlap_segment(w_start: float, w_len: float, seg_lo: float, seg_hi: float) -> bool:
    w_end = w_start + w_len
    return max(seg_lo, w_start) < min(seg_hi, w_end)


def _window_starts_sec(duration_sec: float, win_sec: float, stride_sec: float) -> list[float]:
    starts: list[float] = []
    s = 0.0
    while s + win_sec <= duration_sec + 1e-9:
        starts.append(s)
        s += stride_sec
    return starts


# ---------------------------------------------------------------------------
# 6. process_soundscape_overlap
# ---------------------------------------------------------------------------


def process_soundscape_overlap(
    filepath: str | Path,
    model: nn.Module,
    mel_transform: nn.Module,
    db_transform: nn.Module,
    device: torch.device,
    stride_sec: float = OVERLAP_STRIDE_SEC,
) -> np.ndarray:
    """
    Strided 5s windows fused into 12 competition segments via element-wise max
    over overlapping window predictions.
    """
    min_len = 12 * N_SAMPLES
    wav = load_audio(filepath)
    wav = _prepare_waveform_min_duration(wav, min_len)
    duration_sec = float(wav.shape[-1]) / float(SAMPLE_RATE)

    starts = _window_starts_sec(duration_sec, float(AUDIO_DURATION_SEC), stride_sec)

    if not starts:
        return np.zeros((12, NUM_CLASSES), dtype=np.float32)

    batch_chunks: list[torch.Tensor] = []
    for s_sec in starts:
        i0 = int(round(s_sec * SAMPLE_RATE))
        i1 = i0 + N_SAMPLES
        piece = wav[:, i0:i1]
        piece = center_crop(piece, N_SAMPLES)
        batch_chunks.append(_chunk_to_melspec(piece, mel_transform, db_transform))

    batch = torch.stack(batch_chunks, dim=0).to(device, non_blocking=True)
    model.eval()
    with torch.no_grad():
        with autocast("cuda"):
            logits = model(batch)
        win_probs = torch.sigmoid(logits.float()).cpu().numpy()

    seg_probs = np.zeros((12, NUM_CLASSES), dtype=np.float32)
    for k in range(12):
        seg_lo = k * 5.0
        seg_hi = (k + 1) * 5.0
        idx = [
            i
            for i, s in enumerate(starts)
            if _windows_overlap_segment(s, float(AUDIO_DURATION_SEC), seg_lo, seg_hi)
        ]
        if idx:
            seg_probs[k] = np.maximum.reduce(win_probs[idx])

    return seg_probs


def _submission_column_names() -> tuple[str, list[str]]:
    species_cols = LabelEncoder().idx2label
    return "row_id", species_cols


def _load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BirdCLEFModel(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _build_submission_dataframe(test_dir, model, mel_xfm, db_xfm, device):
    row_id_name, species_cols = _submission_column_names()
    test_files = sorted(Path(test_dir).glob('*.ogg'))
    rows = []
    for filepath in test_files:
        stem = filepath.stem
        probs = process_soundscape_overlap(
            filepath, model, mel_xfm, db_xfm, device
        )
        for chunk_idx in range(probs.shape[0]):
            end_sec = (chunk_idx + 1) * 5
            row = {row_id_name: f"{stem}_{end_sec}"}
            for col_idx, col_name in enumerate(species_cols):
                row[col_name] = float(probs[chunk_idx, col_idx])
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 7. run_ensemble
# ---------------------------------------------------------------------------


def run_ensemble(
    checkpoint_paths: list[str | Path],
    test_dir: str | Path,
    output_csv: str | Path,
) -> Path:
    """
    Average species probabilities across checkpoints (aligned by ``row_id``),
    then save submission CSV. Uses overlap inference only.
    """
    device = torch.device("cuda")
    row_id_name, species_cols = _submission_column_names()
    test_dir = Path(test_dir)
    output_csv = Path(output_csv)

    dfs: list[pd.DataFrame] = []
    for ckpt in checkpoint_paths:
        model = _load_model_from_checkpoint(ckpt, device)
        mel_xfm = build_mel_transform().to(device)
        db_xfm = mel_to_db().to(device)
        df = _build_submission_dataframe(test_dir, model, mel_xfm, db_xfm, device)
        dfs.append(df)

    base = dfs[0].copy()
    for c in species_cols:
        base[c] = float(np.mean([d[c].to_numpy(dtype=np.float64) for d in dfs], axis=0))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_csv, index=False)
    return output_csv


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    checkpoint_path = '/kaggle/input/datasets/nehemiahbalozi/birdclef2026-exp001/fold0_best_clean.pth'
    test_dir = '/kaggle/input/competitions/birdclef-2026/test_soundscapes'
    output = '/kaggle/working/submission.csv'
    run_ensemble([checkpoint_path], test_dir, output)
    print('Saved to', output)
