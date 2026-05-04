"""
BirdCLEF+ 2026 — soundscape inference and Kaggle-style submission CSV export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast

from config import config
from dataset import (
    LabelEncoder,
    build_mel_transform,
    center_crop,
    load_audio,
    mel_to_db,
    normalize_melspec,
)
from model import build_model

import torch.backends.cudnn as cudnn

cudnn.enabled = False


def _prepare_waveform_min_duration(wav: torch.Tensor, min_samples: int) -> torch.Tensor:
    """Right-pad mono waveform ``(1, n)`` to at least ``min_samples``."""
    n = wav.shape[-1]
    if n >= min_samples:
        return wav
    pad = min_samples - n
    return nn.functional.pad(wav, (0, pad))


def _chunk_to_melspec(
    chunk: torch.Tensor,
    mel_transform: nn.Module,
    db_transform: nn.Module,
) -> torch.Tensor:
    """Mel → dB → normalize; input ``(1, n_samples)`` → ``(1, n_mels, time)``."""
    mel = mel_transform(chunk)
    mel = db_transform(mel)
    return normalize_melspec(mel)


def process_soundscape(
    filepath: str | Path,
    model: nn.Module,
    mel_transform: nn.Module,
    db_transform: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Non-overlapping 12×5s windows on a (padded) 60s soundscape.

    Loads the full file, pads to 60s if shorter, splits into 12 contiguous
    5-second chunks, runs one forward pass on the batch, applies sigmoid.

    Returns
    -------
    np.ndarray
        Shape ``(12, num_classes)`` with probabilities in ``[0, 1]``.
    """
    n_samples = config.audio.n_samples
    min_len = 12 * n_samples

    wav = load_audio(filepath)
    wav = _prepare_waveform_min_duration(wav, min_len)

    chunks: list[torch.Tensor] = []
    for i in range(12):
        piece = wav[:, i * n_samples : (i + 1) * n_samples]
        piece = center_crop(piece, n_samples)
        chunks.append(_chunk_to_melspec(piece, mel_transform, db_transform))

    batch = torch.stack(chunks, dim=0).to(device, non_blocking=True)
    model.eval()
    with torch.no_grad():
        with autocast("cuda"):
            logits = model(batch)
        probs = torch.sigmoid(logits.float())
    return probs.cpu().numpy()


def _windows_overlap_segment(
    w_start: float,
    w_len: float,
    seg_lo: float,
    seg_hi: float,
) -> bool:
    """True if window ``[w_start, w_start + w_len)`` intersects ``[seg_lo, seg_hi)``."""
    w_end = w_start + w_len
    return max(seg_lo, w_start) < min(seg_hi, w_end)


def _window_starts_sec(duration_sec: float, win_sec: float, stride_sec: float) -> list[float]:
    """Start times (seconds) for sliding windows that fully fit in ``duration_sec``."""
    starts: list[float] = []
    s = 0.0
    while s + win_sec <= duration_sec + 1e-9:
        starts.append(s)
        s += stride_sec
    return starts


def process_soundscape_overlap(
    filepath: str | Path,
    model: nn.Module,
    mel_transform: nn.Module,
    db_transform: nn.Module,
    device: torch.device,
    stride_sec: float = 2.5,
) -> np.ndarray:
    """
    Sliding 5s windows with stride ``stride_sec``; fuse into 12 official segments via max.

    Each competition segment ``[k*5, (k+1)*5)`` seconds receives the **element-wise
    maximum** over probabilities from all windows that overlap that interval.
    Audio is padded to at least 60s so segment indices align with the baseline.
    """
    sr = config.audio.sample_rate
    n_samples = config.audio.n_samples
    win_sec = float(config.audio.duration)
    min_len = 12 * n_samples

    wav = load_audio(filepath)
    wav = _prepare_waveform_min_duration(wav, min_len)
    duration_sec = float(wav.shape[-1]) / float(sr)

    starts = _window_starts_sec(duration_sec, win_sec, stride_sec)
    n_classes = config.model.num_classes

    if not starts:
        return np.zeros((12, n_classes), dtype=np.float32)

    batch_chunks: list[torch.Tensor] = []
    for s_sec in starts:
        i0 = int(round(s_sec * sr))
        i1 = i0 + n_samples
        piece = wav[:, i0:i1]
        piece = center_crop(piece, n_samples)
        batch_chunks.append(_chunk_to_melspec(piece, mel_transform, db_transform))

    batch = torch.stack(batch_chunks, dim=0).to(device, non_blocking=True)
    model.eval()
    with torch.no_grad():
        with autocast("cuda"):
            logits = model(batch)
        win_probs = torch.sigmoid(logits.float()).cpu().numpy()

    seg_probs = np.zeros((12, n_classes), dtype=np.float32)
    for k in range(12):
        seg_lo = k * 5.0
        seg_hi = (k + 1) * 5.0
        idx = [i for i, s in enumerate(starts) if _windows_overlap_segment(s, win_sec, seg_lo, seg_hi)]
        if idx:
            seg_probs[k] = np.maximum.reduce(win_probs[idx])

    return seg_probs


def _submission_column_names() -> tuple[str, list[str]]:
    """``row_id`` key and species column names in sample submission order."""
    sample_path = Path(config.paths.data_root) / "sample_submission.csv"
    if sample_path.is_file():
        cols = list(pd.read_csv(sample_path, nrows=0).columns)
        if cols and cols[0] == "row_id":
            return cols[0], cols[1:]
    enc = LabelEncoder()
    return "row_id", enc.idx2label


def _load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Build model (no ImageNet download) and load ``model_state`` weights."""
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


def _build_submission_dataframe(
    test_dir: Path,
    model: nn.Module,
    mel_xfm: nn.Module,
    db_xfm: nn.Module,
    device: torch.device,
    use_overlap: bool,
) -> pd.DataFrame:
    """All test files → long-form dataframe with ``row_id`` and species columns."""
    row_id_name, species_cols = _submission_column_names()
    proc = process_soundscape_overlap if use_overlap else process_soundscape

    files = sorted(test_dir.glob("*.ogg"))
    rows: list[dict] = []

    for fp in files:
        stem = fp.stem
        probs = proc(fp, model, mel_xfm, db_xfm, device)
        for j in range(12):
            end_sec = (j + 1) * 5
            rid = f"{stem}_{end_sec}"
            row = {row_id_name: rid}
            for c, p in zip(species_cols, probs[j].tolist()):
                row[c] = p
            rows.append(row)

    df = pd.DataFrame(rows)
    return df[[row_id_name] + species_cols]


def run_inference(
    checkpoint_path: str | Path,
    test_dir: str | Path,
    output_csv: str | Path,
    use_overlap: bool = True,
) -> Path:
    """
    Run a single checkpoint over all ``.ogg`` files in ``test_dir`` and save CSV.

    Parameters
    ----------
    checkpoint_path :
        ``.pth`` file with ``model_state`` (or flat ``state_dict``).
    test_dir :
        Directory containing competition ``test_soundscapes`` audio.
    output_csv :
        Destination path for submission CSV.
    use_overlap :
        If ``True``, use :func:`process_soundscape_overlap`; else
        :func:`process_soundscape`.
    """
    device = torch.device("cuda")
    model = _load_model_from_checkpoint(checkpoint_path, device)
    mel_xfm = build_mel_transform().to(device)
    db_xfm = mel_to_db().to(device)

    test_dir = Path(test_dir)
    output_csv = Path(output_csv)
    df = _build_submission_dataframe(test_dir, model, mel_xfm, db_xfm, device, use_overlap)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def run_ensemble(
    checkpoint_paths: list[str | Path],
    test_dir: str | Path,
    output_csv: str | Path,
    use_overlap: bool = True,
) -> Path:
    """
    Average per-row probabilities across multiple checkpoints, then save CSV.

    Each checkpoint is evaluated on the full ``test_dir``; predictions are
    aligned by ``row_id`` and averaged element-wise for the 234 species columns.
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
        df = _build_submission_dataframe(test_dir, model, mel_xfm, db_xfm, device, use_overlap)
        dfs.append(df)

    base = dfs[0].copy()
    for c in species_cols:
        base[c] = float(np.mean([d[c].to_numpy(dtype=np.float64) for d in dfs], axis=0))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 inference")
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path(config.paths.data_root) / "test_soundscapes",
        help="Directory of test .ogg files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(config.paths.submission_dir) / "submission.csv",
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="One or more .pth checkpoint paths",
    )
    parser.add_argument(
        "--no_overlap",
        action="store_true",
        help="Use non-overlapping 12×5s chunks instead of strided overlap + max fuse",
    )
    args = parser.parse_args()

    use_overlap = not args.no_overlap
    if len(args.checkpoints) > 1:
        out = run_ensemble(args.checkpoints, args.test_dir, args.output, use_overlap=use_overlap)
    else:
        out = run_inference(args.checkpoints[0], args.test_dir, args.output, use_overlap=use_overlap)

    print(f"Saved submission: {out.resolve()}")


if __name__ == "__main__":
    main()
