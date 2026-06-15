"""
Pre-extract PERCH ONNX embeddings for XC clips and soundscape segments.

Saves (under ``<data_root>/perch_embeddings/``):

- ``xc_embeddings.npz``: ``embeddings`` (N, 1536), ``labels`` (N, 234) multihot
  float32, ``primary_label`` (N,) int32 taxonomy index (-1 if unknown),
  ``filepaths`` (N,) str, ``completed`` (N,) bool (for resume).
- ``soundscape_embeddings.npz``: ``embeddings`` (M, 1536), ``labels`` (M, 234),
  ``filename``, ``start_sec``, ``end_sec``, ``completed``.

Resume: ``completed`` mask; unfinished rows are filled on rerun. Checkpoint save
every 100 batches; progress printed every 100 batches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from config import config
from dataset import (
    LabelEncoder,
    _add_filepath_column,
    _apply_xc_low_rating_filter,
    _parse_secondary_labels,
    _parse_soundscape_time_to_seconds,
    center_crop,
    load_audio,
)

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("pip install onnxruntime") from e


def _make_ort_session(onnx_path: Path) -> tuple[ort.InferenceSession, str]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def _encode_row_multihot(row: pd.Series, enc: LabelEncoder) -> np.ndarray:
    primary = [str(row["primary_label"]).strip()]
    secondary = _parse_secondary_labels(row["secondary_labels"])
    return enc.encode_labels(primary + secondary)


def _encode_scape_multihot(cell: object, enc: LabelEncoder) -> np.ndarray:
    parts = [p.strip() for p in str(cell).split(";") if p.strip()]
    return enc.encode_labels(parts)


def _waveform_xc(filepath: str, n_samples: int) -> np.ndarray:
    """Mono 32 kHz, center-cropped to n_samples (same as val-style crop)."""
    try:
        wav = load_audio(filepath)
        wav = center_crop(wav, n_samples)
        return wav.squeeze(0).numpy().astype(np.float32, copy=False)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)


def _waveform_soundscape(
    fp: Path,
    t0: float,
    t1: float,
    sr: int,
    n_samples: int,
) -> np.ndarray:
    try:
        f0 = int(round(t0 * sr))
        f1 = int(round(t1 * sr))
        n_frames = max(1, f1 - f0)
        wav, file_sr = torchaudio.load(str(fp), frame_offset=f0, num_frames=n_frames)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        wav = center_crop(wav, n_samples)
        return wav.squeeze(0).numpy().astype(np.float32, copy=False)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)


def _run_batch(session: ort.InferenceSession, in_name: str, batch_wav: np.ndarray) -> np.ndarray:
    """batch_wav: (B, n_samples) float32"""
    out = session.run(None, {in_name: batch_wav})[0]
    return np.asarray(out, dtype=np.float32)


def extract_xc(
    session: ort.InferenceSession,
    in_name: str,
    out_path: Path,
    batch_size: int,
    n_samples: int,
) -> None:
    enc = LabelEncoder()
    df = pd.read_csv(config.paths.train_csv)
    df = _add_filepath_column(df)
    df = _apply_xc_low_rating_filter(df).reset_index(drop=True)
    n = len(df)
    filepaths = df["filepath"].astype(str).tolist()
    labels = np.stack([_encode_row_multihot(df.iloc[i], enc) for i in range(n)], axis=0).astype(
        np.float32
    )
    primary_idx = np.array(
        [enc.label2idx.get(str(df.iloc[i]["primary_label"]).strip(), -1) for i in range(n)],
        dtype=np.int32,
    )

    embeddings = np.zeros((n, 1536), dtype=np.float32)
    completed = np.zeros(n, dtype=bool)

    if out_path.is_file():
        z = np.load(out_path, allow_pickle=True)
        ok = (
            z["embeddings"].shape[0] == n
            and z["filepaths"].shape[0] == n
            and np.all(z["filepaths"].astype(str) == np.array(filepaths, dtype=object))
        )
        if ok:
            embeddings = np.array(z["embeddings"], dtype=np.float32, copy=True)
            completed = np.array(z["completed"], dtype=bool, copy=True)
            print(f"Resume XC: {int(completed.sum())}/{n} rows already done.")
        else:
            print("XC npz mismatch — starting fresh.")

    pending = np.where(~completed)[0]
    n_batches = (len(pending) + batch_size - 1) // batch_size
    done_batches = 0

    for s in range(0, len(pending), batch_size):
        chunk_idx = pending[s : s + batch_size]
        bw = np.stack([_waveform_xc(filepaths[i], n_samples) for i in chunk_idx], axis=0)
        emb = _run_batch(session, in_name, bw)
        embeddings[chunk_idx] = emb
        completed[chunk_idx] = True
        done_batches += 1
        if done_batches % 100 == 0 or s + batch_size >= len(pending):
            print(f"XC batch {done_batches}/{n_batches} — completed {int(completed.sum())}/{n}")
            np.savez_compressed(
                out_path,
                embeddings=embeddings,
                labels=labels,
                primary_label=primary_idx,
                filepaths=np.array(filepaths, dtype=object),
                completed=completed,
            )

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        labels=labels,
        primary_label=primary_idx,
        filepaths=np.array(filepaths, dtype=object),
        completed=completed,
    )
    print(f"Wrote {out_path} (XC N={n})")


def extract_soundscapes(
    session: ort.InferenceSession,
    in_name: str,
    out_path: Path,
    batch_size: int,
    n_samples: int,
    sr: int,
) -> None:
    enc = LabelEncoder()
    scape = pd.read_csv(config.paths.soundscape_labels)
    scape = scape.drop_duplicates(subset=["filename", "start", "end"], keep="first").reset_index(
        drop=True
    )
    m = len(scape)
    root = Path(config.paths.train_soundscapes)

    filenames: list[str] = []
    start_secs: list[float] = []
    end_secs: list[float] = []
    labels_list: list[np.ndarray] = []
    for i in range(m):
        row = scape.iloc[i]
        filenames.append(str(row["filename"]))
        start_secs.append(float(_parse_soundscape_time_to_seconds(row["start"])))
        end_secs.append(float(_parse_soundscape_time_to_seconds(row["end"])))
        labels_list.append(_encode_scape_multihot(row["primary_label"], enc))
    labels = np.stack(labels_list, axis=0).astype(np.float32)
    start_arr = np.array(start_secs, dtype=np.float32)
    end_arr = np.array(end_secs, dtype=np.float32)
    fn_arr = np.array(filenames, dtype=object)

    embeddings = np.zeros((m, 1536), dtype=np.float32)
    completed = np.zeros(m, dtype=bool)

    if out_path.is_file():
        z = np.load(out_path, allow_pickle=True)
        ok = (
            z["embeddings"].shape[0] == m
            and np.all(z["filename"].astype(str) == fn_arr.astype(str))
            and np.allclose(z["start_sec"].astype(np.float32), start_arr)
            and np.allclose(z["end_sec"].astype(np.float32), end_arr)
        )
        if ok:
            embeddings = np.array(z["embeddings"], dtype=np.float32, copy=True)
            completed = np.array(z["completed"], dtype=bool, copy=True)
            print(f"Resume soundscapes: {int(completed.sum())}/{m} rows already done.")
        else:
            print("Soundscape npz mismatch — starting fresh.")

    pending = np.where(~completed)[0]
    n_batches = (len(pending) + batch_size - 1) // batch_size
    done_batches = 0

    for s in range(0, len(pending), batch_size):
        chunk_idx = pending[s : s + batch_size]
        bw_list: list[np.ndarray] = []
        for i in chunk_idx:
            fp = root / filenames[i]
            bw_list.append(
                _waveform_soundscape(fp, start_secs[i], end_secs[i], sr, n_samples)
            )
        bw = np.stack(bw_list, axis=0)
        emb = _run_batch(session, in_name, bw)
        embeddings[chunk_idx] = emb
        completed[chunk_idx] = True
        done_batches += 1
        if done_batches % 100 == 0 or s + batch_size >= len(pending):
            print(
                f"Soundscape batch {done_batches}/{n_batches} — completed {int(completed.sum())}/{m}"
            )
            np.savez_compressed(
                out_path,
                embeddings=embeddings,
                labels=labels,
                filename=fn_arr,
                start_sec=start_arr,
                end_sec=end_arr,
                completed=completed,
            )

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        labels=labels,
        filename=fn_arr,
        start_sec=start_arr,
        end_sec=end_arr,
        completed=completed,
    )
    print(f"Wrote {out_path} (soundscapes M={m})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PERCH embeddings to disk")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--xc_only", action="store_true")
    parser.add_argument("--scape_only", action="store_true")
    args = parser.parse_args()

    root = Path(config.paths.data_root)
    onnx_path = Path(config.paths.perch_onnx)
    out_dir = root / "perch_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    session, in_name = _make_ort_session(onnx_path)
    n_samples = config.audio.n_samples
    sr = config.audio.sample_rate

    if not args.scape_only:
        extract_xc(session, in_name, out_dir / "xc_embeddings.npz", args.batch_size, n_samples)
    if not args.xc_only:
        extract_soundscapes(
            session, in_name, out_dir / "soundscape_embeddings.npz", args.batch_size, n_samples, sr
        )


if __name__ == "__main__":
    main()
