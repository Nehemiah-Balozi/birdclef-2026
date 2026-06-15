#!/usr/bin/env python3
"""
Smoke test: load PERCH as ONNX and extract a 1536-d embedding from one audio file.

Research summary (BirdCLEF+ / integration prep)
-----------------------------------------------
**PERCH** (part of Google Research `chirp` / “Perch” line of work) is a
bioacoustics model trained on large-scale animal vocalizations. It maps
short audio clips to a fixed-dimensional embedding suitable for retrieval,
few-shot, or downstream classifiers. The public codebase lives at:
https://github.com/google-research/perch

**Distribution formats**
- Original exports are often **TensorFlow SavedModel** / **TFLite**.
- Community **ONNX** builds are easier for PyTorch-centric pipelines; documented
  I/O appears in projects such as Hugging Face `justinchuby/Perch-onnx` and the
  Kaggle dataset `rishikeshjani/perch-onnx-for-birdclef-2026` (download the
  `.onnx` file(s) locally — this script does not fetch from the network).

**Important I/O detail for ONNX**
The common PERCH v2 ONNX graph takes **raw waveform**, not a user-supplied mel:
  - Input name is typically ``inputs`` with shape ``[batch, 160000]`` float32
    (5 seconds at **32000 Hz**, mono).
  - The model’s **frontend** computes the PCEN mel / spectrogram internally.
  - Main embedding output is shape ``[batch, 1536]`` (plus optional tensors
    such as spatial embedding, internal spectrogram, and a large classifier
    head over ~14k taxa labels).

So for exp010 we will either:
- feed **waveform chunks** into ONNX / ORT (matches current export), or
- later replace the frontend in a PyTorch module if we need true “mel-in” API.

This script only verifies ONNX loading + one forward pass.

Dependencies
------------
  pip install onnxruntime torchaudio torch

Usage
-----
  export PERCH_ONNX_PATH=/path/to/perch_v2.onnx   # or perch_v2_no_dft.onnx
  python test_perch.py --audio /path/to/file.ogg

If ``--audio`` is omitted, uses the first ``*.ogg`` under ``train_soundscapes/``
relative to this repo’s data root (see ``_DEFAULT_DATA_ROOT`` below).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

import torch.nn.functional as F
import torchaudio

# Same default local root as config.py when not on Kaggle
_DEFAULT_DATA_ROOT = Path("/home/rise/Documents/Acoustics/BirdCLEF/birdclef-2026")
_PERCH_SAMPLE_RATE = 32000
_PERCH_N_SAMPLES = 160000  # 5 s @ 32 kHz


def _resolve_onnx_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()
    env = os.environ.get("PERCH_ONNX_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Convenient local layout after manual download
    candidate = _DEFAULT_DATA_ROOT / "perch_onnx" / "perch_v2.onnx"
    return candidate.resolve()


def _default_audio_path() -> Path:
    scapes = _DEFAULT_DATA_ROOT / "train_soundscapes"
    if scapes.is_dir():
        oggs = sorted(scapes.glob("*.ogg"))
        if oggs:
            return oggs[0]
    ta = _DEFAULT_DATA_ROOT / "train_audio"
    if ta.is_dir():
        for ext in ("*.ogg", "*.flac", "*.wav", "*.mp3"):
            files = sorted(ta.rglob(ext))
            if files:
                return files[0]
    raise FileNotFoundError(
        f"No default audio under {scapes} or {ta}. Pass --audio explicitly."
    )


def load_mono_32k(path: Path, max_samples: int) -> np.ndarray:
    """Load file, mono mix, resample to 32 kHz, return float32 (max_samples,)"""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != _PERCH_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, _PERCH_SAMPLE_RATE)
    x = wav.squeeze(0)
    n = x.shape[0]
    if n >= max_samples:
        # center crop a 5 s window
        start = (n - max_samples) // 2
        x = x[start : start + max_samples]
    else:
        x = F.pad(x, (0, max_samples - n))
    return x.numpy().astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="PERCH ONNX embedding smoke test")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to PERCH .onnx (else PERCH_ONNX_PATH env or ./perch_onnx/perch_v2.onnx)",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Audio file (ogg/wav/…). Default: first train_soundscapes/*.ogg",
    )
    args = parser.parse_args()

    if ort is None:
        print("Install ONNX Runtime:  pip install onnxruntime", file=sys.stderr)
        return 1

    onnx_path = _resolve_onnx_path(args.onnx)
    if not onnx_path.is_file():
        print(
            "PERCH ONNX not found.\n"
            f"  Expected: {onnx_path}\n"
            "  Download an ONNX from e.g. Kaggle `rishikeshjani/perch-onnx-for-birdclef-2026` "
            "or Hugging Face `justinchuby/Perch-onnx`, then set:\n"
            "    export PERCH_ONNX_PATH=/full/path/to/perch_v2.onnx",
            file=sys.stderr,
        )
        return 1

    audio_path = args.audio.expanduser().resolve() if args.audio else _default_audio_path()
    if not audio_path.is_file():
        print(f"Audio not found: {audio_path}", file=sys.stderr)
        return 1

    providers = ["CPUExecutionProvider"]
    if ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"ONNX:   {onnx_path}")
    print(f"Audio:  {audio_path}")
    print(f"ORT:    providers={providers}")

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    in_meta = session.get_inputs()
    out_meta = session.get_outputs()
    print("Inputs:")
    for i in in_meta:
        print(f"  {i.name}: shape={i.shape}, type={i.type}")
    print("Outputs:")
    for i in out_meta:
        print(f"  {i.name}: shape={i.shape}, type={i.type}")

    waveform = load_mono_32k(audio_path, _PERCH_N_SAMPLES)
    batch = waveform.reshape(1, -1)

    inp0 = in_meta[0]
    shp = list(inp0.shape) if inp0.shape is not None else []
    ok = False
    if len(shp) == 2:
        d1 = shp[1]
        if d1 in (160000, "160000"):
            ok = True
    if not ok:
        print(
            f"Warning: unexpected input shape {shp}; "
            "expected [batch, 160000] for standard PERCH v2 ONNX.",
            file=sys.stderr,
        )

    feeds = {inp0.name: batch}
    outputs = session.run(None, feeds)

    # Convention: first output = embedding [1, 1536]
    embedding = np.asarray(outputs[0], dtype=np.float32)
    print(f"Embedding output shape: {embedding.shape} dtype={embedding.dtype}")
    print(f"Embedding L2 norm: {float(np.linalg.norm(embedding.ravel())):.4f}")
    print(f"Embedding min/max: {float(embedding.min()):.4f} / {float(embedding.max()):.4f}")

    if embedding.ndim == 2 and embedding.shape[1] == 1536:
        print("OK: 1536-dim embedding as expected for standard PERCH v2 ONNX.")
    else:
        print(
            f"Note: embedding dim is {embedding.shape}; adjust exp010 head if needed.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
