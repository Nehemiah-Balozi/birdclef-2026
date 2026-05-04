"""Generate pseudo labels for train_soundscapes from exp006 checkpoint."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast

from config import config
from dataset import LabelEncoder, build_mel_transform, center_crop, load_audio, mel_to_db, normalize_melspec
from model import build_model

cudnn.enabled = False


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Build BirdCLEFModel and load checkpoint weights."""
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


def _segments_to_mel_batch(
    wav: torch.Tensor,
    mel_transform: torch.nn.Module,
    db_transform: torch.nn.Module,
    sr: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert first 60 seconds into a 12x(1,n_mels,time) mel batch."""
    n_samples = config.audio.n_samples
    target_total = 12 * n_samples  # 60 sec
    n = wav.shape[-1]
    if n < target_total:
        wav = torch.nn.functional.pad(wav, (0, target_total - n))
    else:
        wav = wav[:, :target_total]

    chunks: list[torch.Tensor] = []
    for i in range(12):
        t0 = i * n_samples
        t1 = (i + 1) * n_samples
        piece = wav[:, t0:t1]
        piece = center_crop(piece, n_samples)
        piece = piece.to(device)
        mel = mel_transform(piece)
        mel = db_transform(mel)
        mel = normalize_melspec(mel)
        chunks.append(mel)
    return torch.stack(chunks, dim=0)


def _processed_filenames(csv_path: Path) -> set[str]:
    """Read existing pseudo-label csv and return already processed filenames."""
    if not csv_path.exists():
        return set()
    prev = pd.read_csv(csv_path, usecols=["filename"])
    return set(prev["filename"].astype(str).unique().tolist())


def main() -> None:
    root = Path(config.paths.data_root)
    checkpoint_path = root / "experiments/exp006_soundscape_mixed_fold1/checkpoints/fold1_best.pth"
    soundscape_dir = root / "train_soundscapes"
    output_csv = root / "pseudo_labels/pseudo_labels_exp006.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    enc = LabelEncoder()
    species_cols = enc.idx2label  # sorted taxonomy order

    processed = _processed_filenames(output_csv)
    files = sorted(soundscape_dir.glob("*.ogg"))
    pending = [fp for fp in files if fp.name not in processed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(checkpoint_path, device)
    mel_xfm = build_mel_transform().to(device)
    db_xfm = mel_to_db().to(device)

    total = len(files)
    done_before = len(processed)
    print(f"Found {total} soundscape files. Already processed: {done_before}. Pending: {len(pending)}")

    header_cols = ["filename", "start_sec", "end_sec"] + species_cols
    write_header = not output_csv.exists()

    for idx, fp in enumerate(pending, start=1):
        wav = load_audio(fp)
        batch = _segments_to_mel_batch(wav, mel_xfm, db_xfm, config.audio.sample_rate, device)
        batch = batch.to(device, non_blocking=True)

        amp_ctx = autocast("cuda") if device.type == "cuda" else nullcontext()
        with torch.no_grad():
            with amp_ctx:
                logits = model(batch)
            probs = torch.sigmoid(logits.float()).cpu().numpy()

        rows: list[dict[str, float | int | str]] = []
        for seg_idx in range(12):
            row: dict[str, float | int | str] = {
                "filename": fp.name,
                "start_sec": int(seg_idx * 5),
                "end_sec": int((seg_idx + 1) * 5),
            }
            for c, p in zip(species_cols, probs[seg_idx].tolist()):
                row[c] = float(p)
            rows.append(row)

        chunk_df = pd.DataFrame(rows, columns=header_cols)
        chunk_df.to_csv(output_csv, mode="a", header=write_header, index=False)
        write_header = False

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(pending)} pending files ({done_before + idx}/{total} total)")

    print(f"Saved pseudo labels to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
