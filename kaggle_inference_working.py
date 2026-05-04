# BirdCLEF+ 2026 - Kaggle inference (self-contained)
# Follows Marcus Wang template pattern exactly

from pathlib import Path
import os
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta

torch.backends.cudnn.enabled = False

# Config
SAMPLE_RATE  = 32000
DURATION_SEC = 5
N_SAMPLES    = SAMPLE_RATE * DURATION_SEC
N_SEGMENTS   = 12
N_MELS       = 256
FMIN         = 250
FMAX         = 16000
HOP_LENGTH   = 512
N_FFT        = 2048
MEL_SCALE    = "htk"
BACKBONE     = "tf_efficientnet_b4_ns"
NUM_CLASSES  = 234
GEM_P        = 3.0
DROPOUT      = 0.2
STRIDE_SEC   = 2.5

DATA_ROOT  = "/kaggle/input/competitions/birdclef-2026"
CHECKPOINT = "/kaggle/input/datasets/nehemiahbalozi/birdclef2026-exp001/fold0_best_clean.pth"
PATH       = "/kaggle/input/competitions/birdclef-2026/test_soundscapes/"

# Model
class GeMPooling(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor([float(p)]))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)

class BirdCLEFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=False, in_chans=1, num_classes=0, global_pool="")
        feat_dim  = int(self.backbone.num_features)
        self.pool = GeMPooling()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(torch.flatten(x, 1))

# Audio utils
def load_audio(filepath):
    wav, sr = torchaudio.load(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav

def center_crop(wav, n):
    length = wav.shape[-1]
    if length == n:
        return wav
    if length > n:
        start = (length - n) // 2
        return wav[:, start: start + n]
    return F.pad(wav, (0, n - length))

def normalize(mel):
    return (mel - mel.mean()) / (mel.std() + 1e-6)

def get_species_cols():
    tax = pd.read_csv(Path(DATA_ROOT) / "taxonomy.csv")
    return sorted(tax["primary_label"].astype(str).unique().tolist())

def infer_file(filepath, model, mel_xfm, db_xfm, device):
    try:
        min_len  = N_SEGMENTS * N_SAMPLES
        wav      = load_audio(filepath)
        if wav.shape[-1] < min_len:
            wav = F.pad(wav, (0, min_len - wav.shape[-1]))
        duration = wav.shape[-1] / SAMPLE_RATE

        starts, s = [], 0.0
        while s + DURATION_SEC <= duration + 1e-9:
            starts.append(s)
            s += STRIDE_SEC

        if not starts:
            return np.zeros((N_SEGMENTS, NUM_CLASSES), dtype=np.float32)

        chunks = []
        for s in starts:
            i0    = int(round(s * SAMPLE_RATE))
            piece = center_crop(wav[:, i0: i0 + N_SAMPLES], N_SAMPLES)
            mel   = mel_xfm(piece)
            mel   = db_xfm(mel)
            mel   = normalize(mel)
            chunks.append(mel)

        batch = torch.stack(chunks, dim=0).to(device)
        with torch.no_grad():
            logits    = model(batch)
            win_probs = torch.sigmoid(logits).cpu().numpy()

        seg_probs = np.zeros((N_SEGMENTS, NUM_CLASSES), dtype=np.float32)
        for k in range(N_SEGMENTS):
            lo, hi = k * 5.0, (k + 1) * 5.0
            idx    = [i for i, s in enumerate(starts)
                      if max(lo, s) < min(hi, s + DURATION_SEC)]
            if idx:
                seg_probs[k] = np.maximum.reduce(win_probs[idx])
        return seg_probs

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return np.zeros((N_SEGMENTS, NUM_CLASSES), dtype=np.float32)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("Loading model...")
ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model = BirdCLEFModel()
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()
print("Model loaded.")

mel_xfm      = Ta.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, mel_scale=MEL_SCALE).to(device)
db_xfm       = Ta.AmplitudeToDB().to(device)
species_cols = get_species_cols()

# Template pattern: empty list during Save & Run is fine
testSounds = [os.path.join(PATH, f)
              for f in sorted(os.listdir(PATH)) if f.endswith(".ogg")]
print("Test files found:", len(testSounds))

# Collect rows as list (fast), not loc append (very slow)
rows = []
for sound in testSounds:
    stem  = Path(sound).stem
    probs = infer_file(sound, model, mel_xfm, db_xfm, device)
    for k in range(N_SEGMENTS):
        row = {"row_id": f"{stem}_{(k+1)*5}"}
        for j, col in enumerate(species_cols):
            row[col] = float(probs[k, j])
        rows.append(row)
    print("Processed:", stem)

predictions = pd.DataFrame(rows, columns=["row_id"] + species_cols)
predictions.to_csv("submission.csv", index=False)
print("Done! Rows:", len(predictions))
