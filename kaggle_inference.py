# BirdCLEF 2026 - SAFE + OPTIMIZED INFERENCE + SITE PRIORS

from pathlib import Path
import os
import numpy as np
import pandas as pd
import timm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Ta
from scipy.ndimage import uniform_filter1d
import traceback

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
NUM_CLASSES  = 234
GEM_P        = 3.0
DROPOUT      = 0.2
STRIDE_SEC   = 5.0
BATCH_SIZE   = 8

DATA_ROOT = "/kaggle/input/competitions/birdclef-2026"
PATH      = "/kaggle/input/competitions/birdclef-2026/test_soundscapes"

CHECKPOINTS = [
    ("/kaggle/input/datasets/nehemiahbalozi/birdclef2026-exp001/exp025_perch_distill_fold0_clean.pth", "tf_efficientnet_b5_ns"),
    ("/kaggle/input/datasets/nehemiahbalozi/birdclef2026-exp001/exp026_perch_distill_fold1_clean.pth", "tf_efficientnet_b5_ns"),
    ("/kaggle/input/datasets/nehemiahbalozi/birdclef2026-exp001/exp029_perch_distill_fold4_clean.pth", "tf_efficientnet_b5_ns"),
]

SITE_PRIORS_PATH = "/kaggle/input/datasets/nehemiahbalozi/birdclef2026-exp001/site_priors.json"
SITE_PRIOR_WEIGHT = 0.9  # model^0.9 * prior^0.1

# Model
class GeMPooling(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([float(p)]))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)

class BirdCLEFModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0, global_pool="")
        feat_dim = int(self.backbone.num_features)
        self.pool = GeMPooling()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
        print(f"Model feature dim = {feat_dim}")
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
        return wav[:, start:start+n]
    return F.pad(wav, (0, n - length))

def normalize(mel):
    mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
    std = mel.std()
    if std < 1e-6:
        return torch.zeros_like(mel)
    return (mel - mel.mean()) / (std + 1e-6)

def get_species_cols():
    tax = pd.read_csv(Path(DATA_ROOT) / "taxonomy.csv")
    return sorted(tax["primary_label"].astype(str).unique().tolist())

def get_site(stem):
    parts = stem.split('_')
    for p in parts:
        if p.startswith('S') and p[1:].isdigit():
            return p
    return None

def infer_file_single(filepath, model, mel_xfm, db_xfm, device):
    min_len = N_SEGMENTS * N_SAMPLES
    wav = load_audio(filepath)
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
        i0 = int(round(s * SAMPLE_RATE))
        piece = center_crop(wav[:, i0:i0 + N_SAMPLES], N_SAMPLES)
        mel = mel_xfm(piece)
        mel = mel.clamp(min=1e-10)
        mel = db_xfm(mel)
        mel = normalize(mel)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        chunks.append(mel)
    win_probs_all = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(0, len(chunks), BATCH_SIZE):
            mini = torch.stack(chunks[i:i + BATCH_SIZE], dim=0).to(device)
            logits = model(mini)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            win_probs_all.append(probs.cpu().numpy())
    win_probs = np.concatenate(win_probs_all, axis=0)
    seg_probs = np.zeros((N_SEGMENTS, NUM_CLASSES), dtype=np.float32)
    for k in range(N_SEGMENTS):
        lo, hi = k * 5.0, (k + 1) * 5.0
        idx = [i for i, s in enumerate(starts) if max(lo, s) < min(hi, s + DURATION_SEC)]
        if idx:
            seg_probs[k] = np.maximum.reduce(win_probs[idx])
    return seg_probs

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load site priors
print("Loading site priors...")
if os.path.exists(SITE_PRIORS_PATH):
    with open(SITE_PRIORS_PATH) as f:
        site_priors = json.load(f)
    print(f"Site priors loaded for {len(site_priors)} sites")
else:
    site_priors = {}
    print("WARNING: site_priors.json not found — running without priors")

# Species
species_cols = get_species_cols()
print("Species count:", len(species_cols))
assert len(species_cols) == NUM_CLASSES

# Load models
models = []
for ckpt_path, backbone in CHECKPOINTS:
    print(f"Loading {Path(ckpt_path).name}...")
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BirdCLEFModel(backbone)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    assert len(unexpected) == 0
    model.to(device)
    model.eval()
    models.append(model)
    print(f"  Loaded! AUC={ckpt.get('auc', 'N/A')}")

# Mel transforms
mel_xfm = Ta.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, mel_scale=MEL_SCALE).to(device)
db_xfm = Ta.AmplitudeToDB().to(device)

# Test files
testSounds = [os.path.join(PATH, f) for f in sorted(os.listdir(PATH)) if f.endswith(".ogg")]
print("Test files found:", len(testSounds))

# Inference loop
rows = []
for sound in testSounds:
    stem = Path(sound).stem
    try:
        all_probs = []
        for model in models:
            probs = infer_file_single(sound, model, mel_xfm, db_xfm, device)
            all_probs.append(probs)
        avg_probs = np.mean(all_probs, axis=0)
        avg_probs = uniform_filter1d(avg_probs, size=3, axis=0)
        avg_probs = np.clip(avg_probs, 0, 1)

        # Apply site prior (w=0.9)
        site = get_site(stem)
        if site and site in site_priors:
            prior = np.array([site_priors[site].get(col, 0.5) for col in species_cols])
            avg_probs = (avg_probs ** SITE_PRIOR_WEIGHT) * (prior ** (1 - SITE_PRIOR_WEIGHT))
            avg_probs = np.clip(avg_probs, 0, 1)

        for k in range(N_SEGMENTS):
            row = {"row_id": f"{stem}_{(k+1)*5}"}
            for j, col in enumerate(species_cols):
                row[col] = float(avg_probs[k, j])
            rows.append(row)

    except Exception as e:
        print("ERROR:", stem, type(e).__name__, str(e))
        traceback.print_exc()
        raise e

# Save
predictions = pd.DataFrame(rows, columns=["row_id"] + species_cols)
print("Rows:", len(predictions))
if len(predictions) > 0:
    preds = predictions.iloc[:, 1:].values
    print("Min:", preds.min(), "Max:", preds.max(), "Mean:", preds.mean())
predictions.to_csv("submission.csv", index=False)
print("Done!")
