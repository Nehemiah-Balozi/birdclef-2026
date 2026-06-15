"""
Grid-search the site-prior blend weight ``model_weight`` against the honest
148-segment soundscape validation set, reporting macro ROC-AUC before / after.

Uses the same val split, mel pipeline and checkpoint loader as
``tune_thresholds.py``, and the same blending formula as ``apply_priors.py``::

    final = clip(model, eps)^w * clip(site_prior, eps)^(1 - w)

For each row in val, the site is parsed from the segment ``filename`` (e.g.
``BC2026_Train_0039_S22_...`` → ``S22``); if a site isn't in the priors JSON,
the global prior is used as a safe fallback.

Inference is run **once**; blending and AUC are computed in NumPy for each
grid value, so the grid is cheap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config as cfg
from apply_priors import combine_geometric, extract_site_from_stem, load_site_priors
from dataset import LabelEncoder
from tune_thresholds import (
    DEFAULT_CHECKPOINT,
    ValSegmentDataset,
    build_val_segments_df,
    load_model_from_checkpoint,
    run_inference,
)


DEFAULT_SITE_PRIORS = Path(cfg.config.paths.data_root) / "data" / "site_priors.json"
DEFAULT_GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def macro_auc(labels: np.ndarray, probs: np.ndarray) -> tuple[float, int]:
    """Macro ROC-AUC over classes with both positives and negatives.

    Matches the convention used by ``train.compute_auc``. Returns
    ``(mean_auc, n_classes_scored)``.
    """
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    scores: list[float] = []
    n_c = labels.shape[1]
    for j in range(n_c):
        y = labels[:, j]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos == 0 or neg == 0:
            continue
        if np.isnan(probs[:, j]).any():
            continue
        scores.append(float(roc_auc_score(y, probs[:, j])))
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


def build_row_priors(
    val_df: pd.DataFrame,
    sites: dict[str, np.ndarray],
    global_prior: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], int]:
    """Per-row prior matrix ``(N, C)``; fallback to ``global_prior`` if site missing.

    Returns ``(prior_mat, site_hits, n_fallback)``.
    """
    n = len(val_df)
    c = global_prior.shape[0]
    out = np.empty((n, c), dtype=np.float64)
    site_hits: dict[str, int] = {}
    n_fallback = 0
    for i, fname in enumerate(val_df["filename"].astype(str).to_numpy()):
        stem = fname[: -len(".ogg")] if fname.endswith(".ogg") else fname
        site = extract_site_from_stem(stem)
        if site is not None and site in sites:
            out[i] = sites[site]
            site_hits[site] = site_hits.get(site, 0) + 1
        else:
            out[i] = global_prior
            n_fallback += 1
    return out, site_hits, n_fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search site-prior blend weight on the honest val set."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint .pth (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--site_priors",
        type=Path,
        default=DEFAULT_SITE_PRIORS,
        help=f"Site priors JSON (default: {DEFAULT_SITE_PRIORS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg.config.training.batch_size,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cfg.config.training.num_workers,
    )
    parser.add_argument(
        "--grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_GRID),
        help="Values of model_weight to sweep (default: 0.70 .. 1.00 step 0.05)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.site_priors.is_file():
        raise FileNotFoundError(
            f"Site priors not found: {args.site_priors} "
            "(run compute_site_priors.py first)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Site priors: {args.site_priors}")

    val_df = build_val_segments_df()
    print(f"Val segments: {len(val_df)}")

    enc = LabelEncoder()
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

    species_order, sites, global_prior = load_site_priors(args.site_priors)
    if species_order != enc.idx2label:
        raise ValueError(
            "Species order mismatch between site_priors.json and current LabelEncoder."
        )

    prior_mat, site_hits, n_fallback = build_row_priors(val_df, sites, global_prior)
    print(f"\nVal site coverage: {len(val_df) - n_fallback}/{len(val_df)} "
          f"segments matched a known site prior ({n_fallback} fell back to global).")
    for s, n in sorted(site_hits.items(), key=lambda kv: -kv[1]):
        print(f"  {s}: {n}")

    baseline_auc, n_scored = macro_auc(labels, probs)
    print(f"\nBaseline macro AUC (w=1.0, no prior): {baseline_auc:.6f}  "
          f"over {n_scored} classes with both pos and neg")

    grid = sorted(set(round(float(w), 4) for w in args.grid))
    print(f"\nGrid sweep ({len(grid)} values):")
    print(f"  {'w':>6}  {'macro_auc':>10}  {'delta':>9}")
    results: list[tuple[float, float]] = []
    for w in grid:
        if w == 1.0:
            adj = probs
        else:
            adj = combine_geometric(probs, prior_mat, model_weight=w)
        auc, _ = macro_auc(labels, adj)
        results.append((w, auc))
        delta = auc - baseline_auc
        marker = "  <- baseline" if w == 1.0 else ""
        print(f"  {w:>6.2f}  {auc:>10.6f}  {delta:>+9.6f}{marker}")

    best_w, best_auc = max(results, key=lambda x: x[1])
    delta = best_auc - baseline_auc

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Baseline (w=1.00): {baseline_auc:.6f}")
    print(f"  Best:   w={best_w:.2f}   {best_auc:.6f}")
    print(f"  Δ AUC vs baseline: {delta:+.6f}")
    if best_w == 1.0:
        print("  → Priors did not help on this val. (Likely the val is dominated by "
              "one site, so the per-row prior is nearly constant per class and the "
              "blend is monotonic → AUC unchanged.)")


if __name__ == "__main__":
    main()
