"""
Compute ecological priors from labeled soundscape segments.

Joins ``data/soundscape_location_features.csv`` (from
``extract_location_features.py``) with ``train_soundscapes_labels.csv`` by
``filename``, then for each segment treats the semicolon-separated
``primary_label`` as a 234-dim multihot vector (taxonomy order).

Outputs (under ``<data_root>/data/``):

- ``site_priors.json``  — ``P(species | site)`` per site (and a global mean)
- ``hour_priors.json``  — ``P(species | hour_bucket)`` for buckets
  ``night`` (20–04), ``dawn`` (05–07), ``evening`` (18–19), ``other``

Both JSONs share the schema::

    {
      "species_order": [...234 strings in LabelEncoder.idx2label order],
      "sites" | "buckets": {"<key>": [...234 floats], ...},
      "global":  [...234 floats],
      "support": {"<key>": <n_segments>, ...}
    }

Also prints the **top 5 most site-specific species** (highest variance of
``P(species | site)`` across sites with sufficient support).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from dataset import LabelEncoder


DEFAULT_FEATURES_CSV = (
    Path(cfg.config.paths.data_root) / "data" / "soundscape_location_features.csv"
)
DEFAULT_SITE_OUT = Path(cfg.config.paths.data_root) / "data" / "site_priors.json"
DEFAULT_HOUR_OUT = Path(cfg.config.paths.data_root) / "data" / "hour_priors.json"

MIN_SEGMENTS_FOR_VARIANCE = 30  # sites with fewer segments are noisy; skip in ranking


def get_val_segment_keys() -> set[tuple[str, str, str]]:
    """Return ``(filename, start, end)`` keys for the honest-val 20% segment split.

    Mirrors the val split built in :func:`dataset.get_dataloaders`
    (and :func:`tune_thresholds.build_val_segments_df`) — seeded by
    ``config.training.seed`` so it stays in sync across scripts.
    """
    scape = pd.read_csv(cfg.config.paths.soundscape_labels)
    scape = scape.drop_duplicates(
        subset=["filename", "start", "end"], keep="first"
    ).reset_index(drop=True)
    n_seg = len(scape)
    if n_seg == 0:
        return set()
    rng = np.random.default_rng(cfg.config.training.seed)
    perm = rng.permutation(n_seg)
    n_val = int(round(0.2 * n_seg))
    n_val = max(1, min(n_val, n_seg))
    val_idx = perm[:n_val]
    val_df = scape.iloc[val_idx]
    return set(
        zip(
            val_df["filename"].astype(str),
            val_df["start"].astype(str),
            val_df["end"].astype(str),
        )
    )


def hour_bucket(h: int) -> str:
    """Bucket an hour ``0..23`` into ``night | dawn | evening | other``."""
    if h >= 20 or h <= 4:
        return "night"
    if 5 <= h <= 7:
        return "dawn"
    if 18 <= h <= 19:
        return "evening"
    return "other"


def encode_segment(cell: object, enc: LabelEncoder) -> np.ndarray:
    """Semicolon-separated species cell → ``(num_classes,)`` multihot float32."""
    parts = [p.strip() for p in str(cell).split(";") if p.strip()]
    return enc.encode_labels(parts)


def build_segment_matrix(
    merged: pd.DataFrame, enc: LabelEncoder
) -> np.ndarray:
    """Multihot ``(N, num_classes)`` over rows of ``merged`` (column ``primary_label``)."""
    n = len(merged)
    out = np.zeros((n, enc.num_classes), dtype=np.float32)
    for i, cell in enumerate(merged["primary_label"].to_numpy()):
        out[i] = encode_segment(cell, enc)
    return out


def group_means(
    label_mat: np.ndarray, keys: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Group-wise mean of multihot labels by ``keys`` (1-D array of equal length).

    Returns ``(priors, support)`` where ``priors[k]`` is the mean vector
    (``P(species | k)``) and ``support[k]`` is the row count.
    """
    priors: dict[str, np.ndarray] = {}
    support: dict[str, int] = {}
    keys_str = np.asarray(keys).astype(str)
    for k in np.unique(keys_str):
        mask = keys_str == k
        priors[k] = label_mat[mask].mean(axis=0).astype(np.float64)
        support[k] = int(mask.sum())
    return priors, support


def save_priors_json(
    out_path: Path,
    species_order: list[str],
    group_key: str,
    priors: dict[str, np.ndarray],
    support: dict[str, int],
    global_prior: np.ndarray,
) -> None:
    """Persist priors to JSON using the shared schema."""
    payload = {
        "species_order": list(species_order),
        group_key: {k: priors[k].tolist() for k in sorted(priors.keys())},
        "global": global_prior.tolist(),
        "support": {k: int(support[k]) for k in sorted(support.keys())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def report_top_site_specific(
    site_priors: dict[str, np.ndarray],
    support: dict[str, int],
    species_order: list[str],
    top_n: int = 5,
    min_support: int = MIN_SEGMENTS_FOR_VARIANCE,
) -> None:
    """Print the ``top_n`` species with highest variance of ``P(sp|site)`` across sites."""
    eligible = sorted(s for s, n in support.items() if n >= min_support)
    if len(eligible) < 2:
        print("\nNot enough high-support sites to rank site-specific species.")
        return
    mat = np.stack([site_priors[s] for s in eligible], axis=0)
    var = mat.var(axis=0)
    order = np.argsort(-var)[:top_n]

    print(f"\nTop {top_n} most site-specific species "
          f"(variance of P(species|site) across {len(eligible)} sites with >= {min_support} segments):")
    print(f"  {'species':<14} {'variance':>10} {'min_site':<14} {'max_site':<14}")
    for j in order:
        col = mat[:, j]
        i_min = int(col.argmin())
        i_max = int(col.argmax())
        sp = species_order[j]
        print(
            f"  {sp:<14} {float(var[j]):>10.5f} "
            f"{eligible[i_min]}={col[i_min]:5.3f}  "
            f"{eligible[i_max]}={col[i_max]:5.3f}"
        )


def report_hour_top_species(
    hour_priors: dict[str, np.ndarray],
    species_order: list[str],
    top_n: int = 5,
) -> None:
    """Per bucket, print top species by prior probability."""
    print("\nTop species per hour bucket (by P(species|bucket)):")
    for bucket in ("night", "dawn", "evening", "other"):
        if bucket not in hour_priors:
            continue
        vec = hour_priors[bucket]
        order = np.argsort(-vec)[:top_n]
        items = ", ".join(f"{species_order[int(j)]}={float(vec[int(j)]):.3f}" for j in order)
        print(f"  [{bucket:<7}] {items}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute site / hour species priors from labeled soundscape segments."
    )
    parser.add_argument(
        "--features_csv",
        type=Path,
        default=DEFAULT_FEATURES_CSV,
        help=f"Per-file features CSV (default: {DEFAULT_FEATURES_CSV})",
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        default=Path(cfg.config.paths.soundscape_labels),
        help=f"Soundscape labels CSV (default: {cfg.config.paths.soundscape_labels})",
    )
    parser.add_argument(
        "--site_out",
        type=Path,
        default=DEFAULT_SITE_OUT,
        help=f"Output site priors JSON (default: {DEFAULT_SITE_OUT})",
    )
    parser.add_argument(
        "--hour_out",
        type=Path,
        default=DEFAULT_HOUR_OUT,
        help=f"Output hour priors JSON (default: {DEFAULT_HOUR_OUT})",
    )
    parser.add_argument(
        "--exclude_val",
        action="store_true",
        help="Exclude the honest-val 20% soundscape segments (same split as "
        "get_dataloaders / tune_thresholds) before computing priors. Use this "
        "to get unbiased downstream eval — otherwise priors leak val labels.",
    )
    args = parser.parse_args()

    if not args.features_csv.is_file():
        raise FileNotFoundError(
            f"Features CSV not found: {args.features_csv} "
            "(run extract_location_features.py first)"
        )

    feat = pd.read_csv(args.features_csv)
    needed = {"filename", "site_id", "hour"}
    missing = needed - set(feat.columns)
    if missing:
        raise ValueError(f"features_csv missing columns: {sorted(missing)}")

    labels = pd.read_csv(args.labels_csv)
    labels = labels.drop_duplicates(
        subset=["filename", "start", "end"], keep="first"
    ).reset_index(drop=True)

    if args.exclude_val:
        val_keys = get_val_segment_keys()
        if val_keys:
            keys = list(
                zip(
                    labels["filename"].astype(str),
                    labels["start"].astype(str),
                    labels["end"].astype(str),
                )
            )
            mask = np.array([k not in val_keys for k in keys])
            n_before = len(labels)
            labels = labels.loc[mask].reset_index(drop=True)
            print(
                f"--exclude_val: dropped {n_before - len(labels)} val segments "
                f"({len(val_keys)} val keys), kept {len(labels)} train segments"
            )

    merged = labels.merge(
        feat[["filename", "site_id", "hour"]], on="filename", how="inner"
    ).reset_index(drop=True)
    n_unmatched = len(labels) - len(merged)
    print(f"Labeled segments: {len(labels)}  matched to features: {len(merged)}  "
          f"unmatched: {n_unmatched}")

    enc = LabelEncoder()
    label_mat = build_segment_matrix(merged, enc)
    print(f"Multihot matrix: {label_mat.shape}")

    # --- Site priors ---------------------------------------------------------
    site_keys = np.array([f"S{int(s):02d}" for s in merged["site_id"].to_numpy()])
    site_priors, site_support = group_means(label_mat, site_keys)
    global_prior = label_mat.mean(axis=0).astype(np.float64)
    save_priors_json(
        args.site_out, enc.idx2label, "sites", site_priors, site_support, global_prior
    )
    print(f"\nSaved {args.site_out}  ({len(site_priors)} sites)")
    print("  site segment support (top 10):")
    top_sites = sorted(site_support.items(), key=lambda kv: -kv[1])[:10]
    for s, n in top_sites:
        print(f"    {s}: {n}")

    # --- Hour-bucket priors --------------------------------------------------
    bucket_keys = np.array([hour_bucket(int(h)) for h in merged["hour"].to_numpy()])
    hour_priors, hour_support = group_means(label_mat, bucket_keys)
    save_priors_json(
        args.hour_out, enc.idx2label, "buckets", hour_priors, hour_support, global_prior
    )
    print(f"\nSaved {args.hour_out}  ({len(hour_priors)} buckets)")
    for b in ("night", "dawn", "evening", "other"):
        n = hour_support.get(b, 0)
        print(f"    {b:<7} : {n} segments")

    # --- Reports -------------------------------------------------------------
    report_top_site_specific(site_priors, site_support, enc.idx2label, top_n=5)
    report_hour_top_species(hour_priors, enc.idx2label, top_n=5)


if __name__ == "__main__":
    main()
