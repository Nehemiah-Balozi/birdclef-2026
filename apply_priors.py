"""
Blend model predictions with ecological priors via a weighted geometric mean.

Formula (per class, independent for multilabel sigmoid outputs)::

    final = clip(model_prob, eps)^w * clip(prior, eps)^(1 - w)

For a checkpoint's submission CSV (``row_id``, then 234 species columns) this
script parses the site ID from each ``row_id`` (pattern ``_S\\d{2}_``), looks
up ``P(species | site)``, and writes an adjusted submission CSV.

If a ``row_id``'s site is unknown (test files without site info, or a site
not seen during training), the row is blended with the **global** prior
instead, which is a safe no-op for ranking when ``w`` is close to ``1``.

Library usage::

    from apply_priors import (
        extract_site_from_stem, combine_geometric, load_site_priors,
    )

CLI usage::

    python apply_priors.py \\
        --predictions experiments/<exp>/submission/submission.csv \\
        --site_priors data/site_priors.json \\
        --output experiments/<exp>/submission/submission_priors.csv \\
        --model_weight 0.8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


EPS = 1e-6
SITE_PATTERN = re.compile(r"_S(\d{2})_")

DEFAULT_SITE_PRIORS = Path(cfg.config.paths.data_root) / "data" / "site_priors.json"


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def extract_site_from_stem(stem: str) -> str | None:
    """Find ``S<NN>`` substring in a filename / row_id stem, return ``"S<NN>"`` or ``None``.

    Examples
    --------
    >>> extract_site_from_stem("BC2026_Train_0039_S22_20211231_201500")
    'S22'
    >>> extract_site_from_stem("BC2026_Train_0039_S22_20211231_201500_5")
    'S22'
    >>> extract_site_from_stem("some_unknown_test_file") is None
    True
    """
    m = SITE_PATTERN.search(stem)
    if m is None:
        return None
    return f"S{m.group(1)}"


def combine_geometric(
    probs: np.ndarray,
    prior: np.ndarray,
    model_weight: float = 0.8,
    eps: float = EPS,
) -> np.ndarray:
    """Per-class geometric blend ``probs^w * prior^(1 - w)``.

    Parameters
    ----------
    probs : ``(..., C)`` model probabilities in ``[0, 1]``.
    prior : ``(C,)`` or ``(..., C)`` ecological prior.
    model_weight : weight on ``probs``; ``1.0`` returns ``probs`` unchanged.
    eps : floor applied to both arrays before the log to avoid ``log(0)``.
    """
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be in [0, 1]")
    p = np.clip(probs, eps, 1.0)
    q = np.clip(np.broadcast_to(prior, p.shape), eps, 1.0)
    return np.exp(model_weight * np.log(p) + (1.0 - model_weight) * np.log(q))


def load_site_priors(path: Path) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """Load priors JSON. Returns ``(species_order, sites_dict, global_prior)``."""
    with Path(path).open() as f:
        data = json.load(f)
    species_order = list(data["species_order"])
    sites = {k: np.asarray(v, dtype=np.float64) for k, v in data["sites"].items()}
    global_prior = np.asarray(data["global"], dtype=np.float64)
    return species_order, sites, global_prior


# ---------------------------------------------------------------------------
# CLI: apply priors to a submission CSV
# ---------------------------------------------------------------------------


def _stem_from_row_id(row_id: str) -> str:
    """Submission ``row_id`` is ``"{stem}_{end_sec}"``; strip the trailing seconds."""
    parts = str(row_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return str(row_id)


def apply_to_submission(
    df: pd.DataFrame,
    species_order: list[str],
    sites: dict[str, np.ndarray],
    global_prior: np.ndarray,
    model_weight: float = 0.8,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return ``(adjusted_df, stats)``. ``df`` must have ``row_id`` and 234 species columns."""
    if "row_id" not in df.columns:
        raise ValueError("predictions CSV must have a 'row_id' column")
    missing = [c for c in species_order if c not in df.columns]
    if missing:
        raise ValueError(
            f"predictions CSV is missing {len(missing)} species columns "
            f"(first few: {missing[:5]})"
        )

    probs = df[species_order].to_numpy(dtype=np.float64)
    n_rows, n_classes = probs.shape

    prior_mat = np.empty_like(probs)
    n_site_match = 0
    n_global_fallback = 0
    site_hits: dict[str, int] = {}

    for i, rid in enumerate(df["row_id"].astype(str).to_numpy()):
        stem = _stem_from_row_id(rid)
        site = extract_site_from_stem(stem)
        if site is not None and site in sites:
            prior_mat[i] = sites[site]
            n_site_match += 1
            site_hits[site] = site_hits.get(site, 0) + 1
        else:
            prior_mat[i] = global_prior
            n_global_fallback += 1

    adjusted = combine_geometric(probs, prior_mat, model_weight=model_weight)

    out = df.copy()
    out[species_order] = adjusted

    stats = {
        "rows": int(n_rows),
        "classes": int(n_classes),
        "site_match": int(n_site_match),
        "global_fallback": int(n_global_fallback),
        **{f"site:{k}": int(v) for k, v in sorted(site_hits.items())},
    }
    return out, stats


def _demo() -> None:
    """Tiny self-contained example (no I/O) showing the blend behaviour."""
    print("Library demo: combine_geometric")
    model_probs = np.array([0.90, 0.05, 0.40, 0.99])
    site_prior = np.array([0.50, 0.30, 0.001, 0.20])
    for w in (1.0, 0.9, 0.8, 0.5):
        out = combine_geometric(model_probs, site_prior, model_weight=w)
        print(f"  w={w:.2f}  ->  {np.round(out, 4).tolist()}")
    print("  (w=1.0 returns model_probs unchanged; smaller w shrinks classes")
    print("   that are ecologically implausible at this site.)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply site priors to a submission CSV via geometric blend."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=False,
        help="Input submission CSV (row_id + 234 species columns)",
    )
    parser.add_argument(
        "--site_priors",
        type=Path,
        default=DEFAULT_SITE_PRIORS,
        help=f"Site priors JSON (default: {DEFAULT_SITE_PRIORS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Output CSV path (default: <predictions>.priors.csv)",
    )
    parser.add_argument(
        "--model_weight",
        type=float,
        default=0.8,
        help="Weight on model probabilities in the geometric blend (default: 0.8)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Print a small library usage demo and exit",
    )
    args = parser.parse_args()

    if args.demo or args.predictions is None:
        _demo()
        if args.predictions is None:
            print("(Pass --predictions <csv> to apply priors to a submission.)")
            return

    if not args.site_priors.is_file():
        raise FileNotFoundError(
            f"Site priors not found: {args.site_priors} "
            "(run compute_site_priors.py first)"
        )

    species_order, sites, global_prior = load_site_priors(args.site_priors)
    print(f"Loaded {len(sites)} sites, {len(species_order)} species from {args.site_priors}")

    df = pd.read_csv(args.predictions)
    adjusted, stats = apply_to_submission(
        df, species_order, sites, global_prior, model_weight=args.model_weight
    )

    out_path = args.output or args.predictions.with_suffix(".priors.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(out_path, index=False)
    print(f"Saved adjusted submission: {out_path}")

    print("\nSummary:")
    print(f"  rows:             {stats['rows']}")
    print(f"  classes:          {stats['classes']}")
    print(f"  site_match:       {stats['site_match']}")
    print(f"  global_fallback:  {stats['global_fallback']}")
    print(f"  model_weight:     {args.model_weight}")
    if stats["site_match"]:
        print("  rows per site (top 10):")
        site_counts = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("site:")}
        for site, n in sorted(site_counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {site}: {n}")


if __name__ == "__main__":
    main()
