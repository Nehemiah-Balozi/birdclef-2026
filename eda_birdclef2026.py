#!/usr/bin/env python3
"""BirdCLEF+ 2026 — exploratory data analysis (standalone script).

Reads train.csv, taxonomy.csv, train_soundscapes_labels.csv, and train_audio/.
Writes figures to --out and prints summaries to stdout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from tqdm.auto import tqdm

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(root / "train.csv")
    tax = pd.read_csv(root / "taxonomy.csv")
    scape_labels = pd.read_csv(root / "train_soundscapes_labels.csv")
    return train, tax, scape_labels


def explode_soundscape_labels(series: pd.Series) -> set[str]:
    out: set[str] = set()
    for cell in series.dropna():
        for part in str(cell).split(";"):
            part = part.strip()
            if part:
                out.add(part)
    return out


def parse_site_id(filename: str) -> str | None:
    m = re.search(r"_S(\d+)_\d{8}_", filename)
    return f"S{m.group(1)}" if m else None


def section_1_class_distribution(
    train: pd.DataFrame, taxonomy: pd.DataFrame, out_dir: Path
) -> None:
    counts = train["primary_label"].value_counts().sort_values(ascending=True)
    counts_df = (
        train["primary_label"]
        .value_counts()
        .rename_axis("primary_label")
        .reset_index(name="n_samples")
        .merge(
            taxonomy[["primary_label", "scientific_name", "common_name", "class_name"]],
            on="primary_label",
            how="left",
        )
    )

    fig, ax = plt.subplots(figsize=(14, max(6, len(counts) * 0.12)))
    colors = sns.color_palette("viridis", n_colors=len(counts))
    ax.barh(range(len(counts)), counts.values, color=colors)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index.astype(str), fontsize=6)
    ax.set_xlabel("Number of train.csv rows (reference recordings)")
    ax.set_title("Samples per species (sorted ascending)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "01_samples_per_species.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== 1. Class distribution (top 15 by count) ===")
    print(counts_df.sort_values("n_samples", ascending=False).head(15).to_string(index=False))
    print("\n... bottom 15 ...")
    print(counts_df.sort_values("n_samples", ascending=True).head(15).to_string(index=False))


def section_2_taxonomy_breakdown(taxonomy: pd.DataFrame, out_dir: Path) -> None:
    class_counts = taxonomy["class_name"].value_counts()
    print("\n=== 2. Taxonomy — species per class ===")
    print(class_counts.to_frame("n_species").to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    class_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Number of species in taxonomy")
    ax.set_xlabel("Class")
    ax.set_title("Taxonomy: species per class (234 total labels)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "02_taxonomy_class_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def section_3_soundscape_only(
    taxonomy: pd.DataFrame,
    soundscape_labels_dedup: pd.DataFrame,
    audio_species_dirs: list[str],
) -> None:
    labels_in_soundscapes = explode_soundscape_labels(soundscape_labels_dedup["primary_label"])
    audio_dir_set = set(audio_species_dirs)
    soundscape_only = sorted(labels_in_soundscapes - audio_dir_set)
    tax_labels = set(taxonomy["primary_label"].astype(str))
    unknown_in_tax = sorted(labels_in_soundscapes - tax_labels)

    print("\n=== 3. Soundscape-only species (in labels, no train_audio folder) ===")
    print(f"Unique label tokens in soundscape annotations: {len(labels_in_soundscapes)}")
    print(f"Tokens not in taxonomy.csv: {unknown_in_tax if unknown_in_tax else 'none'}")
    print(f"Tokens in soundscapes but NOT in train_audio/: {len(soundscape_only)}")
    only_tbl = taxonomy[taxonomy["primary_label"].isin(soundscape_only)].sort_values("class_name")
    if len(only_tbl):
        print(only_tbl.to_string(index=False))
    no_audio_in_tax = taxonomy[~taxonomy["primary_label"].isin(audio_dir_set)]["primary_label"].tolist()
    print(f"\nTaxonomy rows with no train_audio folder: {len(no_audio_in_tax)}")


def section_4_durations(train_audio_root: Path, taxonomy: pd.DataFrame, out_dir: Path) -> None:
    print("\n=== 4. Audio durations (train_audio/) ===")
    ogg_paths = sorted(train_audio_root.rglob("*.ogg"))
    records: list[dict] = []
    errors: list[tuple[str, str]] = []

    for path in tqdm(ogg_paths, desc="Reading .ogg durations"):
        species = path.parent.name
        try:
            info = sf.info(path)
            dur = float(info.duration)
            records.append(
                {
                    "path": str(path),
                    "primary_label": species,
                    "duration_sec": dur,
                    "samplerate": info.samplerate,
                }
            )
        except Exception as e:
            errors.append((str(path), repr(e)))

    durations_df = pd.DataFrame.from_records(records)
    print(f"Parsed {len(durations_df):,} files; failures: {len(errors)}")
    if errors[:5]:
        print("Example errors:", errors[:5])

    if len(durations_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(durations_df["duration_sec"], bins=80, color="coral", edgecolor="white")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count (files)")
    axes[0].set_title("All reference recordings: duration distribution")

    merged = durations_df.merge(taxonomy[["primary_label", "class_name"]], on="primary_label", how="left")
    order = merged["class_name"].value_counts().index.tolist()
    sns.boxplot(data=merged, x="class_name", y="duration_sec", order=order, ax=axes[1])
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Duration (s)")
    axes[1].set_title("Duration by taxonomic class")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "04_audio_durations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    per_species = (
        durations_df.groupby("primary_label")["duration_sec"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
        .merge(taxonomy[["primary_label", "scientific_name", "class_name"]], on="primary_label", how="left")
        .sort_values("mean", ascending=False)
    )
    print("\nLongest mean duration (12 species):")
    print(per_species.head(12).to_string(index=False))
    print("\nShortest mean duration (12 species):")
    print(per_species.tail(12).to_string(index=False))


def section_5_ratings(train: pd.DataFrame, out_dir: Path) -> None:
    print("\n=== 5. Rating distribution ===")
    r = pd.to_numeric(train["rating"], errors="coerce")
    low = (r < 2).sum()
    total = r.notna().sum()
    print(f"Rows with rating < 2: {low:,} / {total:,} ({100 * low / total:.2f}%)")
    print(f"Rows with rating NaN: {r.isna().sum():,}")

    fig, ax = plt.subplots(figsize=(10, 4))
    r.dropna().hist(bins=50, ax=ax, color="seagreen", edgecolor="white")
    ax.axvline(2, color="red", linestyle="--", label="rating = 2")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("train.csv rating distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "05_ratings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    by_coll = train.assign(rating=r).groupby("collection")["rating"].agg(
        n="count", frac_below_2=lambda s: (s < 2).mean()
    )
    print(by_coll.to_string())


def section_6_collection(train: pd.DataFrame, taxonomy: pd.DataFrame, out_dir: Path) -> None:
    print("\n=== 6. Collection split (XC vs iNat) ===")
    overall = train["collection"].value_counts()
    print(overall.to_frame("n_rows").to_string())

    ct = (
        train.groupby(["primary_label", "collection"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=lambda c: f"n_{str(c).lower()}")
    )
    if "n_xc" not in ct.columns:
        ct["n_xc"] = 0
    if "n_inat" not in ct.columns:
        ct["n_inat"] = 0
    ct = ct.reset_index().merge(
        taxonomy[["primary_label", "scientific_name", "class_name"]], on="primary_label", how="left"
    )
    ncols = [c for c in ct.columns if c.startswith("n_")]
    ct["n_total"] = ct[ncols].sum(axis=1)
    ct["frac_xc"] = ct["n_xc"] / ct["n_total"].replace(0, np.nan)
    print("\nTop 20 species by total train rows:")
    print(ct.sort_values("n_total", ascending=False).head(20).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4))
    overall.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black")
    ax.set_title("train.csv rows by collection")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "06_collection_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def section_7_geography(train: pd.DataFrame, out_dir: Path) -> None:
    print("\n=== 7. Geographic spread ===")
    geo = train[["latitude", "longitude", "primary_label", "class_name"]].copy()
    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    gvalid = geo.dropna(subset=["latitude", "longitude"])
    print(f"Rows with valid lat/lon: {len(gvalid):,} / {len(train):,}")

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, sub in gvalid.groupby("class_name"):
        ax.scatter(sub["longitude"], sub["latitude"], s=4, alpha=0.35, label=cls)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("train.csv recording locations (by class_name)")
    ax.legend(markerscale=3, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "07_lat_lon_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def section_8_labeled_soundscapes(soundscape_labels_dedup: pd.DataFrame) -> None:
    print("\n=== 8. Labeled soundscape segments (deduplicated) ===")
    n_seg = len(soundscape_labels_dedup)
    n_files = soundscape_labels_dedup["filename"].nunique()
    species_in_scape = explode_soundscape_labels(soundscape_labels_dedup["primary_label"])
    print(f"Labeled 5-second segments: {n_seg:,}")
    print(f"Unique soundscape files with ≥1 labeled segment: {n_files:,}")
    print(f"Unique species / label tokens in annotations: {len(species_in_scape)}")
    rows_per_file = soundscape_labels_dedup.groupby("filename").size()
    print("\nSegments per file (describe):")
    print(rows_per_file.describe().to_frame("segments_per_file").to_string())


def section_9_sites(data_root: Path, soundscape_labels_dedup: pd.DataFrame) -> None:
    soundscape_labels_dedup = soundscape_labels_dedup.assign(
        site_id=soundscape_labels_dedup["filename"].map(parse_site_id)
    )
    print("\n=== 9. Site metadata (from filenames) ===")
    missing_site = soundscape_labels_dedup["site_id"].isna().sum()
    print(f"Labeled rows without parsed site_id: {missing_site}")
    sites_from_labels = soundscape_labels_dedup["site_id"].dropna().unique()
    print(f"Unique sites in labeled soundscape filenames: {len(sites_from_labels)}")
    print("\nTop 20 sites by labeled segment count:")
    print(soundscape_labels_dedup["site_id"].value_counts().head(20).to_string())

    ts_root = data_root / "train_soundscapes"
    if ts_root.is_dir():
        on_disk = [p.name for p in ts_root.glob("*.ogg")]
        sites_disk = {parse_site_id(f) for f in on_disk}
        sites_disk.discard(None)
        print(f"\ntrain_soundscapes/ .ogg count: {len(on_disk)}")
        print(f"Unique sites (parsed from on-disk filenames): {len(sites_disk)}")


def section_10_imbalance(train: pd.DataFrame, taxonomy: pd.DataFrame) -> None:
    print("\n=== 10. Class imbalance ===")
    c = train["primary_label"].value_counts()
    positive = c[c > 0]
    ratio = positive.max() / positive.min()
    print(f"Species with ≥1 train row: {len(positive)} / {len(taxonomy)} taxonomy entries")
    print(f"Most samples: {positive.idxmax()} → {positive.max():,} rows")
    print(f"Fewest samples: {positive.idxmin()} → {positive.min():,} rows")
    print(f"Imbalance ratio (max/min): {ratio:,.2f}")
    zero_in_train = sorted(set(taxonomy["primary_label"]) - set(positive.index.astype(str)))
    print(f"\nTaxonomy species with ZERO train.csv rows: {len(zero_in_train)}")
    if zero_in_train:
        z = taxonomy[taxonomy["primary_label"].isin(zero_in_train)]
        print(z.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 EDA")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder with train.csv, train_audio/, etc. (default: script directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for figures (default: <data-root>/eda_output)",
    )
    parser.add_argument(
        "--skip-durations",
        action="store_true",
        help="Skip scanning all train_audio .ogg files (faster)",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    out_dir = (args.out or (data_root / "eda_output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100

    print("DATA_ROOT =", data_root)
    print("OUT_DIR   =", out_dir)

    train, taxonomy, soundscape_labels = load_tables(data_root)
    train["primary_label"] = train["primary_label"].astype(str)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    soundscape_labels_dedup = soundscape_labels.drop_duplicates(
        subset=["filename", "start", "end"], keep="first"
    ).reset_index(drop=True)

    train_audio_root = data_root / "train_audio"
    audio_species_dirs = (
        sorted([p.name for p in train_audio_root.iterdir() if p.is_dir()])
        if train_audio_root.is_dir()
        else []
    )

    print(f"\ntrain.csv rows: {len(train):,}")
    print(f"taxonomy species: {len(taxonomy)}")
    print(f"soundscape label rows (raw): {len(soundscape_labels):,}")
    print(f"soundscape segments (unique): {len(soundscape_labels_dedup):,}")
    print(f"train_audio subfolders: {len(audio_species_dirs)}")

    section_1_class_distribution(train, taxonomy, out_dir)
    section_2_taxonomy_breakdown(taxonomy, out_dir)
    section_3_soundscape_only(taxonomy, soundscape_labels_dedup, audio_species_dirs)
    if not args.skip_durations and train_audio_root.is_dir():
        section_4_durations(train_audio_root, taxonomy, out_dir)
    elif args.skip_durations:
        print("\n=== 4. Audio durations — skipped (--skip-durations) ===")
    else:
        print("\n=== 4. Audio durations — skipped (no train_audio/) ===")

    section_5_ratings(train, out_dir)
    section_6_collection(train, taxonomy, out_dir)
    section_7_geography(train, out_dir)
    section_8_labeled_soundscapes(soundscape_labels_dedup)
    section_9_sites(data_root, soundscape_labels_dedup)
    section_10_imbalance(train, taxonomy)

    print(f"\nFigures saved under: {out_dir}")


if __name__ == "__main__":
    main()
