"""
Parse soundscape filenames into temporal / spatial features.

Filename convention (BirdCLEF+ 2026):

    BC2026_Train_<seq>_S<site>_<YYYYMMDD>_<HHMMSS>.ogg

Extracted columns per file:

- ``filename``  — full basename (key for joining to labels)
- ``site_id``   — integer (e.g. ``S22`` → ``22``)
- ``hour``      — 0..23
- ``month``     — 1..12
- ``is_dawn``   — bool, ``5 <= hour <= 7`` (dawn chorus)
- ``is_night``  — bool, ``hour >= 20 or hour < 5``
- ``season``    — ``0`` = wet (Dec–Mar) / ``1`` = dry (Apr–Nov), Pantanal

Output: ``data/soundscape_location_features.csv`` (under ``config.paths.data_root``
by default). Also prints top sites by recording count and a 24-hour
text histogram of recording times.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import config as cfg

FILENAME_RE = re.compile(
    r"^BC2026_Train_(?P<seq>\d+)_S(?P<site>\d+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\.ogg$"
)

DEFAULT_OUTPUT = (
    Path(cfg.config.paths.data_root) / "data" / "soundscape_location_features.csv"
)

WET_MONTHS = (12, 1, 2, 3)


def parse_filename(name: str) -> dict | None:
    """Return parsed feature dict, or ``None`` if ``name`` doesn't match the schema."""
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    site = int(m.group("site"))
    date_str = m.group("date")
    time_str = m.group("time")
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(time_str[:2])
    minute = int(time_str[2:4])

    is_dawn = 5 <= hour <= 7
    is_night = (hour >= 20) or (hour < 5)
    season = 0 if month in WET_MONTHS else 1

    return {
        "filename": name,
        "site_id": site,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "is_dawn": bool(is_dawn),
        "is_night": bool(is_night),
        "season": int(season),
    }


def build_features_df(soundscapes_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Iterate ``*.ogg`` in ``soundscapes_dir``; return ``(df, skipped_names)``."""
    files = sorted(soundscapes_dir.glob("*.ogg"))
    rows: list[dict] = []
    skipped: list[str] = []
    for fp in files:
        d = parse_filename(fp.name)
        if d is None:
            skipped.append(fp.name)
            continue
        rows.append(d)
    df = pd.DataFrame(rows)
    return df, skipped


def print_site_distribution(df: pd.DataFrame, top_n: int = 15) -> None:
    """Print top ``top_n`` sites by recording count."""
    counts = df["site_id"].value_counts().sort_values(ascending=False)
    print(f"\nSites: {len(counts)} unique. Top {min(top_n, len(counts))} by recording count:")
    print(f"  {'site_id':>8}  {'count':>6}")
    for site, c in counts.head(top_n).items():
        print(f"  S{int(site):02d}    {int(c):>6d}")
    if len(counts) > top_n:
        print(f"  (+{len(counts) - top_n} more)")


def print_hour_histogram(df: pd.DataFrame, width: int = 40) -> None:
    """Print a 24-hour text histogram of recording start times."""
    counts = df["hour"].value_counts().sort_index()
    max_c = int(counts.max()) if len(counts) else 0
    print("\nTime-of-day distribution (recording start hour):")
    print(f"  {'hour':>4}  {'count':>6}")
    for h in range(24):
        c = int(counts.get(h, 0))
        bar_len = int(round(width * c / max_c)) if max_c > 0 else 0
        bar = "#" * bar_len
        marker = ""
        if 5 <= h <= 7:
            marker = "  [dawn]"
        elif h >= 20 or h < 5:
            marker = "  [night]"
        print(f"  {h:02d}    {c:>6d}  {bar}{marker}")


def print_summary(df: pd.DataFrame) -> None:
    """Print dawn / night / season / month aggregates."""
    n = len(df)
    print(f"\nAggregate counts (n={n}):")
    print(f"  is_dawn  = True : {int(df['is_dawn'].sum()):>6d}  ({df['is_dawn'].mean() * 100:5.1f}%)")
    print(f"  is_night = True : {int(df['is_night'].sum()):>6d}  ({df['is_night'].mean() * 100:5.1f}%)")
    season_counts = df["season"].value_counts().sort_index()
    print("\nSeason distribution (0=wet Dec-Mar, 1=dry Apr-Nov):")
    for s, c in season_counts.items():
        label = "wet" if int(s) == 0 else "dry"
        print(f"  season={int(s)} ({label}) : {int(c):>6d}  ({c / n * 100:5.1f}%)")
    month_counts = df["month"].value_counts().sort_index()
    print("\nMonth distribution:")
    for m, c in month_counts.items():
        print(f"  month={int(m):>2d} : {int(c):>6d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse soundscape filenames into temporal/spatial features CSV."
    )
    parser.add_argument(
        "--soundscapes_dir",
        type=Path,
        default=Path(cfg.config.paths.train_soundscapes),
        help=f"Directory of *.ogg files (default: {cfg.config.paths.train_soundscapes})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.soundscapes_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.soundscapes_dir}")

    df, skipped = build_features_df(args.soundscapes_dir)
    n_total = len(df) + len(skipped)
    print(f"Scanned {n_total} files in {args.soundscapes_dir}")
    print(f"Parsed:  {len(df)}")
    print(f"Skipped: {len(skipped)} (filename pattern mismatch)")
    if skipped:
        for name in skipped[:5]:
            print(f"    e.g. {name}")
        if len(skipped) > 5:
            print(f"    (+{len(skipped) - 5} more)")

    if df.empty:
        raise RuntimeError("No filenames matched the expected schema.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    print_site_distribution(df)
    print_hour_histogram(df)
    print_summary(df)


if __name__ == "__main__":
    main()
