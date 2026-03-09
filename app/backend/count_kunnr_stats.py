#!/usr/bin/env python3
"""
Count non-null Z_MU_KUNNR and Z_PR_KUNNR values per country across all recipe JSON files.

Usage:
    python count_kunnr_stats.py [--data-dir /path/to/data] [--workers N]

Defaults:
    --data-dir  ../data  (relative to this script)
    --workers   number of CPU cores
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

COUNTRY_CODE_RE = re.compile(r'_([A-Z]{2})\d{2}_')
TARGET_CHARACTS = {"Z_MU_KUNNR", "Z_PR_KUNNR"}

COUNTRY_CODE_MAP = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LV": "Latvia", "NL": "Netherlands",
    "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
    "TR": "Turkey", "UA": "Ukraine", "US": "United States",
}


def extract_country(filename: str) -> str:
    m = COUNTRY_CODE_RE.search(filename)
    return m.group(1) if m else "XX"


def process_batch(file_paths: list[str]) -> dict:
    """Process a batch of files and return per-country counts."""
    counts = defaultdict(lambda: {
        "total": 0,
        "has_mu": 0,
        "has_pr": 0,
        "has_both": 0,
        "has_neither": 0,
    })

    for fpath in file_paths:
        fname = os.path.basename(fpath)
        country = extract_country(fname)
        counts[country]["total"] += 1

        found = set()
        try:
            with open(fpath, "rb") as f:
                data = json.loads(f.read())

            valuesnum = data.get("Classification", {}).get("valuesnum", [])
            for item in valuesnum:
                ch = item.get("charact")
                if ch in TARGET_CHARACTS and item.get("valueFrom") is not None:
                    found.add(ch)
                    if len(found) == 2:
                        break
        except Exception:
            pass

        has_mu = "Z_MU_KUNNR" in found
        has_pr = "Z_PR_KUNNR" in found

        if has_mu:
            counts[country]["has_mu"] += 1
        if has_pr:
            counts[country]["has_pr"] += 1
        if has_mu and has_pr:
            counts[country]["has_both"] += 1
        if not has_mu and not has_pr:
            counts[country]["has_neither"] += 1

    return dict(counts)


def merge_counts(target: dict, source: dict):
    for country, vals in source.items():
        if country not in target:
            target[country] = {"total": 0, "has_mu": 0, "has_pr": 0, "has_both": 0, "has_neither": 0}
        for k in vals:
            target[country][k] += vals[k]


def main():
    parser = argparse.ArgumentParser(description="Count Z_MU_KUNNR / Z_PR_KUNNR stats per country")
    parser.add_argument("--data-dir", default=None, help="Path to the data directory with JSON files")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Files per worker batch (default: 2000)")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    workers = args.workers or os.cpu_count() or 4
    batch_size = args.batch_size

    print(f"Scanning {data_dir} ...")
    t0 = time.time()

    files = [
        entry.path for entry in os.scandir(data_dir)
        if entry.is_file() and entry.name.endswith(".json")
    ]
    scan_time = time.time() - t0
    total_files = len(files)
    print(f"Found {total_files:,} JSON files in {scan_time:.1f}s")
    print(f"Processing with {workers} workers, batch size {batch_size} ...")

    batches = [files[i:i + batch_size] for i in range(0, total_files, batch_size)]
    merged = {}
    processed = 0
    t1 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_batch, batch): len(batch) for batch in batches}
        for future in as_completed(futures):
            result = future.result()
            merge_counts(merged, result)
            processed += futures[future]
            elapsed = time.time() - t1
            rate = processed / elapsed if elapsed > 0 else 0
            pct = processed / total_files * 100
            print(f"\r  Progress: {processed:>9,} / {total_files:,}  ({pct:5.1f}%)  [{rate:,.0f} files/s]", end="", flush=True)

    total_time = time.time() - t0
    print(f"\n\nDone in {total_time:.1f}s ({total_files / total_time:,.0f} files/s)\n")

    # --- Print results ---
    sorted_countries = sorted(merged.keys())

    hdr = f"{'Country':<6} {'Name':<20} {'Total':>10} {'Z_MU_KUNNR':>12} {'Z_PR_KUNNR':>12} {'Both':>10} {'Neither':>10} {'MU%':>6} {'PR%':>6}"
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    grand = {"total": 0, "has_mu": 0, "has_pr": 0, "has_both": 0, "has_neither": 0}

    for cc in sorted_countries:
        c = merged[cc]
        name = COUNTRY_CODE_MAP.get(cc, "Unknown")
        t = c["total"]
        mu_pct = (c["has_mu"] / t * 100) if t else 0
        pr_pct = (c["has_pr"] / t * 100) if t else 0
        print(f"{cc:<6} {name:<20} {t:>10,} {c['has_mu']:>12,} {c['has_pr']:>12,} {c['has_both']:>10,} {c['has_neither']:>10,} {mu_pct:>5.1f}% {pr_pct:>5.1f}%")
        for k in grand:
            grand[k] += c[k]

    print(sep)
    gt = grand["total"]
    mu_pct = (grand["has_mu"] / gt * 100) if gt else 0
    pr_pct = (grand["has_pr"] / gt * 100) if gt else 0
    print(f"{'ALL':<6} {'':<20} {gt:>10,} {grand['has_mu']:>12,} {grand['has_pr']:>12,} {grand['has_both']:>10,} {grand['has_neither']:>10,} {mu_pct:>5.1f}% {pr_pct:>5.1f}%")
    print(sep)


if __name__ == "__main__":
    main()
