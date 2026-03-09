#!/usr/bin/env python3
"""
Count non-null Z_MU_KUNNR and Z_PR_KUNNR values per country and per version (L/P/Missing).

Version is extracted from the JSON content using the same stlan extraction logic
as init_vector_index_qdrant.py (recursive search, P > L > Missing priority).

Usage:
    python count_kunnr_stats.py [--data-dir /path/to/data] [--workers N]

Defaults:
    --data-dir  ../data  (relative to this script)
    --workers   number of CPU cores
"""

import argparse
import csv
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

EMPTY_ROW = {"total": 0, "has_mu": 0, "has_pr": 0, "has_both": 0, "has_neither": 0}


def extract_country(filename: str) -> str:
    m = COUNTRY_CODE_RE.search(filename)
    return m.group(1) if m else "XX"


def extract_stlan(data: dict) -> str:
    """
    Extract stlan value from parsed recipe JSON.
    Same logic as init_vector_index_qdrant.py: recursive search, priority P > L > Missing.
    """
    found = set()

    def _find(obj):
        if isinstance(obj, dict):
            val = obj.get("stlan")
            if val in ("L", "P"):
                found.add(val)
                if len(found) == 2:
                    return
            for v in obj.values():
                _find(v)
                if len(found) == 2:
                    return
        elif isinstance(obj, list):
            for item in obj:
                _find(item)
                if len(found) == 2:
                    return

    _find(data)

    if "P" in found:
        return "P"
    if "L" in found:
        return "L"
    return "Missing"


def new_row():
    return EMPTY_ROW.copy()


def process_batch(file_paths: list[str]) -> dict:
    """Process a batch of files. Returns {(country, version): counts}."""
    counts: dict[tuple[str, str], dict] = defaultdict(new_row)

    for fpath in file_paths:
        fname = os.path.basename(fpath)
        country = extract_country(fname)

        found_characts = set()
        version = "Missing"

        try:
            with open(fpath, "rb") as f:
                data = json.loads(f.read())

            version = extract_stlan(data)

            valuesnum = data.get("Classification", {}).get("valuesnum", [])
            for item in valuesnum:
                ch = item.get("charact")
                if ch in TARGET_CHARACTS and item.get("valueFrom") is not None:
                    found_characts.add(ch)
                    if len(found_characts) == 2:
                        break
        except Exception:
            pass

        key = (country, version)
        counts[key]["total"] += 1

        has_mu = "Z_MU_KUNNR" in found_characts
        has_pr = "Z_PR_KUNNR" in found_characts

        if has_mu:
            counts[key]["has_mu"] += 1
        if has_pr:
            counts[key]["has_pr"] += 1
        if has_mu and has_pr:
            counts[key]["has_both"] += 1
        if not has_mu and not has_pr:
            counts[key]["has_neither"] += 1

    return {k: dict(v) for k, v in counts.items()}


def merge_counts(target: dict, source: dict):
    for key, vals in source.items():
        if key not in target:
            target[key] = new_row()
        for k in vals:
            target[key][k] += vals[k]


def add_rows(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in EMPTY_ROW}


def fmt_row(label: str, name: str, ver: str, c: dict) -> str:
    t = c["total"]
    mu_pct = (c["has_mu"] / t * 100) if t else 0
    pr_pct = (c["has_pr"] / t * 100) if t else 0
    return (
        f"{label:<6} {name:<20} {ver:<8} "
        f"{t:>10,} {c['has_mu']:>12,} {c['has_pr']:>12,} "
        f"{c['has_both']:>10,} {c['has_neither']:>10,} "
        f"{mu_pct:>5.1f}% {pr_pct:>5.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Count Z_MU_KUNNR / Z_PR_KUNNR stats per country & version")
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
    merged: dict[tuple[str, str], dict] = {}
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

    # --- Collect unique countries and versions ---
    countries = sorted({k[0] for k in merged})
    versions = sorted({k[1] for k in merged})

    hdr = (
        f"{'Country':<6} {'Name':<20} {'Version':<8} "
        f"{'Total':>10} {'Z_MU_KUNNR':>12} {'Z_PR_KUNNR':>12} "
        f"{'Both':>10} {'Neither':>10} {'MU%':>6} {'PR%':>6}"
    )
    sep = "=" * len(hdr)
    thin = "-" * len(hdr)

    # ========== TABLE 1: Per country + version breakdown ==========
    print(sep)
    print("  DETAILED BREAKDOWN (per country, per version)")
    print(sep)
    print(hdr)
    print(sep)

    grand = new_row()

    for cc in countries:
        name = COUNTRY_CODE_MAP.get(cc, "Unknown")
        country_total = new_row()

        for ver in versions:
            key = (cc, ver)
            if key not in merged:
                continue
            c = merged[key]
            country_total = add_rows(country_total, c)
            print(fmt_row(cc, name, ver, c))

        if country_total["total"] > 0:
            print(fmt_row(cc, name, "ALL", country_total))
            print(thin)

        grand = add_rows(grand, country_total)

    print(fmt_row("ALL", "", "ALL", grand))
    print(sep)

    # ========== TABLE 2: Summary by country only ==========
    print(f"\n{sep}")
    print("  SUMMARY BY COUNTRY")
    print(sep)
    print(hdr)
    print(sep)

    grand = new_row()
    for cc in countries:
        name = COUNTRY_CODE_MAP.get(cc, "Unknown")
        country_total = new_row()
        for ver in versions:
            key = (cc, ver)
            if key in merged:
                country_total = add_rows(country_total, merged[key])
        print(fmt_row(cc, name, "", country_total))
        grand = add_rows(grand, country_total)

    print(sep)
    print(fmt_row("ALL", "", "", grand))
    print(sep)

    # ========== TABLE 3: Summary by version only ==========
    print(f"\n{sep}")
    print("  SUMMARY BY VERSION")
    print(sep)
    print(hdr)
    print(sep)

    grand = new_row()
    for ver in versions:
        ver_total = new_row()
        for cc in countries:
            key = (cc, ver)
            if key in merged:
                ver_total = add_rows(ver_total, merged[key])
        print(fmt_row("ALL", "", ver, ver_total))
        grand = add_rows(grand, ver_total)

    print(sep)
    print(fmt_row("ALL", "", "ALL", grand))
    print(sep)

    # ========== Export CSV ==========
    app_dir = Path(__file__).resolve().parent.parent
    csv_path = app_dir / "kunnr_stats.csv"

    CSV_HEADER = ["Country", "Name", "Version", "Total", "Z_MU_KUNNR", "Z_PR_KUNNR", "Both", "Neither", "MU%", "PR%"]

    def csv_row(cc: str, name: str, ver: str, c: dict) -> list:
        t = c["total"]
        mu_pct = round(c["has_mu"] / t * 100, 1) if t else 0.0
        pr_pct = round(c["has_pr"] / t * 100, 1) if t else 0.0
        return [cc, name, ver, t, c["has_mu"], c["has_pr"], c["has_both"], c["has_neither"], mu_pct, pr_pct]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Section 1: Detailed breakdown
        writer.writerow(["# DETAILED BREAKDOWN (per country, per version)"])
        writer.writerow(CSV_HEADER)
        grand = new_row()
        for cc in countries:
            name = COUNTRY_CODE_MAP.get(cc, "Unknown")
            country_total = new_row()
            for ver in versions:
                key = (cc, ver)
                if key not in merged:
                    continue
                c = merged[key]
                country_total = add_rows(country_total, c)
                writer.writerow(csv_row(cc, name, ver, c))
            if country_total["total"] > 0:
                writer.writerow(csv_row(cc, name, "ALL", country_total))
            grand = add_rows(grand, country_total)
        writer.writerow(csv_row("ALL", "", "ALL", grand))
        writer.writerow([])

        # Section 2: Summary by country
        writer.writerow(["# SUMMARY BY COUNTRY"])
        writer.writerow(CSV_HEADER)
        grand = new_row()
        for cc in countries:
            name = COUNTRY_CODE_MAP.get(cc, "Unknown")
            country_total = new_row()
            for ver in versions:
                key = (cc, ver)
                if key in merged:
                    country_total = add_rows(country_total, merged[key])
            writer.writerow(csv_row(cc, name, "", country_total))
            grand = add_rows(grand, country_total)
        writer.writerow(csv_row("ALL", "", "", grand))
        writer.writerow([])

        # Section 3: Summary by version
        writer.writerow(["# SUMMARY BY VERSION"])
        writer.writerow(CSV_HEADER)
        grand = new_row()
        for ver in versions:
            ver_total = new_row()
            for cc in countries:
                key = (cc, ver)
                if key in merged:
                    ver_total = add_rows(ver_total, merged[key])
            writer.writerow(csv_row("ALL", "", ver, ver_total))
            grand = add_rows(grand, ver_total)
        writer.writerow(csv_row("ALL", "", "ALL", grand))

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
