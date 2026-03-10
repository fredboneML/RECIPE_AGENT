#!/usr/bin/env python3
"""
Generate plots from kunnr_stats.csv and save them in app/plots/.

Usage:
    python plot_kunnr_stats.py [--csv path/to/kunnr_stats.csv]

Defaults:
    --csv  ../kunnr_stats.csv  (relative to this script → app/kunnr_stats.csv)
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = APP_DIR / "kunnr_stats.csv"
PLOTS_DIR = APP_DIR / "plots"


def load_sections(csv_path: Path):
    """Parse CSV into three DataFrames: detailed, by_country, by_version."""
    with open(csv_path) as f:
        lines = f.readlines()

    detailed_rows = []
    country_rows = []
    version_rows = []
    current = None
    header = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = "detailed" if "DETAILED" in line else "country" if "BY COUNTRY" in line else "version" if "BY VERSION" in line else None
            continue
        parts = [p.strip('"') for p in line.split(",")]
        if parts[0] == "Country" and parts[1] == "Name":
            header = parts
            continue
        if header is None:
            continue
        row = dict(zip(header, parts))
        for k in ["Total", "Z_MU_KUNNR", "Z_PR_KUNNR", "Both", "Neither", "MU%", "PR%"]:
            if k in row:
                try:
                    row[k] = float(row[k]) if "." in str(row[k]) else int(row[k])
                except (ValueError, TypeError):
                    row[k] = 0

        if current == "detailed":
            detailed_rows.append(row)
        elif current == "country":
            country_rows.append(row)
        elif current == "version":
            version_rows.append(row)

    detailed = pd.DataFrame(detailed_rows) if detailed_rows else pd.DataFrame()
    by_country = pd.DataFrame(country_rows) if country_rows else pd.DataFrame()
    by_version = pd.DataFrame(version_rows) if version_rows else pd.DataFrame()
    return detailed, by_country, by_version


def main():
    parser = argparse.ArgumentParser(description="Plot KUNNR statistics from CSV")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to kunnr_stats.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {csv_path} ...")
    detailed, by_country, by_version = load_sections(csv_path)

    # ---- Export Excel (same data as CSV, three sheets) ----
    excel_path = csv_path.parent / "kunnr_stats.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        detailed.to_excel(writer, sheet_name="Detailed", index=False)
        by_country.to_excel(writer, sheet_name="By Country", index=False)
        by_version.to_excel(writer, sheet_name="By Version", index=False)
    print(f"Excel saved to: {excel_path}")

    # Exclude ALL row for per-country plots; keep only real countries
    df_country = by_country[by_country["Country"] != "ALL"].copy()
    df_country = df_country.sort_values("Total", ascending=True)

    # Country axis label: "CC Name" or just "CC" when name is Unknown (no "?")
    labels = (df_country["Country"] + " " + df_country["Name"]).where(
        df_country["Name"] != "Unknown", df_country["Country"]
    )

    # ---- 1. Total recipes per country (horizontal bar) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, df_country["Total"], color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_xlabel("Number of recipes")
    ax.set_title("Total recipes per country")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "01_total_per_country.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ---- 2. Z_MU_KUNNR vs Z_PR_KUNNR per country (grouped bars) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df_country))
    w = 0.35
    ax.bar([i - w / 2 for i in x], df_country["Z_MU_KUNNR"], w, label="Z_MU_KUNNR", color="coral", alpha=0.9)
    ax.bar([i + w / 2 for i in x], df_country["Z_PR_KUNNR"], w, label="Z_PR_KUNNR", color="seagreen", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Z_MU_KUNNR vs Z_PR_KUNNR per country")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "02_mu_vs_pr_per_country.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ---- 3. MU% and PR% per country (grouped bars) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - w / 2 for i in x], df_country["MU%"], w, label="MU%", color="coral", alpha=0.9)
    ax.bar([i + w / 2 for i in x], df_country["PR%"], w, label="PR%", color="seagreen", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Share of recipes with Z_MU_KUNNR / Z_PR_KUNNR per country")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "03_mu_pr_pct_per_country.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ---- 4. Version distribution (pie) ----
    df_ver = by_version[by_version["Country"] == "ALL"].copy()
    df_ver = df_ver[df_ver["Version"].isin(["L", "P", "Missing"])]
    if not df_ver.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ["#2ecc71", "#3498db", "#95a5a6"]
        ax.pie(
            df_ver["Total"],
            labels=df_ver["Version"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors[: len(df_ver)],
        )
        ax.set_title("Recipe count by version (L / P / Missing)")
        plt.tight_layout()
        out = PLOTS_DIR / "04_version_pie.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out.name}")

    # ---- 5. KUNNR coverage: Both / MU only / PR only / Neither (stacked bar) ----
    left_neither = df_country["Neither"].values
    left_mu_only = (df_country["Z_MU_KUNNR"] - df_country["Both"]).values
    left_pr_only = (df_country["Z_PR_KUNNR"] - df_country["Both"]).values
    both_vals = df_country["Both"].values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, left_neither, label="Neither", color="lightgray", alpha=0.9)
    ax.barh(labels, both_vals, left=left_neither, label="Both", color="darkgreen", alpha=0.85)
    ax.barh(labels, left_mu_only, left=left_neither + both_vals, label="MU only", color="coral", alpha=0.8)
    ax.barh(labels, left_pr_only, left=left_neither + both_vals + left_mu_only, label="PR only", color="seagreen", alpha=0.8)
    ax.set_xlabel("Number of recipes")
    ax.set_title("KUNNR coverage: Both / Z_MU only / Z_PR only / Neither")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOTS_DIR / "05_has_kunnr_vs_neither.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ---- 6. Version breakdown per country (stacked: L, P, Missing) from detailed ----
    if not detailed.empty:
        detail = detailed[detailed["Version"].isin(["L", "P", "Missing"])]
        pivot = detail.pivot_table(index=["Country", "Name"], columns="Version", values="Total", aggfunc="sum", fill_value=0).reset_index()
        for v in ["L", "P", "Missing"]:
            if v not in pivot.columns:
                pivot[v] = 0
        pivot["_total"] = pivot["L"] + pivot["P"] + pivot["Missing"]
        pivot = pivot.merge(df_country[["Country", "Name"]].drop_duplicates(), on=["Country", "Name"], how="right")
        pivot = pivot.fillna(0)
        pivot["_total"] = pivot["L"] + pivot["P"] + pivot["Missing"]
        pivot = pivot.sort_values("_total", ascending=True)
        labels_v = (pivot["Country"] + " " + pivot["Name"]).where(
            pivot["Name"] != "Unknown", pivot["Country"]
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels_v, pivot["L"], label="L", color="#2ecc71", alpha=0.9)
        ax.barh(labels_v, pivot["P"], left=pivot["L"], label="P", color="#3498db", alpha=0.9)
        ax.barh(labels_v, pivot["Missing"], left=pivot["L"] + pivot["P"], label="Missing", color="#95a5a6", alpha=0.9)
        ax.set_xlabel("Number of recipes")
        ax.set_title("Version breakdown (L / P / Missing) per country")
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = PLOTS_DIR / "06_version_per_country.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out.name}")

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
