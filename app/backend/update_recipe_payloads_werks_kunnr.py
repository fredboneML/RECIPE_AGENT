#!/usr/bin/env python3
"""
Update Qdrant recipe payloads with:
- country: from 'werks' field (first 2 chars → country name) instead of filename
- Z_MU_KUNNR: valueFrom from Classification.valuesnum or 'Missing'
- Z_PR_KUNNR: valueFrom from Classification.valuesnum or 'Missing'

Uses the same extraction logic as count_kunnr_stats.py. Run on the server after
indexing so filtering by country/KUNNR is correct and fast (create_payload_indexes
adds indexes for Z_MU_KUNNR and Z_PR_KUNNR).

Usage:
    python update_recipe_payloads_werks_kunnr.py [--data-dir /path/to/data] [--batch-size 500]

  Local (Mac/Linux, Qdrant in Docker): use default host localhost
    python update_recipe_payloads_werks_kunnr.py --data-dir app/data

  Server (Qdrant as service or same Docker network): use host qdrant
    python update_recipe_payloads_werks_kunnr.py --host qdrant --port 6333 --data-dir /datadrive/RECIPE_AGENT/app/data --batch-size 1000
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Same as count_kunnr_stats.py
COUNTRY_CODE_RE = re.compile(r'_([A-Z]{2})\d{2}_')
TARGET_CHARACTS = {"Z_MU_KUNNR", "Z_PR_KUNNR"}

COUNTRY_CODE_MAP = {
    "AT": "Austria", "AU": "Australia", "BE": "Belgium", "BG": "Bulgaria", "BR": "Brazil",
    "CH": "Switzerland", "CN": "China", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LT": "Lithuania", "LV": "Latvia", "MX": "Mexico", "NL": "Netherlands", "NO": "Norway",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "RS": "Serbia", "RU": "Russia",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "TR": "Turkey", "UA": "Ukraine",
    "US": "United States", "ZZ": "Custom / Unassigned",
}


def extract_country_from_filename(filename: str) -> str:
    m = COUNTRY_CODE_RE.search(filename)
    return m.group(1) if m else "XX"


def extract_country_from_werks(data: dict) -> str | None:
    """First 2 chars of first 'werks' in JSON (e.g. AU10 -> AU)."""
    found: list[str] = []

    def _find(obj):
        if isinstance(obj, dict):
            if "werks" in obj:
                val = obj.get("werks")
                if val and isinstance(val, str) and len(val) >= 2:
                    found.append(str(val)[:2].upper())
                    return
            for v in obj.values():
                _find(v)
                if found:
                    return
        elif isinstance(obj, list):
            for item in obj:
                _find(item)
                if found:
                    return

    _find(data)
    return found[0] if found else None


def extract_kunnr_values(data: dict) -> tuple[str, str]:
    """
    Extract Z_MU_KUNNR and Z_PR_KUNNR from Classification.valuesnum.
    Returns (value_mu, value_pr); each is str(valueFrom) or 'Missing' for filtering.
    """
    mu_val: str = "Missing"
    pr_val: str = "Missing"
    valuesnum = data.get("Classification", {}).get("valuesnum", [])
    for item in valuesnum:
        ch = item.get("charact")
        if ch not in TARGET_CHARACTS:
            continue
        v = item.get("valueFrom")
        if v is not None:
            try:
                v = str(int(v) if isinstance(v, (int, float)) else int(float(v)))
            except (TypeError, ValueError):
                v = str(v)
            if ch == "Z_MU_KUNNR":
                mu_val = v
            else:
                pr_val = v
        if mu_val != "Missing" and pr_val != "Missing":
            break
    return mu_val, pr_val


def get_country_name(code: str) -> str:
    """UI expects 'Other' for XX and unknown codes."""
    return COUNTRY_CODE_MAP.get(code, "Other")


def extract_payload_from_file(data_dir: Path, recipe_name: str) -> dict | None:
    """
    Read recipe JSON and return payload update dict: country, Z_MU_KUNNR, Z_PR_KUNNR.
    Returns None if file missing or unreadable.
    """
    path = data_dir / f"{recipe_name}.json"
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
    except Exception:
        return None
    code_werks = extract_country_from_werks(data)
    if code_werks:
        country = get_country_name(code_werks)
    else:
        code_fn = extract_country_from_filename(recipe_name)
        country = get_country_name(code_fn)
    mu_val, pr_val = extract_kunnr_values(data)
    return {"country": country, "Z_MU_KUNNR": mu_val, "Z_PR_KUNNR": pr_val}


def run(
    data_dir: Path,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "food_recipes_two_step",
    batch_size: int = 500,
    workers: int = 8,
    dry_run: bool = False,
):
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=300)
    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        logger.error("Collection not found: %s", e)
        return False

    total_points = info.points_count
    logger.info("Collection %s has %s points", collection_name, f"{total_points:,}")
    logger.info("Data dir: %s", data_dir)
    if dry_run:
        logger.info("DRY RUN - no payload updates will be applied")

    processed = 0
    updated = 0
    missing_file = 0
    errors = 0
    offset = None
    t0 = time.time()

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=["recipe_name"],
            with_vectors=False,
        )
        points, next_offset = result
        if not points:
            break

        recipe_names = [str(p.payload.get("recipe_name", "")) for p in points]
        point_ids = [p.id for p in points]

        # Load payloads from JSON files (parallel)
        updates = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(extract_payload_from_file, data_dir, name): (pid, name)
                for pid, name in zip(point_ids, recipe_names)
            }
            for future in as_completed(futures):
                point_id, name = futures[future]
                try:
                    payload = future.result()
                    if payload is not None:
                        updates.append((point_id, payload))
                    else:
                        missing_file += 1
                except Exception as e:
                    logger.debug("Error for %s: %s", name, e)
                    errors += 1

        processed += len(points)
        updated += len(updates)

        if updates and not dry_run:
            mini = 100
            for i in range(0, len(updates), mini):
                batch = updates[i : i + mini]
                for point_id, payload in batch:
                    try:
                        client.set_payload(
                            collection_name=collection_name,
                            payload=payload,
                            points=[point_id],
                            wait=False,
                        )
                    except Exception as e:
                        logger.warning("set_payload failed for %s: %s", point_id, e)
                        errors += 1

        if processed % 5000 == 0 or processed == total_points:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %s / %s (%.1f%%) | updated: %s | missing file: %s | %.0f/s",
                f"{processed:,}",
                f"{total_points:,}",
                100 * processed / total_points,
                updated,
                missing_file,
                rate,
            )

        offset = next_offset
        if offset is None:
            break

    elapsed = time.time() - t0
    logger.info(
        "Done. Processed: %s | Updated: %s | Missing file: %s | Errors: %s | %.1fs (%.0f/s)",
        f"{processed:,}",
        f"{updated:,}",
        missing_file,
        errors,
        elapsed,
        processed / elapsed if elapsed > 0 else 0,
    )
    if dry_run:
        logger.info("DRY RUN - no changes were written. Run without --dry-run to apply.")
    return True


def main():
    script_dir = Path(__file__).resolve().parent
    app_dir = script_dir.parent
    default_data = app_dir / "data"

    parser = argparse.ArgumentParser(description="Update Qdrant payloads: country from werks, Z_MU_KUNNR, Z_PR_KUNNR")
    parser.add_argument("--data-dir", type=Path, default=default_data, help="Path to recipe JSON data directory")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="food_recipes_two_step", help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=500, help="Scroll batch size")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for reading JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Do not write payloads")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        logger.error("Data directory not found: %s", args.data_dir)
        return 1

    ok = run(
        data_dir=args.data_dir,
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection,
        batch_size=args.batch_size,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
