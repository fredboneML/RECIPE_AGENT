#!/usr/bin/env python3
"""
Generate constraint-based test briefs from qdrant_recipes_export.json,
send them to the local API, and verify expected recipes appear in top results.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/query"
LOGIN_URL = f"{BASE_URL}/api/login"

# Default test credentials (from environment)
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin321"

EXPORT_PATH = Path(__file__).resolve().parents[2] / "qdrant_recipes_export.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_constraint_briefs"


def get_auth_token() -> Optional[str]:
    """Get authentication token from the API."""
    try:
        response = requests.post(
            LOGIN_URL,
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("access_token")
        if response.status_code == 400:
            print("   ⚠️  Local auth disabled, attempting without authentication...")
            return None
        print(f"   ⚠️  Login failed: {response.status_code} - {response.text}")
        return None
    except Exception as exc:
        print(f"   ⚠️  Could not get auth token: {exc}")
        return None


def send_brief_to_api(brief_text: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
    """Send a brief to the API and return the response."""
    payload = {"query": brief_text, "country_filter": None, "version_filter": None}
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    response = None
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        return {
            "error": str(exc),
            "status_code": getattr(response, "status_code", None),
            "response_text": getattr(response, "text", "")[:500],
        }


def normalize_percent(value: Any) -> Optional[float]:
    """Extract a numeric percent value from strings like '10.0 %' or '0,5 %'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def format_percent(value: float) -> str:
    """Format numeric percent without trailing .0 where possible."""
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%"


def extract_material_short_text(payload: Dict[str, Any]) -> str:
    """Best-effort extraction of MaterialMasterShorttext."""
    description = payload.get("description", "") or ""
    if "MaterialMasterShorttext:" in description:
        return description.split("MaterialMasterShorttext:")[1].split(",")[0].strip()
    spec_fields = payload.get("spec_fields", {}) or {}
    return spec_fields.get("Z_MAKTX") or payload.get("recipe_name", "Unknown")


def build_brief(payload: Dict[str, Any]) -> Tuple[str, int]:
    """Build a constraint-based brief following the server pattern."""
    spec = payload.get("spec_fields", {}) or {}
    numerical = payload.get("numerical", {}) or {}

    lines: List[str] = [extract_material_short_text(payload)]

    # Categorical constraints (German phrasing used by parser)
    if spec.get("Z_INH02") == "No":
        lines.append("Kein Süßstoff")
    if spec.get("Z_INH03") in ("Saccharose", "Saccarose", "Yes"):
        lines.append("Saccharose")
    if spec.get("Z_INH04") == "No":
        lines.append("Nicht konserviert")
    if spec.get("Z_INH12") == "Yes":
        lines.append("Allergenfrei")
    if spec.get("Z_INH05") == "No":
        lines.append("Kein künstliches Farbe")
    if isinstance(spec.get("Z_INH06"), str) and spec.get("Z_INH06", "").lower().startswith("naturident"):
        lines.append("Naturidentes Aroma")
    if spec.get("Z_INH06Z") == "No":
        lines.append("Kein natürliches Aroma")
    if spec.get("Z_INH13") == "Yes":
        lines.append("Stärke enthalten")
    if spec.get("Z_INH14") == "Yes":
        lines.append("Pektin enthalten")
    if isinstance(spec.get("Z_INH17"), str) and spec.get("Z_INH17", "").lower().startswith("andere"):
        lines.append("Andere Stabilisatoren")
    if spec.get("Z_INH17") == "Other stabilizer":
        lines.append("Andere Stabilisatoren")

    # Numerical constraints
    dosage = normalize_percent(numerical.get("Z_DOSIER") or spec.get("Z_DOSIER"))
    if dosage is not None:
        lines.append(f"Dosage {format_percent(dosage)}")

    fruit = normalize_percent(numerical.get("Z_FRUCHTG") or spec.get("Z_FRUCHTG"))
    if fruit is not None:
        lines.append(f"{format_percent(fruit)} Frucht")

    constraint_count = max(0, len(lines) - 1)
    return "\n".join(lines), constraint_count


def select_test_briefs(points: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """Select recipes with sufficient constraints to build test briefs."""
    tests: List[Dict[str, Any]] = []
    for point in points:
        payload = point.get("payload", {})
        brief, constraint_count = build_brief(payload)
        if constraint_count < 6:
            continue
        tests.append(
            {
                "recipe_name": payload.get("recipe_name", "Unknown"),
                "brief": brief,
                "constraint_count": constraint_count,
            }
        )
        if len(tests) >= limit:
            break

    # If not enough, relax the constraint threshold
    if len(tests) < limit:
        for point in points:
            payload = point.get("payload", {})
            brief, constraint_count = build_brief(payload)
            if constraint_count < 4:
                continue
            if any(t["recipe_name"] == payload.get("recipe_name") for t in tests):
                continue
            tests.append(
                {
                    "recipe_name": payload.get("recipe_name", "Unknown"),
                    "brief": brief,
                    "constraint_count": constraint_count,
                }
            )
            if len(tests) >= limit:
                break
    return tests[:limit]


def check_recipe_match(search_results: List[Dict[str, Any]], expected_recipe: str) -> bool:
    """Check if expected recipe appears in top results."""
    expected_upper = (expected_recipe or "").upper()
    for recipe in search_results:
        recipe_name = recipe.get("recipe_name", "").upper()
        if expected_upper and expected_upper in recipe_name:
            return True
    return False


def run_tests() -> None:
    if not EXPORT_PATH.exists():
        raise FileNotFoundError(f"Export file not found at {EXPORT_PATH}")

    with EXPORT_PATH.open("r", encoding="utf-8") as handle:
        export_data = json.load(handle)

    points = export_data.get("result", {}).get("points", [])
    if not points:
        raise ValueError("No points found in export data.")

    tests = select_test_briefs(points, limit=10)
    if not tests:
        raise ValueError("No valid test briefs could be generated.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, test in enumerate(tests, 1):
        brief_path = OUTPUT_DIR / f"generated_brief_{idx:02d}.txt"
        brief_path.write_text(test["brief"], encoding="utf-8")

    print("=" * 80)
    print("GENERATED CONSTRAINT TEST SUITE")
    print("=" * 80)
    print(f"Generated {len(tests)} briefs in {OUTPUT_DIR}")
    print()

    print("🔐 Authenticating...")
    auth_token = get_auth_token()
    if auth_token:
        print("   ✅ Authentication successful")
    else:
        print("   ⚠️  Running without authentication (may fail)")
    print()

    passed = 0
    for idx, test in enumerate(tests, 1):
        print(f"\n{'─' * 80}")
        print(f"📄 Test {idx:02d}: {test['recipe_name']}")
        print(f"{'─' * 80}")
        print(f"   Constraints: {test['constraint_count']}")
        print("   Sending to API...")

        response = send_brief_to_api(test["brief"], auth_token)
        if "error" in response:
            print(f"   ❌ API error: {response['error']}")
            continue

        search_results = response.get("search_results", [])
        top_recipes = [r.get("recipe_name", "Unknown") for r in search_results[:3]]
        top_scores = [round(r.get("combined_score", r.get("text_score", 0)), 4) for r in search_results[:3]]

        matched = check_recipe_match(search_results[:5], test["recipe_name"])
        if matched:
            passed += 1
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL")
        print(f"   Top 3 recipes: {top_recipes}")
        print(f"   Top 3 scores:  {top_scores}")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"   Total tests: {len(tests)}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {len(tests) - passed}")


if __name__ == "__main__":
    run_tests()
