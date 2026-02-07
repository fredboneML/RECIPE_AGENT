#!/usr/bin/env python3
"""
Test script to validate the data extractor agent's ability to extract
both numerical and categorical features from customer briefs.
"""

import requests
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/query"
LOGIN_URL = f"{BASE_URL}/api/login"

# Default test credentials (from environment)
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin321"


def get_auth_token() -> Optional[str]:
    """Get authentication token from the API."""
    try:
        response = requests.post(
            LOGIN_URL,
            json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("access_token")
        elif response.status_code == 400:
            # Local auth might be disabled, try without auth
            print("   ⚠️  Local auth disabled, attempting without authentication...")
            return None
        else:
            print(f"   ⚠️  Login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"   ⚠️  Could not get auth token: {e}")
        return None

# Expected results mapping: brief name -> expected recipe keywords in top results
# Recipe names from qdrant_recipes_export.json:
# - MATCHA SIGGIS (Austria)
# - TROPICAL 140G
# - GYROS OVCHEES (Germany) 
# - STRAWBERRY FIT
# - Cherry Vanilla Drink (Poland)
# - PASSIONFRUIT FIT
# - PEACH CHO POUCH
# - BLACK CHERRY MARVEL
# - APPLE MATCHA SIGGIS
# - LIME MATCHA SIGGIS
# - PFIRSICH APRIKOSE (Austria)
# - STRAWBERRY SPEZIAL
# - BANANA H FOB
# - BANANA FIT FLIP
# - BANANA FIT HERCULES
# - WITCHES CAULDRON (APPLE PEAR)

EXPECTED_RESULTS = {
    "brief_test_01_peach_apricot.txt": {
        "expected_recipes": ["PFIRSICH", "APRIKOSE", "PEACH"],  # Could match Austrian or other peach recipes
        "expected_countries": ["Austria", "Germany"],
    },
    "brief_test_02_tropical_halal.txt": {
        "expected_recipes": ["TROPICAL"],
        "expected_countries": [],
    },
    "brief_test_03_banana_high_fruit.txt": {
        "expected_recipes": ["BANANA"],
        "expected_countries": [],
    },
    "brief_test_04_strawberry_high_fruit.txt": {
        "expected_recipes": ["STRAWBERRY"],
        "expected_countries": [],
    },
    "brief_test_05_matcha_quark.txt": {
        "expected_recipes": ["MATCHA", "SIGGIS"],
        "expected_countries": ["Austria"],
    },
    "brief_test_06_cherry_drink.txt": {
        "expected_recipes": ["Cherry", "Vanilla", "Drink"],
        "expected_countries": ["Poland"],
    },
    "brief_test_07_passionfruit_fit.txt": {
        "expected_recipes": ["PASSIONFRUIT"],
        "expected_countries": [],
    },
    "brief_test_08_peach_pouch.txt": {
        "expected_recipes": ["PEACH"],
        "expected_countries": [],
    },
    "brief_test_09_apple_pear_seasonal.txt": {
        "expected_recipes": ["APPLE", "PEAR", "CAULDRON"],
        "expected_countries": [],
    },
    "brief_test_10_german_vegan.txt": {
        "expected_recipes": ["GYROS", "OVCHEES", "VEGAN"],
        "expected_countries": ["Germany"],
    },
}


def send_brief_to_api(brief_text: str, country_filter: List[str] = None, auth_token: str = None) -> Dict[str, Any]:
    """Send a brief to the API and return the response."""
    payload = {
        "query": brief_text,
        "country_filter": country_filter,
    }
    
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status_code": getattr(response, 'status_code', None), "response_text": getattr(response, 'text', '')[:500]}


def check_recipe_match(result_recipes: List[Dict], expected_keywords: List[str]) -> bool:
    """Check if any of the expected keywords appear in the result recipe names."""
    for recipe in result_recipes:
        recipe_name = recipe.get("recipe_name", "").upper()
        for keyword in expected_keywords:
            if keyword.upper() in recipe_name:
                return True
    return False


def analyze_extraction(response: Dict, expected: Dict, brief_name: str) -> Dict:
    """Analyze the extraction results against expectations."""
    analysis = {
        "brief_name": brief_name,
        "success": True,
        "issues": [],
        "extraction_details": {},
        "raw_response_keys": list(response.keys()) if isinstance(response, dict) else [],
    }
    
    if "error" in response:
        analysis["success"] = False
        analysis["issues"].append(f"API Error: {response['error']}")
        return analysis
    
    # Extract key information from response
    # The API returns 'search_results' not 'similar_recipes'
    search_results = response.get("search_results", [])
    extraction = response.get("extraction_result", {})
    
    # Store extraction details
    analysis["extraction_details"] = {
        "response_preview": response.get("response", "")[:300] + "..." if response.get("response") else "",
        "numerical_filters": extraction.get("numerical_filters", {}),
        "categorical_filters": extraction.get("categorical_filters", {}),
        "features": extraction.get("features", {}),
        "num_search_results": len(search_results),
        "top_recipes": [r.get("recipe_name", "Unknown") for r in search_results[:3]],
        "top_scores": [round(r.get("combined_score", r.get("text_score", 0)), 4) for r in search_results[:3]],
    }
    
    # Check if expected recipes are in top results
    if expected.get("expected_recipes"):
        if not check_recipe_match(search_results[:5], expected["expected_recipes"]):
            analysis["issues"].append(
                f"Expected recipes containing {expected['expected_recipes']} not found in top 5. "
                f"Got: {[r.get('recipe_name', 'Unknown')[:50] for r in search_results[:3]]}"
            )
    
    # The API doesn't currently return numerical/categorical filters in response
    # So we'll just check if results were returned
    if not search_results:
        analysis["issues"].append("No search results returned")
    
    if analysis["issues"]:
        analysis["success"] = False
    
    return analysis


def run_tests():
    """Run all test briefs and report results."""
    briefs_dir = Path(__file__).parent
    results = []
    
    print("=" * 80)
    print("RECIPE AGENT EXTRACTION TEST SUITE")
    print("=" * 80)
    print()
    
    # Get authentication token
    print("🔐 Authenticating...")
    auth_token = get_auth_token()
    if auth_token:
        print("   ✅ Authentication successful")
    else:
        print("   ⚠️  Running without authentication (may fail)")
    print()
    
    for brief_file, expected in EXPECTED_RESULTS.items():
        brief_path = briefs_dir / brief_file
        
        if not brief_path.exists():
            print(f"⚠️  Brief file not found: {brief_file}")
            continue
        
        print(f"\n{'─' * 80}")
        print(f"📄 Testing: {brief_file}")
        print(f"{'─' * 80}")
        
        # Read brief content
        with open(brief_path, 'r') as f:
            brief_text = f.read()
        
        # Send to API with country filter if specified
        country_filter = expected.get("expected_countries") if expected.get("expected_countries") else None
        print(f"   Sending to API (country filter: {country_filter})...")
        
        response = send_brief_to_api(brief_text, country_filter, auth_token)
        
        # Analyze results
        analysis = analyze_extraction(response, expected, brief_file)
        results.append(analysis)
        
        # Print results
        if analysis["success"]:
            print(f"   ✅ PASS")
        else:
            print(f"   ❌ FAIL")
            for issue in analysis["issues"]:
                print(f"      - {issue}")
        
        # Print extraction details
        details = analysis["extraction_details"]
        if details:
            print(f"\n   📊 Extraction Details:")
            print(f"      Response keys: {analysis.get('raw_response_keys', [])}")
            if details.get('response_preview'):
                print(f"      Response preview: {details['response_preview'][:200]}...")
            print(f"      Num results: {details.get('num_search_results', 0)}")
            print(f"      Top 3 recipes: {details.get('top_recipes', [])}")
            print(f"      Top 3 scores: {details.get('top_scores', [])}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"\n   Total tests: {len(results)}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   Success rate: {passed/len(results)*100:.1f}%")
    
    if failed > 0:
        print("\n   Failed tests:")
        for r in results:
            if not r["success"]:
                print(f"      - {r['brief_name']}")
    
    # Save detailed results to JSON
    output_path = briefs_dir / "test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Detailed results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_tests()
