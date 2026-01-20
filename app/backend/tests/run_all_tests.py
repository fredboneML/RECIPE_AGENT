#!/usr/bin/env python3
"""
Test Runner for Recipe Agent System

Runs all tests and generates a summary report.
Usage: python run_all_tests.py [--verbose] [--html-report]
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

def run_tests(verbose=False, html_report=False):
    """Run all tests and return results"""
    
    # Test files to run
    test_files = [
        "test_edge_cases.py",
        "test_agent_integration.py",
    ]
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append("--tb=short")
    
    if html_report:
        report_name = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        cmd.extend(["--html", os.path.join(script_dir, report_name)])
    
    # Add test files
    for test_file in test_files:
        test_path = os.path.join(script_dir, test_file)
        if os.path.exists(test_path):
            cmd.append(test_path)
        else:
            print(f"Warning: Test file not found: {test_path}")
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Recipe Agent tests")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--html-report", action="store_true",
                       help="Generate HTML report")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RECIPE AGENT TEST SUITE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    exit_code = run_tests(verbose=args.verbose, html_report=args.html_report)
    
    print()
    print("=" * 60)
    if exit_code == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
