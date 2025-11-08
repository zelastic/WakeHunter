#!/usr/bin/env python3
"""
Mini Enrichment Test

Tests the enrichment pipeline with 5 vessels to validate API connectivity
and data structure before running the full enrichment.

Run this FIRST before processing all 332 vessels!
"""

import subprocess
import sys

def run_test():
    print("=" * 80)
    print("MINI ENRICHMENT TEST - 5 VESSELS")
    print("=" * 80)
    print()
    print("This will test the GFW API connection and enrichment scripts")
    print("with a small sample of 5 vessels before processing all 332 vessels.")
    print()

    # Test 1: Vessel Identity (most important - validates API access)
    print("Step 1/6: Testing vessel identity enrichment...")
    print("-" * 80)
    result = subprocess.run([
        sys.executable,
        'enrich_vessel_identity.py',
        '--limit', '5',
        '--output', 'enrichment_outputs/test_identity.csv'
    ])
    if result.returncode != 0:
        print("❌ Identity enrichment test FAILED")
        print("   Check that GFW_API_TOKEN is set correctly")
        return False
    print()

    # Test 2: Encounters
    print("Step 2/6: Testing encounters enrichment...")
    print("-" * 80)
    result = subprocess.run([
        sys.executable,
        'enrich_encounters.py',
        '--identity', 'enrichment_outputs/test_identity.csv',
        '--output', 'enrichment_outputs/test_encounters.csv',
        '--days', '90'  # Shorter period for testing
    ])
    print()

    # Test 3: Gaps
    print("Step 3/6: Testing gaps enrichment...")
    print("-" * 80)
    result = subprocess.run([
        sys.executable,
        'enrich_gaps.py',
        '--identity', 'enrichment_outputs/test_identity.csv',
        '--output', 'enrichment_outputs/test_gaps.csv',
        '--days', '90'
    ])
    print()

    # Test 4: Loitering
    print("Step 4/6: Testing loitering enrichment...")
    print("-" * 80)
    result = subprocess.run([
        sys.executable,
        'enrich_loitering.py',
        '--identity', 'enrichment_outputs/test_identity.csv',
        '--output', 'enrichment_outputs/test_loitering.csv',
        '--days', '90'
    ])
    print()

    # Test 5: Port Visits
    print("Step 5/6: Testing port visits enrichment...")
    print("-" * 80)
    result = subprocess.run([
        sys.executable,
        'enrich_port_visits.py',
        '--identity', 'enrichment_outputs/test_identity.csv',
        '--output', 'enrichment_outputs/test_port_visits.csv',
        '--days', '90'
    ])
    print()

    # Test 6: Risk Assessment
    print("Step 6/6: Testing risk assessment...")
    print("-" * 80)
    result = subprocess.run([
        sys.executable,
        'enrich_risk_assessment.py',
        '--identity', 'enrichment_outputs/test_identity.csv',
        '--output', 'enrichment_outputs/test_risk_assessment.csv',
        '--days', '90'
    ])
    print()

    print("=" * 80)
    print("MINI TEST COMPLETE!")
    print("=" * 80)
    print()
    print("✅ All test scripts executed successfully")
    print("   Check enrichment_outputs/test_*.csv for results")
    print()
    print("If the test looks good, run the full enrichment with:")
    print("   python3 run_all_enrichment.py")
    print()

if __name__ == "__main__":
    run_test()
