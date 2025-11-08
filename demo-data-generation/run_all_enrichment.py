#!/usr/bin/env python3
"""
Master Enrichment Orchestrator

Runs all 6 enrichment scripts in sequence to create complete enriched datasets
linking simulated vessels to real Global Fishing Watch data.

Estimated time: ~60 minutes for 332 vessels
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_enrichment():
    print("=" * 80)
    print("GLOBAL FISHING WATCH ENRICHMENT PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create output directory
    Path('enrichment_outputs').mkdir(exist_ok=True)

    start_time = datetime.now()

    # Step 1: Vessel Identity (Foundation - must run first)
    print("\n" + "=" * 80)
    print("STEP 1/6: VESSEL IDENTITY ENRICHMENT")
    print("=" * 80)
    print("Looking up all vessels in Global Fishing Watch database...")
    print()
    
    result = subprocess.run([
        sys.executable,
        'enrich_vessel_identity.py',
        '--output', 'enrichment_outputs/vessel_identity_enrichment.csv'
    ])
    
    if result.returncode != 0:
        print("\n❌ ERROR: Vessel identity enrichment failed!")
        print("   Cannot continue without identity data.")
        print("   Check GFW_API_TOKEN and try again.")
        return

    # Step 2: Encounters
    print("\n" + "=" * 80)
    print("STEP 2/6: ENCOUNTERS/STS TRANSFERS ENRICHMENT")
    print("=" * 80)
    print("Fetching ship-to-ship transfer events...")
    print()
    
    subprocess.run([
        sys.executable,
        'enrich_encounters.py',
        '--identity', 'enrichment_outputs/vessel_identity_enrichment.csv',
        '--output', 'enrichment_outputs/encounters_enrichment.csv'
    ])

    # Step 3: Gaps
    print("\n" + "=" * 80)
    print("STEP 3/6: DARK PERIODS/GAPS ENRICHMENT")
    print("=" * 80)
    print("Fetching AIS gap events (dark periods)...")
    print()
    
    subprocess.run([
        sys.executable,
        'enrich_gaps.py',
        '--identity', 'enrichment_outputs/vessel_identity_enrichment.csv',
        '--output', 'enrichment_outputs/gaps_enrichment.csv'
    ])

    # Step 4: Loitering
    print("\n" + "=" * 80)
    print("STEP 4/6: LOITERING EVENTS ENRICHMENT")
    print("=" * 80)
    print("Fetching loitering events...")
    print()
    
    subprocess.run([
        sys.executable,
        'enrich_loitering.py',
        '--identity', 'enrichment_outputs/vessel_identity_enrichment.csv',
        '--output', 'enrichment_outputs/loitering_enrichment.csv'
    ])

    # Step 5: Port Visits
    print("\n" + "=" * 80)
    print("STEP 5/6: PORT VISITS ENRICHMENT")
    print("=" * 80)
    print("Fetching port visit events...")
    print()
    
    subprocess.run([
        sys.executable,
        'enrich_port_visits.py',
        '--identity', 'enrichment_outputs/vessel_identity_enrichment.csv',
        '--output', 'enrichment_outputs/port_visits_enrichment.csv'
    ])

    # Step 6: Risk Assessment
    print("\n" + "=" * 80)
    print("STEP 6/6: RISK ASSESSMENT ENRICHMENT")
    print("=" * 80)
    print("Calculating risk scores for all vessels...")
    print()
    
    subprocess.run([
        sys.executable,
        'enrich_risk_assessment.py',
        '--identity', 'enrichment_outputs/vessel_identity_enrichment.csv',
        '--output', 'enrichment_outputs/risk_assessment_enrichment.csv'
    ])

    total_time = (datetime.now() - start_time).total_seconds()

    # Final summary
    print("\n" + "=" * 80)
    print("ENRICHMENT PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_time/60:.1f} minutes")
    print()
    print("Enriched datasets created:")
    print("  1. vessel_identity_enrichment.csv")
    print("  2. encounters_enrichment.csv")
    print("  3. gaps_enrichment.csv")
    print("  4. loitering_enrichment.csv")
    print("  5. port_visits_enrichment.csv")
    print("  6. risk_assessment_enrichment.csv")
    print()
    print("All datasets are linked by MMSI to your simulation data.")
    print("=" * 80)

if __name__ == "__main__":
    import os
    
    # Check for API token
    if not os.getenv('GFW_API_TOKEN'):
        print("❌ ERROR: GFW_API_TOKEN not found in environment")
        print()
        print("Set it with:")
        print("  export GFW_API_TOKEN='your_token'")
        print()
        print("Get a token from: https://globalfishingwatch.org/our-apis/tokens")
        sys.exit(1)
    
    print("✅ GFW_API_TOKEN found")
    print()
    
    # Ask for confirmation
    print("This will process all 332 vessels and take approximately 60 minutes.")
    print("Make sure you've run test_enrichment_mini.py first!")
    print()
    response = input("Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        run_enrichment()
    else:
        print("Cancelled.")
