#!/usr/bin/env python3
"""
Encounters/STS Transfers Enrichment Dataset Generator

Creates a dataset of ship-to-ship transfer events (encounters) from Global Fishing Watch.
These events indicate potential transshipment, smuggling, or sanctions evasion activities.

Output: enrichment_outputs/encounters_enrichment.csv
Columns:
  - mmsi: Link to simulation data
  - simulated_vessel_name: Name from our simulation
  - encounter_id: GFW encounter event ID
  - encounter_start: When encounter began
  - encounter_end: When encounter ended
  - duration_hours: How long vessels were in proximity
  - lat: Encounter latitude
  - lon: Encounter longitude
  - other_vessel_id: GFW ID of the other vessel
  - other_vessel_name: Name of the other vessel
  - other_vessel_mmsi: MMSI of the other vessel
  - other_vessel_flag: Flag of the other vessel
  - median_distance_km: Median distance between vessels during encounter
  - median_speed_knots: Median speed during encounter
  - event_timestamp: When event was recorded in GFW
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add GFW service to path
sys.path.append('/Users/zski/PycharmProjects/ElasticShipVision/global-fishing-api-service')

from src.client import GFWClient
from src.config import Config

async def enrich_encounters(identity_csv: str, output_csv: str, days_back: int = 365):
    """
    Generate encounters enrichment dataset.

    Args:
        identity_csv: Path to vessel identity enrichment CSV (must run enrich_vessel_identity.py first)
        output_csv: Output path for encounters CSV
        days_back: How many days back to fetch encounters (default: 365)
    """
    print("=" * 80)
    print("ENCOUNTERS/STS TRANSFERS ENRICHMENT")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load vessel identity data
    print(f"Loading vessel identity data from: {identity_csv}")
    try:
        identity_df = pd.read_csv(identity_csv)
    except FileNotFoundError:
        print(f"❌ ERROR: Identity file not found: {identity_csv}")
        print("   Run enrich_vessel_identity.py first!")
        return

    # Filter to only matched vessels (those found in GFW)
    matched_vessels = identity_df[identity_df['matched'] == True].copy()
    print(f"Found {len(matched_vessels)} matched vessels in GFW")
    print(f"Looking back {days_back} days for encounter events")
    print()

    # Initialize GFW client
    config = Config.from_env()
    if not config.gfw_api_token:
        print("❌ ERROR: GFW_API_TOKEN not found in environment")
        return

    client = GFWClient()

    # Fetch encounters for each vessel
    all_encounters = []
    vessels_with_encounters = 0
    total_encounters = 0
    start_time = datetime.now()

    for idx, row in matched_vessels.iterrows():
        vessel_id = row['gfw_vessel_id']
        vessel_name = row['simulated_vessel_name']
        mmsi = row['mmsi']

        # Progress with ETA
        elapsed = (datetime.now() - start_time).total_seconds()
        if idx > 0:
            avg_time = elapsed / (idx + 1)
            remaining = (len(matched_vessels) - idx - 1) * avg_time
            eta_mins = remaining / 60
            print(f"[{idx+1}/{len(matched_vessels)}] {vessel_name} [ETA: {eta_mins:.1f}m]...", end=" ")
        else:
            print(f"[{idx+1}/{len(matched_vessels)}] {vessel_name}...", end=" ")

        try:
            # Fetch encounter events
            encounters = await client.get_encounters(vessel_id, days_back=days_back)

            if encounters and len(encounters) > 0:
                vessels_with_encounters += 1
                total_encounters += len(encounters)

                for enc in encounters:
                    # Extract encounter data
                    all_encounters.append({
                        'mmsi': mmsi,
                        'simulated_vessel_name': vessel_name,
                        'encounter_id': enc.get('id'),
                        'encounter_start': enc.get('start'),
                        'encounter_end': enc.get('end'),
                        'duration_hours': (
                            (pd.to_datetime(enc.get('end')) - pd.to_datetime(enc.get('start'))).total_seconds() / 3600
                            if enc.get('start') and enc.get('end') else None
                        ),
                        'lat': enc.get('regions', {}).get('eez', [{}])[0].get('lat') if enc.get('regions') else enc.get('position', {}).get('lat'),
                        'lon': enc.get('regions', {}).get('eez', [{}])[0].get('lon') if enc.get('regions') else enc.get('position', {}).get('lon'),
                        'other_vessel_id': enc.get('vessel', {}).get('id'),
                        'other_vessel_name': enc.get('vessel', {}).get('name'),
                        'other_vessel_mmsi': enc.get('vessel', {}).get('ssvid'),
                        'other_vessel_flag': enc.get('vessel', {}).get('flag'),
                        'median_distance_km': enc.get('medianDistanceKm'),
                        'median_speed_knots': enc.get('medianSpeedKnots'),
                        'event_timestamp': datetime.utcnow().isoformat()
                    })

                print(f"✅ {len(encounters)} encounters")
            else:
                print("⭕ No encounters")

        except Exception as e:
            print(f"⚠️  Error: {str(e)[:50]}")

        # Rate limiting
        await asyncio.sleep(0.5)

    # Create DataFrame and save
    encounters_df = pd.DataFrame(all_encounters)

    if len(encounters_df) > 0:
        encounters_df.to_csv(output_csv, index=False)

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 80)
    print("ENCOUNTERS ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Vessels processed: {len(matched_vessels)}")
    print(f"Vessels with encounters: {vessels_with_encounters} ({vessels_with_encounters/len(matched_vessels)*100:.1f}%)")
    print(f"Total encounters: {total_encounters}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"✅ Output saved to: {output_csv}")
    print(f"   Dataset size: {len(encounters_df)} rows")
    print()

    # Show sample
    if len(encounters_df) > 0:
        print("Sample encounters:")
        print(encounters_df[['mmsi', 'simulated_vessel_name', 'encounter_start', 'other_vessel_name', 'duration_hours']].head(5).to_string(index=False))
        print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enrich encounter/STS events from GFW')
    parser.add_argument('--identity', type=str,
                       default='enrichment_outputs/vessel_identity_enrichment.csv',
                       help='Path to vessel identity CSV')
    parser.add_argument('--output', type=str,
                       default='enrichment_outputs/encounters_enrichment.csv',
                       help='Output CSV path')
    parser.add_argument('--days', type=int, default=365,
                       help='Days back to fetch events')

    args = parser.parse_args()

    asyncio.run(enrich_encounters(args.identity, args.output, args.days))
