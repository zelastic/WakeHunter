#!/usr/bin/env python3
"""
Dark Periods/Gaps Enrichment Dataset Generator

Creates a dataset of AIS gap events (dark periods) from Global Fishing Watch.
These events indicate when vessels turned off their AIS transponders, often used
for illegal fishing or sanctions evasion.

Output: enrichment_outputs/gaps_enrichment.csv
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append('/Users/zski/PycharmProjects/ElasticShipVision/global-fishing-api-service')

from src.client import GFWClient
from src.config import Config

async def enrich_gaps(identity_csv: str, output_csv: str, days_back: int = 365):
    print("=" * 80)
    print("DARK PERIODS/GAPS ENRICHMENT")
    print("=" * 80)
    print()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading vessel identity data from: {identity_csv}")
    try:
        identity_df = pd.read_csv(identity_csv)
    except FileNotFoundError:
        print(f"❌ ERROR: Identity file not found: {identity_csv}")
        return

    matched_vessels = identity_df[identity_df['matched'] == True].copy()
    print(f"Found {len(matched_vessels)} matched vessels in GFW")
    print(f"Looking back {days_back} days for gap events")
    print()

    config = Config.from_env()
    if not config.gfw_api_token:
        print("❌ ERROR: GFW_API_TOKEN not found")
        return

    client = GFWClient()

    all_gaps = []
    vessels_with_gaps = 0
    total_gaps = 0
    start_time = datetime.now()

    for idx, row in matched_vessels.iterrows():
        vessel_id = row['gfw_vessel_id']
        vessel_name = row['simulated_vessel_name']
        mmsi = row['mmsi']

        elapsed = (datetime.now() - start_time).total_seconds()
        if idx > 0:
            eta_mins = (elapsed / (idx + 1)) * (len(matched_vessels) - idx - 1) / 60
            print(f"[{idx+1}/{len(matched_vessels)}] {vessel_name} [ETA: {eta_mins:.1f}m]...", end=" ")
        else:
            print(f"[{idx+1}/{len(matched_vessels)}] {vessel_name}...", end=" ")

        try:
            gaps = await client.get_gaps(vessel_id, days_back=days_back)

            if gaps and len(gaps) > 0:
                vessels_with_gaps += 1
                total_gaps += len(gaps)

                for gap in gaps:
                    all_gaps.append({
                        'mmsi': mmsi,
                        'simulated_vessel_name': vessel_name,
                        'gap_id': gap.get('id'),
                        'gap_start': gap.get('start'),
                        'gap_end': gap.get('end'),
                        'duration_hours': (
                            (pd.to_datetime(gap.get('end')) - pd.to_datetime(gap.get('start'))).total_seconds() / 3600
                            if gap.get('start') and gap.get('end') else None
                        ),
                        'start_lat': gap.get('position', {}).get('lat') if isinstance(gap.get('position'), dict) else None,
                        'start_lon': gap.get('position', {}).get('lon') if isinstance(gap.get('position'), dict) else None,
                        'end_lat': gap.get('endPosition', {}).get('lat') if isinstance(gap.get('endPosition'), dict) else None,
                        'end_lon': gap.get('endPosition', {}).get('lon') if isinstance(gap.get('endPosition'), dict) else None,
                        'distance_km': gap.get('distanceKm'),
                        'event_timestamp': datetime.utcnow().isoformat()
                    })

                print(f"✅ {len(gaps)} gaps")
            else:
                print("⭕ No gaps")

        except Exception as e:
            print(f"⚠️  Error: {str(e)[:50]}")

        await asyncio.sleep(0.5)

    gaps_df = pd.DataFrame(all_gaps)
    if len(gaps_df) > 0:
        gaps_df.to_csv(output_csv, index=False)

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 80)
    print("GAPS ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Vessels processed: {len(matched_vessels)}")
    print(f"Vessels with gaps: {vessels_with_gaps} ({vessels_with_gaps/len(matched_vessels)*100:.1f}%)")
    print(f"Total gaps: {total_gaps}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"✅ Output saved to: {output_csv}")
    print(f"   Dataset size: {len(gaps_df)} rows")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enrich gap/dark period events from GFW')
    parser.add_argument('--identity', type=str, default='enrichment_outputs/vessel_identity_enrichment.csv')
    parser.add_argument('--output', type=str, default='enrichment_outputs/gaps_enrichment.csv')
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    asyncio.run(enrich_gaps(args.identity, args.output, args.days))
