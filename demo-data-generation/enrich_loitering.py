#!/usr/bin/env python3
"""
Loitering Events Enrichment Dataset Generator

Creates a dataset of loitering events from Global Fishing Watch.
Loitering indicates extended periods in one location, potentially for illegal transshipment.

Output: enrichment_outputs/loitering_enrichment.csv
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append('/Users/zski/PycharmProjects/ElasticShipVision/global-fishing-api-service')

from src.client import GFWClient
from src.config import Config

async def enrich_loitering(identity_csv: str, output_csv: str, days_back: int = 365):
    print("=" * 80)
    print("LOITERING EVENTS ENRICHMENT")
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
    print(f"Looking back {days_back} days for loitering events")
    print()

    config = Config.from_env()
    if not config.gfw_api_token:
        print("❌ ERROR: GFW_API_TOKEN not found")
        return

    client = GFWClient()

    all_loitering = []
    vessels_with_loitering = 0
    total_loitering = 0
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
            loitering = await client.get_loitering(vessel_id, days_back=days_back)

            if loitering and len(loitering) > 0:
                vessels_with_loitering += 1
                total_loitering += len(loitering)

                for event in loitering:
                    all_loitering.append({
                        'mmsi': mmsi,
                        'simulated_vessel_name': vessel_name,
                        'loitering_id': event.get('id'),
                        'loitering_start': event.get('start'),
                        'loitering_end': event.get('end'),
                        'duration_hours': (
                            (pd.to_datetime(event.get('end')) - pd.to_datetime(event.get('start'))).total_seconds() / 3600
                            if event.get('start') and event.get('end') else None
                        ),
                        'lat': event.get('position', {}).get('lat') if isinstance(event.get('position'), dict) else None,
                        'lon': event.get('position', {}).get('lon') if isinstance(event.get('position'), dict) else None,
                        'total_distance_km': event.get('totalDistanceKm'),
                        'avg_speed_knots': event.get('avgSpeedKnots'),
                        'event_timestamp': datetime.utcnow().isoformat()
                    })

                print(f"✅ {len(loitering)} events")
            else:
                print("⭕ No loitering")

        except Exception as e:
            print(f"⚠️  Error: {str(e)[:50]}")

        await asyncio.sleep(0.5)

    loitering_df = pd.DataFrame(all_loitering)
    if len(loitering_df) > 0:
        loitering_df.to_csv(output_csv, index=False)

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 80)
    print("LOITERING ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Vessels processed: {len(matched_vessels)}")
    print(f"Vessels with loitering: {vessels_with_loitering} ({vessels_with_loitering/len(matched_vessels)*100:.1f}%)")
    print(f"Total loitering events: {total_loitering}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"✅ Output saved to: {output_csv}")
    print(f"   Dataset size: {len(loitering_df)} rows")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enrich loitering events from GFW')
    parser.add_argument('--identity', type=str, default='enrichment_outputs/vessel_identity_enrichment.csv')
    parser.add_argument('--output', type=str, default='enrichment_outputs/loitering_enrichment.csv')
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    asyncio.run(enrich_loitering(args.identity, args.output, args.days))
