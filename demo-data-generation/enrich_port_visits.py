#!/usr/bin/env python3
"""
Port Visits Enrichment Dataset Generator

Creates a dataset of port visit events from Global Fishing Watch.
Shows where vessels docked and for how long.

Output: enrichment_outputs/port_visits_enrichment.csv
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append('/Users/zski/PycharmProjects/ElasticShipVision/global-fishing-api-service')

from src.client import GFWClient
from src.config import Config

async def enrich_port_visits(identity_csv: str, output_csv: str, days_back: int = 365):
    print("=" * 80)
    print("PORT VISITS ENRICHMENT")
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
    print(f"Looking back {days_back} days for port visits")
    print()

    config = Config.from_env()
    if not config.gfw_api_token:
        print("❌ ERROR: GFW_API_TOKEN not found")
        return

    client = GFWClient()

    all_port_visits = []
    vessels_with_visits = 0
    total_visits = 0
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
            port_visits = await client.get_port_visits(vessel_id, days_back=days_back)

            if port_visits and len(port_visits) > 0:
                vessels_with_visits += 1
                total_visits += len(port_visits)

                for visit in port_visits:
                    all_port_visits.append({
                        'mmsi': mmsi,
                        'simulated_vessel_name': vessel_name,
                        'visit_id': visit.get('id'),
                        'port_name': visit.get('port', {}).get('name') if isinstance(visit.get('port'), dict) else None,
                        'port_id': visit.get('port', {}).get('id') if isinstance(visit.get('port'), dict) else None,
                        'port_lat': visit.get('position', {}).get('lat') if isinstance(visit.get('position'), dict) else None,
                        'port_lon': visit.get('position', {}).get('lon') if isinstance(visit.get('position'), dict) else None,
                        'arrival_time': visit.get('start'),
                        'departure_time': visit.get('end'),
                        'duration_hours': (
                            (pd.to_datetime(visit.get('end')) - pd.to_datetime(visit.get('start'))).total_seconds() / 3600
                            if visit.get('start') and visit.get('end') else None
                        ),
                        'confidence': visit.get('confidence'),
                        'event_timestamp': datetime.utcnow().isoformat()
                    })

                print(f"✅ {len(port_visits)} visits")
            else:
                print("⭕ No visits")

        except Exception as e:
            print(f"⚠️  Error: {str(e)[:50]}")

        await asyncio.sleep(0.5)

    port_visits_df = pd.DataFrame(all_port_visits)
    if len(port_visits_df) > 0:
        port_visits_df.to_csv(output_csv, index=False)

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 80)
    print("PORT VISITS ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Vessels processed: {len(matched_vessels)}")
    print(f"Vessels with visits: {vessels_with_visits} ({vessels_with_visits/len(matched_vessels)*100:.1f}%)")
    print(f"Total visits: {total_visits}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"✅ Output saved to: {output_csv}")
    print(f"   Dataset size: {len(port_visits_df)} rows")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enrich port visit events from GFW')
    parser.add_argument('--identity', type=str, default='enrichment_outputs/vessel_identity_enrichment.csv')
    parser.add_argument('--output', type=str, default='enrichment_outputs/port_visits_enrichment.csv')
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    asyncio.run(enrich_port_visits(args.identity, args.output, args.days))
