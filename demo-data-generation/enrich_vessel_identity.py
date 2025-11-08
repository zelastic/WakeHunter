#!/usr/bin/env python3
"""
Vessel Identity Enrichment Dataset Generator

Creates a dataset linking simulated vessels (by MMSI) to their real-world identity
from Global Fishing Watch, including flag, IMO, call sign, and AIS transmission history.

Output: enrichment_outputs/vessel_identity_enrichment.csv
Columns:
  - mmsi: Link to simulation data
  - simulated_vessel_name: Name from our simulation
  - gfw_vessel_name: Official name from GFW
  - gfw_vessel_id: GFW internal identifier
  - flag: Vessel flag country code
  - imo: International Maritime Organization number
  - call_sign: Radio call sign
  - ais_messages_count: Total AIS messages in GFW database
  - first_seen: First AIS transmission date
  - last_seen: Last AIS transmission date
  - years_active: Years of AIS activity
  - matched: Whether vessel was found in GFW
  - lookup_timestamp: When enrichment was performed
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add GFW service to path
sys.path.append('/Users/zski/PycharmProjects/ElasticShipVision/global-fishing-api-service')

from src.client import GFWClient
from src.config import Config

async def enrich_vessel_identity(simulation_parquet: str, output_csv: str, limit: Optional[int] = None):
    """
    Generate vessel identity enrichment dataset.

    Args:
        simulation_parquet: Path to simulation AIS data
        output_csv: Output path for enriched CSV
        limit: Optional limit on number of vessels to process (for testing)
    """
    print("=" * 80)
    print("VESSEL IDENTITY ENRICHMENT")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load simulation data
    print(f"Loading simulation data from: {simulation_parquet}")
    sim_df = pd.read_parquet(simulation_parquet)

    # Get unique vessels
    vessels = sim_df[['vessel_name', 'mmsi']].drop_duplicates().reset_index(drop=True)

    if limit:
        vessels = vessels.head(limit)
        print(f"⚠️  LIMIT: Processing only first {limit} vessels for testing")

    print(f"Found {len(vessels)} unique vessels to enrich")
    print()

    # Initialize GFW client
    config = Config.from_env()
    if not config.gfw_api_token:
        print("❌ ERROR: GFW_API_TOKEN not found in environment")
        print("   Set it with: export GFW_API_TOKEN='your_token'")
        print("   Get token from: https://globalfishingwatch.org/our-apis/tokens")
        return

    client = GFWClient()

    # Enrich each vessel
    enriched_data = []
    matched_count = 0
    failed_count = 0
    start_time = datetime.now()

    for idx, row in vessels.iterrows():
        mmsi_str = str(int(row['mmsi']))
        vessel_name = row['vessel_name']

        # Progress with ETA
        elapsed = (datetime.now() - start_time).total_seconds()
        if idx > 0:
            avg_time_per_vessel = elapsed / idx
            remaining = (len(vessels) - idx) * avg_time_per_vessel
            eta_mins = remaining / 60
            print(f"[{idx+1}/{len(vessels)}] {vessel_name} (MMSI: {mmsi_str}) [ETA: {eta_mins:.1f}m]...", end=" ")
        else:
            print(f"[{idx+1}/{len(vessels)}] {vessel_name} (MMSI: {mmsi_str})...", end=" ")

        try:
            # Search for vessel in GFW
            vessel_id, gfw_name, vessel_obj = await client.search_vessel(mmsi_str)

            if vessel_id and vessel_obj:
                # Extract vessel information
                first_seen = vessel_obj.get('firstTransmissionDate', None)
                last_seen = vessel_obj.get('lastTransmissionDate', None)

                # Calculate years active
                years_active = None
                if first_seen and last_seen:
                    try:
                        first_dt = pd.to_datetime(first_seen)
                        last_dt = pd.to_datetime(last_seen)
                        years_active = (last_dt - first_dt).days / 365.25
                    except:
                        pass

                enriched_data.append({
                    'mmsi': int(row['mmsi']),
                    'simulated_vessel_name': vessel_name,
                    'gfw_vessel_name': gfw_name,
                    'gfw_vessel_id': vessel_id,
                    'flag': vessel_obj.get('flag', None),
                    'imo': vessel_obj.get('imo', None),
                    'call_sign': vessel_obj.get('callsign', None),
                    'ais_messages_count': vessel_obj.get('messagesCounter', None),
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'years_active': round(years_active, 2) if years_active else None,
                    'matched': True,
                    'lookup_timestamp': datetime.utcnow().isoformat()
                })

                matched_count += 1
                print(f"✅ {gfw_name} ({vessel_obj.get('flag', 'N/A')})")
            else:
                # Not found in GFW
                enriched_data.append({
                    'mmsi': int(row['mmsi']),
                    'simulated_vessel_name': vessel_name,
                    'gfw_vessel_name': None,
                    'gfw_vessel_id': None,
                    'flag': None,
                    'imo': None,
                    'call_sign': None,
                    'ais_messages_count': None,
                    'first_seen': None,
                    'last_seen': None,
                    'years_active': None,
                    'matched': False,
                    'lookup_timestamp': datetime.utcnow().isoformat()
                })

                failed_count += 1
                print("❌ Not found")

        except Exception as e:
            # Error during lookup
            enriched_data.append({
                'mmsi': int(row['mmsi']),
                'simulated_vessel_name': vessel_name,
                'gfw_vessel_name': None,
                'gfw_vessel_id': None,
                'flag': None,
                'imo': None,
                'call_sign': None,
                'ais_messages_count': None,
                'first_seen': None,
                'last_seen': None,
                'years_active': None,
                'matched': False,
                'lookup_timestamp': datetime.utcnow().isoformat()
            })

            failed_count += 1
            print(f"⚠️  Error: {str(e)[:50]}")

        # Rate limiting: 0.5s delay between requests
        await asyncio.sleep(0.5)

    # Create DataFrame and save
    enriched_df = pd.DataFrame(enriched_data)
    enriched_df.to_csv(output_csv, index=False)

    total_time = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Total vessels: {len(vessels)}")
    print(f"Matched in GFW: {matched_count} ({matched_count/len(vessels)*100:.1f}%)")
    print(f"Not found: {failed_count} ({failed_count/len(vessels)*100:.1f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"✅ Output saved to: {output_csv}")
    print(f"   Dataset size: {len(enriched_df)} rows")
    print()

    # Show sample
    if len(enriched_df) > 0:
        print("Sample matched vessels:")
        matched_sample = enriched_df[enriched_df['matched'] == True].head(5)
        if len(matched_sample) > 0:
            print(matched_sample[['mmsi', 'simulated_vessel_name', 'gfw_vessel_name', 'flag', 'years_active']].to_string(index=False))
        print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enrich vessel identity from GFW')
    parser.add_argument('--simulation', type=str,
                       default='/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/debug-outputs/realistic_500vessels_july_14day_20251106_202136/ais_202507_dynamic.parquet',
                       help='Path to simulation parquet file')
    parser.add_argument('--output', type=str,
                       default='enrichment_outputs/vessel_identity_enrichment.csv',
                       help='Output CSV path')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of vessels (for testing)')

    args = parser.parse_args()

    asyncio.run(enrich_vessel_identity(args.simulation, args.output, args.limit))
