#!/usr/bin/env python3
"""
Export Demo Data for Vessel Simulation Debug Dashboard

Converts our demo data (vessel tracks, detections) into the format
expected by the vessel-simulation-service debug dashboard.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Export demo data for debug dashboard."""
    logger.info("=" * 70)
    logger.info("EXPORTING DATA FOR DEBUG DASHBOARD")
    logger.info("=" * 70)

    # Paths
    demo_outputs = Path(__file__).parent.parent / 'outputs'
    vessel_sim_dir = Path(__file__).parent.parent.parent / 'vessel-simulation-service'
    output_dir = vessel_sim_dir / 'dark_period_demo'
    output_dir.mkdir(exist_ok=True)

    # Load vessel tracks
    logger.info("Loading vessel tracks...")
    gold_tracks = pd.read_parquet(demo_outputs / 'vessel_tracks_dark_period.parquet')
    background_tracks = pd.read_parquet(demo_outputs / 'background_vessel_tracks.parquet')

    # Merge all vessel tracks
    tracks = pd.concat([gold_tracks, background_tracks], ignore_index=True)
    logger.info(f"  Gold vessels: {gold_tracks['mmsi'].nunique()} vessels, {len(gold_tracks):,} positions")
    logger.info(f"  Background vessels: {background_tracks['mmsi'].nunique()} vessels, {len(background_tracks):,} positions")
    logger.info(f"  Total: {tracks['mmsi'].nunique()} vessels, {len(tracks):,} positions")

    # Convert timestamp to Unix timestamp (seconds since epoch)
    tracks['timestamp_iso'] = tracks['timestamp']
    tracks['timestamp'] = pd.to_datetime(tracks['timestamp']).astype(int) / 10**9

    # Split by vessel and save as AIS dynamic files
    for mmsi in tracks['mmsi'].unique():
        vessel_tracks = tracks[tracks['mmsi'] == mmsi].copy()
        vessel_name = vessel_tracks.iloc[0]['name'].replace(' ', '_')

        # Rename columns to match expected format
        ais_data = pd.DataFrame({
            'timestamp': vessel_tracks['timestamp'],
            'mmsi': vessel_tracks['mmsi'],
            'lat': vessel_tracks['lat'],
            'lon': vessel_tracks['lon'],
            'sog': vessel_tracks['speed_knots'],
            'cog': vessel_tracks['heading'],
            'heading': vessel_tracks['heading'],
            'vessel_name': vessel_tracks['name'],
            'vessel_type': vessel_tracks['vessel_type'],
            'flag': vessel_tracks['flag'],
            'status': vessel_tracks['status'],
            'ais_on': vessel_tracks['ais_on']
        })

        output_path = output_dir / f"ais_{mmsi}_dynamic.parquet"
        ais_data.to_parquet(output_path, index=False)
        logger.info(f"  ✅ Exported {vessel_name} (MMSI: {mmsi}): {len(ais_data)} positions")

    # Load and export detection events as ground truth
    logger.info("Loading detection events...")
    with open(demo_outputs / 'detection_events_dark_period.json') as f:
        detections = json.load(f)

    # Convert to ground truth format
    ground_truth_events = []
    for det in detections['detections']:
        # Convert timestamp to Unix timestamp
        ts = datetime.fromisoformat(det['timestamp'].removesuffix('Z'))
        unix_ts = ts.timestamp()

        ground_truth_events.append({
            'timestamp': unix_ts,
            'event_type': f"satellite_detection_{det['sensor_type'].lower()}",
            'vessel_mmsi': det['vessel_mmsi'],
            'vessel_name': det['vessel_name'],
            'satellite_id': det['satellite_id'],
            'sensor_type': det['sensor_type'],
            'detection_confidence': det['detection_confidence'],
            'lat': det['vessel_lat'],
            'lon': det['vessel_lon'],
            'metadata': {
                'elevation_deg': det['elevation_deg'],
                'weather': det['weather'],
                'cloud_cover_pct': det['cloud_cover_pct']
            }
        })

    # Save ground truth
    gt_df = pd.DataFrame(ground_truth_events)
    gt_path = output_dir / "ground_truth_events.parquet"
    gt_df.to_parquet(gt_path, index=False)
    logger.info(f"  ✅ Exported {len(gt_df)} detection events")

    # Create simulation summary
    summary = {
        'simulation_id': 'dark_period_demo',
        'description': 'Dark Period STS Transfer Demo - July 17-20, 2025',
        'period': {
            'start': '2025-07-17T20:45:00Z',
            'end': '2025-07-20T07:00:00Z',
            'duration_hours': 58.25
        },
        'gold_vessels': [
            {
                'mmsi': 477105700,
                'name': 'YUE CHI',
                'type': 'Tanker',
                'role': 'supplier',
                'is_sanctioned': False
            },
            {
                'mmsi': 412999999,
                'name': 'SANCTIONED VESSEL',
                'type': 'Cargo',
                'role': 'receiver',
                'is_sanctioned': True,
                'dark_ship': True
            }
        ],
        'statistics': {
            'total_vessels': tracks['mmsi'].nunique(),
            'gold_vessels': gold_tracks['mmsi'].nunique(),
            'background_vessels': background_tracks['mmsi'].nunique(),
            'total_positions': len(tracks),
            'total_detections': len(ground_truth_events),
            'satellites': 9,
            'satellite_passes': 48
        }
    }

    summary_path = output_dir / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✅ Exported simulation summary")

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ EXPORT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("To view in debug dashboard:")
    logger.info(f"  http://localhost:8003/debug?simulation_dir=dark_period_demo")
    logger.info("")


if __name__ == "__main__":
    main()
