"""
Add satellite detection events to existing gold_standard vessel simulation
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Add demo-data-generation to path
sys.path.append(str(Path(__file__).parent.parent))

def generate_detections_for_existing_tracks():
    """Generate satellite detections for existing gold_standard vessel tracks"""

    print("="*70)
    print("GENERATING SATELLITE DETECTIONS FOR GOLD STANDARD TRACKS")
    print("="*70)

    # Load existing vessel tracks
    gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
    vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')

    print(f"\n✅ Loaded {len(vessel_tracks)} vessel positions")
    print(f"   Vessels: {vessel_tracks['mmsi'].unique()}")
    print(f"   Time range: {pd.to_datetime(vessel_tracks['timestamp'].min(), unit='s', utc=True)} to {pd.to_datetime(vessel_tracks['timestamp'].max(), unit='s', utc=True)}")

    # Load existing ground truth events to understand dark periods
    existing_events = pd.read_parquet(gold_standard_dir / 'ground_truth_events.parquet')
    print(f"\n✅ Loaded {len(existing_events)} existing ground truth events")
    print(f"   Event types: {existing_events['event_type'].value_counts().to_dict()}")

    # Simple satellite pass simulation (without full orbital propagation)
    # Generate detection events at regular intervals when vessels are visible

    detections = []
    detection_id = 0

    # Group by vessel
    for mmsi in vessel_tracks['mmsi'].unique():
        vessel_data = vessel_tracks[vessel_tracks['mmsi'] == mmsi].sort_values('timestamp')
        vessel_name = vessel_data.iloc[0]['vessel_name']

        print(f"\n📡 Generating detections for {vessel_name} (MMSI: {mmsi})")

        # Generate detections every ~2 hours when vessel is visible
        detection_interval = timedelta(hours=2)
        current_time = pd.to_datetime(vessel_data.iloc[0]['timestamp'], unit='s', utc=True)
        end_time = pd.to_datetime(vessel_data.iloc[-1]['timestamp'], unit='s', utc=True)

        satellites = [
            ('elk-1-rf', 'RF'),
            ('elk-2-rf', 'RF'),
            ('elk-1-sar', 'SAR'),
            ('elk-2-sar', 'SAR'),
            ('elk-1-eo', 'EO'),
            ('elk-2-eo', 'EO'),
        ]

        sat_idx = 0
        while current_time < end_time:
            # Find closest vessel position
            time_diffs = abs(vessel_data['timestamp'] - current_time.timestamp())
            closest_idx = time_diffs.idxmin()
            vessel_pos = vessel_data.loc[closest_idx]

            # Check if AIS is visible
            ais_visible = vessel_pos['ais_visible']

            # Select satellite for this detection
            satellite_id, sensor_type = satellites[sat_idx % len(satellites)]
            sat_idx += 1

            # Generate detection with varying confidence
            if ais_visible:
                # Higher confidence when AIS is on
                confidence = np.random.uniform(0.85, 0.98)
            else:
                # Lower confidence during dark period
                confidence = np.random.uniform(0.65, 0.85)

            # Determine weather conditions
            hour = current_time.hour
            if 6 <= hour < 18:  # Daytime
                weather = 'clear'
                cloud_cover = np.random.uniform(5, 20)
            else:  # Nighttime
                weather = 'partly_cloudy'
                cloud_cover = np.random.uniform(30, 60)

            # Only EO detections during clear weather
            if sensor_type == 'EO' and cloud_cover > 50:
                current_time += detection_interval
                continue

            detections.append({
                'timestamp': current_time.timestamp(),
                'event_type': f'satellite_detection_{sensor_type.lower()}',
                'vessel_mmsi': int(mmsi),
                'vessel_name': vessel_name,
                'satellite_id': satellite_id,
                'sensor_type': sensor_type,
                'detection_confidence': confidence,
                'lat': vessel_pos['latitude'],
                'lon': vessel_pos['longitude'],
                'latitude': vessel_pos['latitude'],  # For frontend compatibility
                'longitude': vessel_pos['longitude'],
                'metadata': {
                    'cloud_cover_pct': float(cloud_cover),
                    'elevation_deg': float(np.random.uniform(30, 75)),
                    'weather': weather,
                    'ais_visible': bool(ais_visible)
                }
            })

            current_time += detection_interval

        print(f"   Generated {len([d for d in detections if d['vessel_mmsi'] == mmsi])} detections")

    # Create detection dataframe
    detections_df = pd.DataFrame(detections)

    # Combine with existing ground truth events
    # Keep original event structure for vessel behavior events
    combined_events = []

    # Add existing events (converted to satellite detection format where possible)
    for _, event in existing_events.iterrows():
        combined_events.append({
            'timestamp': pd.to_datetime(event['event_time']).timestamp() if 'event_time' in event else None,
            'event_type': event['event_type'],
            'vessel_mmsi': event['mmsi'],
            'vessel_name': event['vessel_name'],
            'satellite_id': None,
            'sensor_type': None,
            'detection_confidence': None,
            'lat': event['latitude'],
            'lon': event['longitude'],
            'latitude': event['latitude'],
            'longitude': event['longitude'],
            'metadata': event.get('metadata', {})
        })

    # Add satellite detections
    combined_events.extend(detections)

    combined_df = pd.DataFrame(combined_events)
    detections_df = pd.DataFrame(detections)

    print(f"\n✅ Generated {len(detections)} satellite detections")
    print(f"   RF: {len(detections_df[detections_df['sensor_type'] == 'RF'])}")
    print(f"   SAR: {len(detections_df[detections_df['sensor_type'] == 'SAR'])}")
    print(f"   EO: {len(detections_df[detections_df['sensor_type'] == 'EO'])}")

    # Save combined events
    output_file = gold_standard_dir / 'ground_truth_events_with_detections.parquet'
    combined_df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved combined events to: {output_file}")
    print(f"   Total events: {len(combined_df)}")
    print(f"   Vessel behavior events: {len(existing_events)}")
    print(f"   Satellite detections: {len(detections_df)}")

    # Also update the main ground_truth_events.parquet
    backup_file = gold_standard_dir / 'ground_truth_events_BACKUP.parquet'
    if not backup_file.exists():
        existing_events.to_parquet(backup_file, index=False)
        print(f"   Created backup: {backup_file}")

    combined_df.to_parquet(gold_standard_dir / 'ground_truth_events.parquet', index=False)
    print(f"   Updated: {gold_standard_dir / 'ground_truth_events.parquet'}")

    return combined_df

if __name__ == '__main__':
    generate_detections_for_existing_tracks()
    print("\n✅ Done!")
