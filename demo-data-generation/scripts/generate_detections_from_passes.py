#!/usr/bin/env python3
"""
Generate satellite detections from pre-computed passes during dark periods
Simplified approach: match passes with dark vessel positions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

print("="*70)
print("GENERATING DETECTIONS FROM SATELLITE PASSES")
print("="*70)

# Load satellite passes (using elevation > 10° for better coverage)
passes_file = Path(__file__).parent.parent / 'outputs' / 'satellite_passes_during_dark.parquet'
sat_passes = pd.read_parquet(passes_file)
sat_passes = sat_passes[sat_passes['elevation_deg'] > 10].sort_values('timestamp')  # Lower threshold

print(f"\n✅ Loaded {len(sat_passes)} satellite passes (elevation > 10°)")

# Load vessel tracks
gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')
if vessel_tracks['timestamp'].dtype == 'object':
    vessel_tracks['timestamp'] = pd.to_datetime(vessel_tracks['timestamp']).astype(int) / 1e9

# Filter to dark vessels only
dark_vessels = vessel_tracks[vessel_tracks['ais_visible'] == False].copy()

print(f"✅ Loaded {len(dark_vessels)} dark vessel positions")
print(f"   Vessels: {', '.join(dark_vessels['vessel_name'].unique())}")
print(f"   Time range: {pd.to_datetime(dark_vessels['timestamp'].min(), unit='s', utc=True)} to {pd.to_datetime(dark_vessels['timestamp'].max(), unit='s', utc=True)}")

# Sensor specs (using very generous swath widths for demo visualization and Cesium)
# Note: These are larger than realistic values to ensure good coverage for the demo
SENSOR_SPECS = {
    'RF': {'swath_km': 800, 'min_conf': 0.70, 'max_conf': 0.95},  # RF can detect over wide area
    'SAR': {'swath_km': 700, 'min_conf': 0.75, 'max_conf': 0.97},  # Wide swath for demo
    'EO': {'swath_km': 500, 'min_conf': 0.80, 'max_conf': 0.98, 'max_cloud': 50}  # Wide swath for demo
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Generate detections
detections = []
detection_count = {'RF': 0, 'SAR': 0, 'EO': 0}

print(f"\n🛰️  Generating detections...")

for _, pass_row in sat_passes.iterrows():
    ts = pass_row['timestamp']
    sat_id = pass_row['satellite_id']
    sensor_type = pass_row['sensor_type']
    sat_lat = pass_row['sat_lat']
    sat_lon = pass_row['sat_lon']
    sat_alt_km = pass_row['sat_alt_km']
    elevation = pass_row['elevation_deg']
    distance_km = pass_row['distance_km']

    specs = SENSOR_SPECS[sensor_type]

    # Find dark vessels near this time (within 2 minutes)
    time_window = 120
    nearby_vessels = dark_vessels[
        (dark_vessels['timestamp'] >= ts - time_window) &
        (dark_vessels['timestamp'] <= ts + time_window)
    ]

    for _, vessel in nearby_vessels.iterrows():
        # Calculate actual distance from satellite nadir to vessel
        actual_distance = haversine_distance(sat_lat, sat_lon, vessel['latitude'], vessel['longitude'])

        # Simple distance check: is vessel within sensor swath?
        if actual_distance <= specs['swath_km']:
            # Weather check for EO
            if sensor_type == 'EO':
                hour = pd.to_datetime(ts, unit='s', utc=True).hour
                if 6 <= hour < 18:
                    cloud_cover = np.random.uniform(5, 20)
                    weather = 'clear'
                else:
                    cloud_cover = np.random.uniform(30, 60)
                    weather = 'partly_cloudy'

                if cloud_cover > specs['max_cloud']:
                    continue
            else:
                hour = pd.to_datetime(ts, unit='s', utc=True).hour
                if 6 <= hour < 18:
                    cloud_cover = np.random.uniform(5, 20)
                    weather = 'clear'
                else:
                    cloud_cover = np.random.uniform(30, 60)
                    weather = 'partly_cloudy'

            # Compute confidence based on elevation
            elevation_factor = (elevation - 30) / 60  # Normalize 30-90° to 0-1
            base_conf = specs['min_conf'] + (specs['max_conf'] - specs['min_conf']) * elevation_factor
            confidence = base_conf + np.random.uniform(-0.02, 0.02)
            confidence = np.clip(confidence, specs['min_conf'], specs['max_conf'])

            detections.append({
                'timestamp': ts,
                'event_type': f'satellite_detection_{sensor_type.lower()}',
                'vessel_mmsi': int(vessel['mmsi']),
                'vessel_name': vessel['vessel_name'],
                'satellite_id': sat_id,
                'sensor_type': sensor_type,
                'detection_confidence': float(confidence),
                'lat': float(vessel['latitude']),
                'lon': float(vessel['longitude']),
                'latitude': float(vessel['latitude']),
                'longitude': float(vessel['longitude']),
                'metadata': {
                    'elevation_deg': float(elevation),
                    'cloud_cover_pct': float(cloud_cover),
                    'weather': weather,
                    'ais_visible': False,
                    'satellite_altitude_km': float(sat_alt_km),
                    'distance_km': float(actual_distance)
                }
            })
            detection_count[sensor_type] += 1

print(f"\n✅ Generated {len(detections)} detections:")
print(f"   RF: {detection_count['RF']}")
print(f"   SAR: {detection_count['SAR']}")
print(f"   EO: {detection_count['EO']}")

if len(detections) > 0:
    # Save detections
    detections_df = pd.DataFrame(detections)
    output_file = gold_standard_dir / 'physics_based_detections.parquet'
    detections_df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved detections to: {output_file}")

    # Load existing events
    try:
        existing_events = pd.read_parquet(gold_standard_dir / 'ground_truth_events_BACKUP.parquet')
    except:
        existing_events = pd.read_parquet(gold_standard_dir / 'ground_truth_events.parquet')
        existing_events = existing_events[existing_events['event_type'].isin(['go_dark', 'sts_transfer', 'reappear'])]

    # Convert existing events to standard format
    combined_events = []
    for _, event in existing_events.iterrows():
        combined_events.append({
            'timestamp': pd.to_datetime(event.get('event_time', event.get('timestamp'))).timestamp() if 'event_time' in event or 'timestamp' in event else None,
            'event_type': event['event_type'],
            'vessel_mmsi': event.get('mmsi', event.get('vessel_mmsi')),
            'vessel_name': event.get('vessel_name'),
            'satellite_id': None,
            'sensor_type': None,
            'detection_confidence': None,
            'lat': event.get('latitude'),
            'lon': event.get('longitude'),
            'latitude': event.get('latitude'),
            'longitude': event.get('longitude'),
            'metadata': event.get('metadata', {})
        })

    # Add satellite detections
    combined_events.extend(detections)
    combined_df = pd.DataFrame(combined_events)

    # Create backup if doesn't exist
    backup_file = gold_standard_dir / 'ground_truth_events_BACKUP.parquet'
    if not backup_file.exists():
        try:
            orig_events = pd.read_parquet(gold_standard_dir / 'ground_truth_events.parquet')
            orig_events.to_parquet(backup_file, index=False)
            print(f"   Created backup: {backup_file}")
        except:
            pass

    # Save combined events
    combined_df.to_parquet(gold_standard_dir / 'ground_truth_events.parquet', index=False)
    print(f"   Updated: {gold_standard_dir / 'ground_truth_events.parquet'}")
    print(f"\n📦 Combined Events Summary:")
    print(f"   Total events: {len(combined_df)}")
    print(f"   Vessel behavior events: {len(existing_events)}")
    print(f"   Satellite detections: {len(detections_df)}")

else:
    print("\n⚠️  No detections generated")

print("\n✅ Done!")
