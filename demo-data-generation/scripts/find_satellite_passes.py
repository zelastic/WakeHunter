#!/usr/bin/env python3
"""Find actual satellite passes over dark vessels"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from tensorgator.coord_conv import lla_to_ecef

# Load satellite positions
sat_positions = pd.read_parquet(Path(__file__).parent.parent / 'outputs' / 'satellite_positions_ecef.parquet')
sat_positions['timestamp_unix'] = pd.to_datetime(sat_positions['timestamp']).astype(int) / 1e9

# Load vessel tracks
gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')
if vessel_tracks['timestamp'].dtype == 'object':
    vessel_tracks['timestamp'] = pd.to_datetime(vessel_tracks['timestamp']).astype(int) / 1e9

# Get dark period vessels
dark_vessels = vessel_tracks[vessel_tracks['ais_visible'] == False]

print("Finding satellite passes over dark vessels...")
print(f"Dark period: {pd.to_datetime(dark_vessels['timestamp'].min(), unit='s', utc=True)} to {pd.to_datetime(dark_vessels['timestamp'].max(), unit='s', utc=True)}")
print(f"Vessel locations: ({dark_vessels['latitude'].min():.2f}°N-{dark_vessels['latitude'].max():.2f}°N, {dark_vessels['longitude'].min():.2f}°E-{dark_vessels['longitude'].max():.2f}°E)")

# Use center of dark vessel region
center_lat = dark_vessels['latitude'].mean()
center_lon = dark_vessels['longitude'].mean()

print(f"\nSearching for passes over ({center_lat:.2f}°, {center_lon:.2f}°)...")

# Filter satellite positions to dark period
dark_start = dark_vessels['timestamp'].min()
dark_end = dark_vessels['timestamp'].max()

sat_during_dark = sat_positions[
    (sat_positions['timestamp_unix'] >= dark_start) &
    (sat_positions['timestamp_unix'] <= dark_end)
]

print(f"\nSatellite positions during dark period: {len(sat_during_dark):,}")

# Calculate elevation for all positions
passes = []

for _, sat in sat_during_dark.iterrows():
    sat_lat = sat['latitude']
    sat_lon = sat['longitude']
    sat_alt_km = sat['altitude_km']

    # Calculate elevation
    sat_ecef = lla_to_ecef(sat_lat, sat_lon, sat_alt_km * 1000)
    vessel_ecef = lla_to_ecef(center_lat, center_lon, 0.0)
    los_vector = sat_ecef - vessel_ecef
    vessel_up = vessel_ecef / np.linalg.norm(vessel_ecef)
    los_magnitude = np.linalg.norm(los_vector)
    cos_zenith = np.dot(los_vector, vessel_up) / los_magnitude
    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
    elevation = 90.0 - zenith_angle

    if elevation > 0:  # Above horizon
        # Calculate distance
        from math import radians, sin, cos, sqrt, atan2
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [center_lat, center_lon, sat_lat, sat_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance_km = R * c

        passes.append({
            'timestamp': sat['timestamp_unix'],
            'timestamp_dt': pd.to_datetime(sat['timestamp_unix'], unit='s', utc=True),
            'satellite_id': sat['satellite_id'],
            'sensor_type': sat['sensor_type'],
            'elevation_deg': elevation,
            'distance_km': distance_km,
            'sat_lat': sat_lat,
            'sat_lon': sat_lon,
            'sat_alt_km': sat_alt_km
        })

passes_df = pd.DataFrame(passes)

if len(passes_df) > 0:
    print(f"\n✅ Found {len(passes_df)} satellite positions with elevation > 0°")

    print(f"\nPasses by satellite:")
    for sat_id in passes_df['satellite_id'].unique():
        sat_passes = passes_df[passes_df['satellite_id'] == sat_id]
        sensor_type = sat_passes.iloc[0]['sensor_type']
        max_elev = sat_passes['elevation_deg'].max()
        print(f"  {sat_id} ({sensor_type}): {len(sat_passes)} positions, max elevation: {max_elev:.1f}°")

    print(f"\nPasses with elevation > 30°:")
    high_passes = passes_df[passes_df['elevation_deg'] > 30]
    print(f"  Total: {len(high_passes)} positions")
    if len(high_passes) > 0:
        for sat_id in high_passes['satellite_id'].unique():
            sat_passes = high_passes[high_passes['satellite_id'] == sat_id]
            sensor_type = sat_passes.iloc[0]['sensor_type']
            max_elev = sat_passes['elevation_deg'].max()
            min_dist = sat_passes['distance_km'].min()
            print(f"  {sat_id} ({sensor_type}): {len(sat_passes)} positions, max elevation: {max_elev:.1f}°, min distance: {min_dist:.1f} km")

            # Show first and last timestamps
            first_time = sat_passes['timestamp_dt'].min()
            last_time = sat_passes['timestamp_dt'].max()
            print(f"    Time range: {first_time} to {last_time}")

    # Export for visualization
    output_file = Path(__file__).parent.parent / 'outputs' / 'satellite_passes_during_dark.parquet'
    passes_df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved passes to: {output_file}")

else:
    print("\n⚠️  No satellite passes found with elevation > 0°")
    print("This means satellites never pass over the vessel location during the dark period.")
    print("Need to:")
    print("  1. Adjust satellite orbital elements to ensure coverage")
    print("  2. Or extend the dark period to capture more orbital passes")
    print("  3. Or use a wider geographic region")
