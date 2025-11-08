#!/usr/bin/env python3
"""Debug satellite detection coverage"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from tensorgator.coord_conv import lla_to_ecef

# Load vessel tracks
gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')

# Convert timestamp to unix if needed
if vessel_tracks['timestamp'].dtype == 'object':
    vessel_tracks['timestamp'] = pd.to_datetime(vessel_tracks['timestamp']).astype(int) / 1e9

print("Vessel Tracks Sample:")
print(vessel_tracks.head())
print(f"\nColumns: {vessel_tracks.columns.tolist()}")
print(f"\nTimestamp range: {vessel_tracks['timestamp'].min()} to {vessel_tracks['timestamp'].max()}")
print(f"\nVessels: {vessel_tracks['mmsi'].unique() if 'mmsi' in vessel_tracks.columns else 'N/A'}")

# Check dark period
dark_vessels = vessel_tracks[vessel_tracks['ais_visible'] == False]
print(f"\n\nDark Period Positions: {len(dark_vessels)}")
print(dark_vessels.head(20))

if len(dark_vessels) > 0:
    print(f"\nDark period vessels:")
    for mmsi in dark_vessels['mmsi'].unique():
        vessel_data = dark_vessels[dark_vessels['mmsi'] == mmsi]
        print(f"  {vessel_data.iloc[0]['vessel_name']}: {len(vessel_data)} positions")
        print(f"    Lat range: {vessel_data['latitude'].min():.2f} to {vessel_data['latitude'].max():.2f}")
        print(f"    Lon range: {vessel_data['longitude'].min():.2f} to {vessel_data['longitude'].max():.2f}")
        print(f"    Time: {pd.to_datetime(vessel_data['timestamp'].min(), unit='s', utc=True)} to {pd.to_datetime(vessel_data['timestamp'].max(), unit='s', utc=True)}")

# Load satellite positions
sat_positions = pd.read_parquet(Path(__file__).parent.parent / 'outputs' / 'satellite_positions_ecef.parquet')
sat_positions['timestamp_unix'] = pd.to_datetime(sat_positions['timestamp']).astype(int) / 1e9

print(f"\n\nSatellite Positions Sample:")
print(sat_positions.head())

# Check RF satellite coverage over vessels during dark period
if len(dark_vessels) > 0:
    print(f"\n\nChecking RF satellite coverage during dark period...")

    # Get a specific dark vessel position
    sample_dark = dark_vessels.iloc[0]
    vessel_lat = sample_dark['latitude']
    vessel_lon = sample_dark['longitude']
    vessel_time = sample_dark['timestamp']

    print(f"\nSample dark vessel position:")
    print(f"  Vessel: {sample_dark['vessel_name']}")
    print(f"  Time: {pd.to_datetime(vessel_time, unit='s', utc=True)}")
    print(f"  Position: ({vessel_lat:.4f}, {vessel_lon:.4f})")

    # Find satellites at same time
    time_window = 300  # 5 minutes
    sats_nearby = sat_positions[
        (sat_positions['timestamp_unix'] >= vessel_time - time_window) &
        (sat_positions['timestamp_unix'] <= vessel_time + time_window)
    ]

    print(f"\n  Satellites within {time_window}s:")
    print(f"    Found {len(sats_nearby)} satellite positions")

    if len(sats_nearby) > 0:
        # Calculate distances
        for _, sat in sats_nearby.iterrows():
            sat_lat = sat['latitude']
            sat_lon = sat['longitude']
            sat_alt_km = sat['altitude_km']

            # Haversine distance
            from math import radians, sin, cos, sqrt, atan2
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [vessel_lat, vessel_lon, sat_lat, sat_lon])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance_km = R * c

            # Calculate elevation angle
            sat_ecef = lla_to_ecef(sat_lat, sat_lon, sat_alt_km * 1000)
            vessel_ecef = lla_to_ecef(vessel_lat, vessel_lon, 0.0)
            los_vector = sat_ecef - vessel_ecef
            vessel_up = vessel_ecef / np.linalg.norm(vessel_ecef)
            los_magnitude = np.linalg.norm(los_vector)
            cos_zenith = np.dot(los_vector, vessel_up) / los_magnitude
            zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
            elevation_angle = 90.0 - zenith_angle

            # RF footprint radius
            rf_radius_km = sat_alt_km * np.tan(np.radians(30.0))

            print(f"\n    {sat['satellite_id']} ({sat['sensor_type']}):")
            print(f"      Distance to vessel: {distance_km:.1f} km")
            print(f"      Elevation angle: {elevation_angle:.1f}°")
            print(f"      Altitude: {sat_alt_km:.1f} km")
            if sat['sensor_type'] == 'RF':
                print(f"      RF footprint radius: {rf_radius_km:.1f} km")
                print(f"      In footprint: {distance_km <= rf_radius_km and elevation_angle >= 30}")
