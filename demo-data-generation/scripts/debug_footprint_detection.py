#!/usr/bin/env python3
"""Debug footprint detection logic"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))

import pandas as pd
import numpy as np
from tensorgator.coord_conv import lla_to_ecef

# Load one specific pass
sat_passes = pd.read_parquet(Path(__file__).parent.parent / 'outputs' / 'satellite_passes_during_dark.parquet')
sat_passes = sat_passes[sat_passes['elevation_deg'] > 30].sort_values('elevation_deg', ascending=False)

# Get highest elevation RF pass
rf_pass = sat_passes[sat_passes['sensor_type'] == 'RF'].iloc[0]

print(f"Testing RF footprint detection for:")
print(f"  Satellite: {rf_pass['satellite_id']}")
print(f"  Time: {rf_pass['timestamp_dt']}")
print(f"  Elevation: {rf_pass['elevation_deg']:.1f}°")
print(f"  Sat position: ({rf_pass['sat_lat']:.2f}, {rf_pass['sat_lon']:.2f}, {rf_pass['sat_alt_km']:.1f} km)")

# Load vessel tracks at this time
gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')
if vessel_tracks['timestamp'].dtype == 'object':
    vessel_tracks['timestamp'] = pd.to_datetime(vessel_tracks['timestamp']).astype(int) / 1e9

# Find vessels at this time
ts_unix = rf_pass['timestamp']
for mmsi in vessel_tracks['mmsi'].unique():
    vessel_data = vessel_tracks[vessel_tracks['mmsi'] == mmsi]

    # Find closest timestamp
    time_diffs = abs(vessel_data['timestamp'] - ts_unix)
    if time_diffs.min() > 300:
        print(f"\n  ⚠️  Vessel MMSI {mmsi}: No position within 5 minutes")
        continue

    closest_idx = time_diffs.idxmin()
    vessel_pos = vessel_data.loc[closest_idx]

    print(f"\n  Vessel: {vessel_pos['vessel_name']} (MMSI: {mmsi})")
    print(f"    Position: ({vessel_pos['latitude']:.4f}, {vessel_pos['longitude']:.4f})")
    print(f"    AIS visible: {vessel_pos['ais_visible']}")
    print(f"    Time diff: {time_diffs.min():.1f} seconds")

    # Calculate elevation from vessel to satellite
    sat_ecef = lla_to_ecef(rf_pass['sat_lat'], rf_pass['sat_lon'], rf_pass['sat_alt_km'] * 1000)
    vessel_ecef = lla_to_ecef(vessel_pos['latitude'], vessel_pos['longitude'], 0.0)
    los_vector = sat_ecef - vessel_ecef
    vessel_up = vessel_ecef / np.linalg.norm(vessel_ecef)
    los_magnitude = np.linalg.norm(los_vector)
    cos_zenith = np.dot(los_vector, vessel_up) / los_magnitude
    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
    elevation = 90.0 - zenith_angle

    print(f"    Elevation from vessel: {elevation:.1f}°")

    # RF footprint check
    half_angle = 30.0
    min_elevation = 90.0 - half_angle
    in_footprint = elevation >= min_elevation

    print(f"    Min elevation for footprint: {min_elevation:.1f}°")
    print(f"    In RF footprint: {in_footprint}")

    # Distance check
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [vessel_pos['latitude'], vessel_pos['longitude'], rf_pass['sat_lat'], rf_pass['sat_lon']])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = R * c

    rf_radius_km = rf_pass['sat_alt_km'] * np.tan(np.radians(half_angle))
    print(f"    Distance: {distance_km:.1f} km")
    print(f"    RF footprint radius: {rf_radius_km:.1f} km")
    print(f"    Within radius: {distance_km <= rf_radius_km}")
