"""
Propagate ELK satellite constellation orbits using tensorgator
Compute velocity vectors and ground track headings for gold_standard vessel detection
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timezone, timedelta
import tensorgator as tg
from tensorgator.coord_conv import ecef_to_lla

print("="*70)
print("ORBITAL PROPAGATION FOR GOLD STANDARD DETECTION")
print("="*70)

# Load constellation configuration
config_path = Path(__file__).parent.parent / 'config' / 'satellites.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"\n✅ Loaded configuration from {config_path}")

# Extract satellite orbital elements
satellites = []
satellite_names = []
satellite_types = []

for sensor_type in ['rf', 'sar', 'eo']:
    for sat_config in config['constellation'][sensor_type]:
        oe = sat_config['orbital_elements']

        # Convert to Keplerian elements for tensorgator (in RADIANS and METERS)
        a = oe['semi_major_axis_m']
        e = oe['eccentricity']
        inc = np.radians(oe['inclination_deg'])
        raan = np.radians(oe['raan_deg'])
        argp = np.radians(oe['arg_perigee_deg'])
        M = np.radians(oe['mean_anomaly_deg'])

        satellites.append([a, e, inc, raan, argp, M])
        satellite_names.append(sat_config['name'])
        satellite_types.append(sat_config['sensor_type'])

constellation = np.array(satellites)
print(f"\n✅ Loaded {len(satellites)} satellites:")
for name, stype in zip(satellite_names, satellite_types):
    print(f"   {name} ({stype})")

# Propagation time range (14 days over gold_standard vessel track period)
start_time_str = config['propagation']['start_time']
start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
duration_days = config['propagation']['duration_days']
time_step_sec = config['propagation']['time_step_seconds']

# Create time array (seconds since epoch)
total_seconds = duration_days * 24 * 3600
num_timesteps = int(total_seconds / time_step_sec) + 1
times = np.arange(0, total_seconds + time_step_sec, time_step_sec)

print(f"\n📅 Propagation period:")
print(f"   Start: {start_time.isoformat()}")
print(f"   Duration: {duration_days} days")
print(f"   Time step: {time_step_sec} seconds")
print(f"   Timesteps: {num_timesteps:,}")

# Propagate orbits using tensorgator
print(f"\n🛰️  Propagating {len(satellites)} satellite orbits...")
positions_ecef = tg.satellite_positions(
    times,
    constellation,
    backend='cpu',
    return_frame='ecef',
    input_type='kepler'
)
print(f"✅ Propagation complete!")
print(f"   Position array shape: {positions_ecef.shape}")  # (9, num_timesteps, 3)

# Compute velocity vectors from position deltas
print(f"\n🏃 Computing velocity vectors...")
dt = time_step_sec  # seconds
velocities_ecef = np.diff(positions_ecef, axis=1) / dt

# Pad last timestep to match positions shape
velocities_ecef = np.concatenate([
    velocities_ecef,
    velocities_ecef[:, -1:, :]
], axis=1)

velocity_magnitudes = np.linalg.norm(velocities_ecef, axis=2)
print(f"✅ Velocity computation complete!")
print(f"   Average orbital velocity: {np.mean(velocity_magnitudes):.2f} m/s ({np.mean(velocity_magnitudes)/1000:.2f} km/s)")

# Convert positions to LLA for ground track heading computation
print(f"\n🌍 Converting to LLA coordinates...")
positions_lla = np.zeros_like(positions_ecef)
for sat_idx in range(len(satellites)):
    for t_idx in range(len(times)):
        pos_ecef = positions_ecef[sat_idx, t_idx, :]
        lat, lon, alt = ecef_to_lla(pos_ecef)
        positions_lla[sat_idx, t_idx, :] = [lat, lon, alt]

print(f"✅ LLA conversion complete!")

# Compute ground track heading (azimuth of velocity vector)
print(f"\n🧭 Computing ground track headings...")

def compute_ground_track_heading(sat_lla, vel_ecef):
    """
    Compute azimuth of velocity vector projected onto Earth surface
    Returns heading in degrees (0-360, 0=North)
    """
    lat, lon, alt = sat_lla
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Local ENU (East-North-Up) basis vectors at satellite position
    # East direction
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])

    # North direction
    north = np.array([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad)
    ])

    # Project velocity onto horizontal plane (East-North)
    vel_east = np.dot(vel_ecef, east)
    vel_north = np.dot(vel_ecef, north)

    # Compute azimuth from north (0-360°)
    heading = np.degrees(np.arctan2(vel_east, vel_north)) % 360

    return heading

ground_track_headings = np.zeros((len(satellites), len(times)))
for sat_idx in range(len(satellites)):
    for t_idx in range(len(times)):
        sat_lla = positions_lla[sat_idx, t_idx, :]
        vel_ecef = velocities_ecef[sat_idx, t_idx, :]
        ground_track_headings[sat_idx, t_idx] = compute_ground_track_heading(sat_lla, vel_ecef)

print(f"✅ Ground track heading computation complete!")

# Prepare data for export
print(f"\n💾 Preparing data for export...")

# Flatten arrays for DataFrame
records = []
for sat_idx in range(len(satellites)):
    for t_idx in range(len(times)):
        timestamp = start_time + timedelta(seconds=float(times[t_idx]))

        records.append({
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': times[t_idx],
            'satellite_id': satellite_names[sat_idx],
            'sensor_type': satellite_types[sat_idx],

            # ECEF positions (meters)
            'x_ecef': positions_ecef[sat_idx, t_idx, 0],
            'y_ecef': positions_ecef[sat_idx, t_idx, 1],
            'z_ecef': positions_ecef[sat_idx, t_idx, 2],

            # ECEF velocities (m/s)
            'vx_ecef': velocities_ecef[sat_idx, t_idx, 0],
            'vy_ecef': velocities_ecef[sat_idx, t_idx, 1],
            'vz_ecef': velocities_ecef[sat_idx, t_idx, 2],

            # LLA coordinates
            'latitude': positions_lla[sat_idx, t_idx, 0],
            'longitude': positions_lla[sat_idx, t_idx, 1],
            'altitude_m': positions_lla[sat_idx, t_idx, 2],
            'altitude_km': positions_lla[sat_idx, t_idx, 2] / 1000,

            # Derived quantities
            'velocity_km_s': velocity_magnitudes[sat_idx, t_idx] / 1000,
            'ground_track_heading': ground_track_headings[sat_idx, t_idx]
        })

df = pd.DataFrame(records)
print(f"✅ DataFrame created: {len(df):,} rows")

# Save to parquet
output_dir = Path(__file__).parent.parent / 'outputs'
output_dir.mkdir(exist_ok=True)

output_path = output_dir / 'satellite_positions_ecef.parquet'
df.to_parquet(output_path, index=False)
print(f"\n✅ Saved satellite positions to: {output_path}")
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save metadata
metadata = {
    'num_satellites': len(satellites),
    'satellite_names': satellite_names,
    'satellite_types': satellite_types,
    'start_time': start_time.isoformat(),
    'duration_days': duration_days,
    'time_step_seconds': time_step_sec,
    'num_timesteps': num_timesteps,
    'total_records': len(df),
    'avg_velocity_km_s': float(np.mean(velocity_magnitudes) / 1000)
}

metadata_path = output_dir / 'satellite_positions_metadata.yaml'
with open(metadata_path, 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)
print(f"✅ Saved metadata to: {metadata_path}")

# Summary statistics
print(f"\n📊 Summary Statistics:")
print(f"   Satellites: {len(satellites)}")
print(f"   Time range: {start_time.isoformat()} to {(start_time + timedelta(days=duration_days)).isoformat()}")
print(f"   Total positions: {len(df):,}")
print(f"   Avg velocity: {np.mean(velocity_magnitudes)/1000:.2f} km/s")
print(f"   Altitude range: {df['altitude_km'].min():.1f} - {df['altitude_km'].max():.1f} km")

print("\n✅ Orbital propagation complete!")
