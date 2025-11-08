#!/usr/bin/env python3
"""
Generate Physics-Based Satellite Detections for Gold Standard Vessels

Uses actual orbital mechanics with sensor footprint geometry to detect dark vessels:
- RF: θ=30° half-angle, W=650km circular cone (continuous monitoring)
- SAR: θ=12° half-angle, W=290km rectangle perpendicular to velocity (tasked after AIS off)
- EO: θ=2° half-angle, W=20km narrow rectangle along velocity (weather-dependent)

Multi-stage tasking:
1. RF continuously monitors all vessels (record detections for AIS-off vessels only)
2. When AIS turns off → task SAR for next pass within 6 hours
3. After SAR confirmation → task EO for next clear-weather pass within 12 hours
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timezone, timedelta
from geographiclib.geodesic import Geodesic
import tensorgator as tg
from tensorgator.coord_conv import lla_to_ecef, ecef_to_lla

print("="*70)
print("PHYSICS-BASED SATELLITE DETECTION GENERATION")
print("="*70)

# Load satellite positions
positions_file = Path(__file__).parent.parent / 'outputs' / 'satellite_positions_ecef.parquet'
sat_positions = pd.read_parquet(positions_file)
print(f"\n✅ Loaded {len(sat_positions):,} satellite positions")
print(f"   Satellites: {sat_positions['satellite_id'].nunique()}")
print(f"   Time range: {sat_positions['timestamp'].min()} to {sat_positions['timestamp'].max()}")

# Load vessel tracks
gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')

# Convert timestamp to unix if needed
if vessel_tracks['timestamp'].dtype == 'object':
    vessel_tracks['timestamp'] = pd.to_datetime(vessel_tracks['timestamp']).astype(int) / 1e9

print(f"\n✅ Loaded {len(vessel_tracks):,} vessel positions")
print(f"   Vessels: {', '.join(vessel_tracks['vessel_name'].unique())}")
print(f"   Time range: {pd.to_datetime(vessel_tracks['timestamp'].min(), unit='s', utc=True)} to {pd.to_datetime(vessel_tracks['timestamp'].max(), unit='s', utc=True)}")

# Load satellite configuration
config_path = Path(__file__).parent.parent / 'config' / 'satellites.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Sensor specifications
SENSOR_SPECS = {
    'RF': {
        'half_angle_deg': 30.0,
        'swath_width_km': 650.0,
        'footprint_type': 'cone',  # Circular footprint
        'all_weather': True,
        'min_confidence': 0.70,
        'max_confidence': 0.95
    },
    'SAR': {
        'half_angle_deg': 12.0,
        'swath_width_km': 290.0,
        'footprint_type': 'rectangle_perpendicular',  # Rectangle perpendicular to velocity
        'all_weather': True,
        'min_confidence': 0.75,
        'max_confidence': 0.97
    },
    'EO': {
        'half_angle_deg': 2.0,
        'swath_width_km': 20.0,
        'footprint_type': 'rectangle_along',  # Narrow rectangle along velocity
        'all_weather': False,
        'clear_weather_required': True,
        'max_cloud_cover': 50.0,
        'min_confidence': 0.80,
        'max_confidence': 0.98
    }
}

# Geodesic calculator for accurate distance/bearing computations
geod = Geodesic.WGS84

print(f"\n📡 Sensor Specifications:")
for sensor_type, specs in SENSOR_SPECS.items():
    print(f"   {sensor_type}: θ={specs['half_angle_deg']}°, W={specs['swath_width_km']}km ({specs['footprint_type']})")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def calculate_elevation_angle(sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon):
    """Calculate elevation angle from vessel to satellite (degrees)."""
    # Convert satellite position to ECEF
    sat_ecef = lla_to_ecef(sat_lat, sat_lon, sat_alt_km * 1000)

    # Convert vessel position to ECEF (assume sea level)
    vessel_ecef = lla_to_ecef(vessel_lat, vessel_lon, 0.0)

    # Vector from vessel to satellite
    los_vector = sat_ecef - vessel_ecef

    # Local up vector at vessel (radial direction from Earth center)
    vessel_up = vessel_ecef / np.linalg.norm(vessel_ecef)

    # Elevation angle
    los_magnitude = np.linalg.norm(los_vector)
    cos_zenith = np.dot(los_vector, vessel_up) / los_magnitude
    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
    elevation_angle = 90.0 - zenith_angle

    return elevation_angle


def compute_rf_footprint_radius(sat_alt_km, half_angle_deg):
    """Compute RF footprint radius on Earth surface (km)."""
    # For a cone: ground radius = altitude * tan(half_angle)
    radius_km = sat_alt_km * np.tan(np.radians(half_angle_deg))
    return radius_km


def is_vessel_in_rf_footprint(sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon, half_angle_deg):
    """Check if vessel is within RF cone footprint."""
    # Calculate elevation angle from vessel to satellite
    elevation = calculate_elevation_angle(sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon)

    # Vessel is in footprint if elevation angle >= (90 - half_angle)
    min_elevation = 90.0 - half_angle_deg
    return elevation >= min_elevation


def compute_sar_eo_footprint_corners(sat_lat, sat_lon, sat_alt_km, ground_track_heading,
                                     swath_width_km, along_track_length_km, footprint_type):
    """
    Compute footprint corners for SAR/EO rectangular swaths.

    footprint_type:
    - 'rectangle_perpendicular': SAR swath perpendicular to velocity
    - 'rectangle_along': EO narrow swath along velocity
    """
    if footprint_type == 'rectangle_perpendicular':
        # SAR: Rectangle perpendicular to ground track
        # Width is perpendicular to heading, length is along heading
        cross_track_width = swath_width_km
        along_track_width = along_track_length_km

    elif footprint_type == 'rectangle_along':
        # EO: Narrow rectangle along ground track (push-broom scanner)
        # Width is along heading, length is perpendicular
        cross_track_width = along_track_length_km
        along_track_width = swath_width_km

    # Compute nadir point (sub-satellite point)
    nadir_lat, nadir_lon = sat_lat, sat_lon

    # Compute 4 corners using geodesic offsets
    # Corner 1: Forward + Left
    g1 = geod.Direct(nadir_lat, nadir_lon, ground_track_heading, along_track_width * 500)  # Half length forward
    g2 = geod.Direct(g1['lat2'], g1['lon2'], (ground_track_heading - 90) % 360, cross_track_width * 500)
    corner1 = (g2['lat2'], g2['lon2'])

    # Corner 2: Forward + Right
    g3 = geod.Direct(nadir_lat, nadir_lon, ground_track_heading, along_track_width * 500)
    g4 = geod.Direct(g3['lat2'], g3['lon2'], (ground_track_heading + 90) % 360, cross_track_width * 500)
    corner2 = (g4['lat2'], g4['lon2'])

    # Corner 3: Backward + Right
    g5 = geod.Direct(nadir_lat, nadir_lon, (ground_track_heading + 180) % 360, along_track_width * 500)
    g6 = geod.Direct(g5['lat2'], g5['lon2'], (ground_track_heading + 90) % 360, cross_track_width * 500)
    corner3 = (g6['lat2'], g6['lon2'])

    # Corner 4: Backward + Left
    g7 = geod.Direct(nadir_lat, nadir_lon, (ground_track_heading + 180) % 360, along_track_width * 500)
    g8 = geod.Direct(g7['lat2'], g7['lon2'], (ground_track_heading - 90) % 360, cross_track_width * 500)
    corner4 = (g8['lat2'], g8['lon2'])

    return [corner1, corner2, corner3, corner4]


def point_in_polygon(point_lat, point_lon, polygon_corners):
    """Check if point is inside polygon using ray casting algorithm."""
    x, y = point_lon, point_lat
    n = len(polygon_corners)
    inside = False

    p1_lat, p1_lon = polygon_corners[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon_corners[i % n]
        if y > min(p1_lat, p2_lat):
            if y <= max(p1_lat, p2_lat):
                if x <= max(p1_lon, p2_lon):
                    if p1_lat != p2_lat:
                        xinters = (y - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or x <= xinters:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon

    return inside


def is_vessel_in_sar_eo_footprint(sat_lat, sat_lon, sat_alt_km, ground_track_heading,
                                   vessel_lat, vessel_lon, swath_width_km,
                                   along_track_length_km, footprint_type):
    """Check if vessel is within SAR/EO rectangular footprint."""
    # Compute footprint corners
    corners = compute_sar_eo_footprint_corners(
        sat_lat, sat_lon, sat_alt_km, ground_track_heading,
        swath_width_km, along_track_length_km, footprint_type
    )

    # Check if vessel is inside polygon
    return point_in_polygon(vessel_lat, vessel_lon, corners)


def generate_weather_conditions(timestamp_dt):
    """Generate weather conditions for a given timestamp."""
    # Simplified weather model
    # Daytime (6am-6pm UTC): Clear weather (5-20% cloud cover)
    # Nighttime: Partly cloudy (30-60% cloud cover)

    hour = timestamp_dt.hour
    if 6 <= hour < 18:
        weather = 'clear'
        cloud_cover = np.random.uniform(5, 20)
    else:
        weather = 'partly_cloudy'
        cloud_cover = np.random.uniform(30, 60)

    return weather, cloud_cover


def compute_detection_confidence(sensor_type, elevation_deg, ais_visible):
    """Compute detection confidence based on geometry and AIS status."""
    specs = SENSOR_SPECS[sensor_type]

    # Base confidence from elevation (higher elevation = better geometry)
    # Normalize elevation to 30-90 degrees
    elevation_normalized = (elevation_deg - 30.0) / 60.0
    elevation_factor = np.clip(elevation_normalized, 0.0, 1.0)

    # AIS status affects confidence
    if ais_visible:
        # Higher confidence when AIS is on (can correlate)
        base_conf = specs['max_confidence']
    else:
        # Lower confidence during dark period
        base_conf = (specs['min_confidence'] + specs['max_confidence']) / 2

    # Final confidence with elevation factor
    confidence = base_conf * (0.8 + 0.2 * elevation_factor)

    # Add small random variation
    confidence += np.random.uniform(-0.03, 0.03)

    return np.clip(confidence, specs['min_confidence'], specs['max_confidence'])


# Generate detections
print(f"\n🛰️  Generating physics-based detections...")
detections = []

# Track AIS off events for tasking
ais_off_events = {}  # {mmsi: timestamp_when_ais_turned_off}
sar_tasked_vessels = {}  # {mmsi: last_sar_task_time}
eo_tasked_vessels = {}  # {mmsi: last_eo_task_time}

# Group satellite positions by timestamp for efficient processing
sat_positions['timestamp_unix'] = pd.to_datetime(sat_positions['timestamp']).astype(int) / 1e9
sat_by_time = sat_positions.groupby('timestamp_unix')

# Get unique timestamps sorted
unique_timestamps = sorted(sat_positions['timestamp_unix'].unique())

print(f"   Processing {len(unique_timestamps):,} timesteps...")

# Load pre-computed satellite passes (elevation > 0)
passes_file = Path(__file__).parent.parent / 'outputs' / 'satellite_passes_during_dark.parquet'
if passes_file.exists():
    sat_passes = pd.read_parquet(passes_file)
    # Filter to only high-quality passes (elevation > 30°)
    sat_passes = sat_passes[sat_passes['elevation_deg'] > 30]
    print(f"   Using {len(sat_passes)} pre-computed satellite passes (elevation > 30°)")

    # Get unique timestamps from passes
    sampled_timestamps = sorted(sat_passes['timestamp'].unique())
else:
    # Fallback: sample every 10th timestep
    print(f"   ⚠️  No pre-computed passes found, using full timestamp sampling")
    sampled_timestamps = unique_timestamps[::10]

print(f"   Processing {len(sampled_timestamps):,} timesteps...")

for idx, ts_unix in enumerate(sampled_timestamps):
    if idx % 50 == 0:
        progress_pct = 100 * idx / len(sampled_timestamps)
        print(f"   Progress: {progress_pct:.1f}% ({idx}/{len(sampled_timestamps)} timesteps)")

    ts_dt = pd.to_datetime(ts_unix, unit='s', utc=True)

    # Get all satellites at this timestamp
    sats_at_time = sat_positions[sat_positions['timestamp_unix'] == ts_unix]

    # Get all vessels at this timestamp (find closest vessel positions)
    vessel_positions_at_time = []
    for mmsi in vessel_tracks['mmsi'].unique():
        vessel_data = vessel_tracks[vessel_tracks['mmsi'] == mmsi]

        # Find closest timestamp
        time_diffs = abs(vessel_data['timestamp'] - ts_unix)
        if time_diffs.min() > 300:  # Skip if >5 minutes away
            continue

        closest_idx = time_diffs.idxmin()
        vessel_pos = vessel_data.loc[closest_idx]
        vessel_positions_at_time.append(vessel_pos)

        # Track AIS off events
        if not vessel_pos['ais_visible'] and mmsi not in ais_off_events:
            ais_off_events[mmsi] = ts_unix
            print(f"\n   🚨 AIS OFF detected: {vessel_pos['vessel_name']} at {ts_dt.isoformat()}")
        elif vessel_pos['ais_visible'] and mmsi in ais_off_events:
            # AIS came back on
            del ais_off_events[mmsi]

    # Check each satellite for detections
    for _, sat in sats_at_time.iterrows():
        sat_id = sat['satellite_id']
        sensor_type = sat['sensor_type']
        sat_lat = sat['latitude']
        sat_lon = sat['longitude']
        sat_alt_km = sat['altitude_km']
        ground_track_heading = sat['ground_track_heading']
        velocity_km_s = sat['velocity_km_s']

        specs = SENSOR_SPECS[sensor_type]

        # For SAR/EO, compute along-track length from velocity and dwell time
        # Assume 10-second dwell time per swath
        dwell_time_sec = 10.0
        along_track_length_km = velocity_km_s * dwell_time_sec

        # Check each vessel
        for vessel_pos in vessel_positions_at_time:
            vessel_mmsi = vessel_pos['mmsi']
            vessel_name = vessel_pos['vessel_name']
            vessel_lat = vessel_pos['latitude']
            vessel_lon = vessel_pos['longitude']
            ais_visible = vessel_pos['ais_visible']

            # RF: Always monitors, but only record detections for dark vessels
            if sensor_type == 'RF':
                in_footprint = is_vessel_in_rf_footprint(
                    sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon, specs['half_angle_deg']
                )

                # Only record RF detection if AIS is OFF
                if in_footprint and not ais_visible:
                    elevation = calculate_elevation_angle(sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon)
                    if elevation < 30:  # Minimum elevation threshold
                        continue

                    confidence = compute_detection_confidence(sensor_type, elevation, ais_visible)
                    weather, cloud_cover = generate_weather_conditions(ts_dt)

                    detections.append({
                        'timestamp': ts_unix,
                        'event_type': 'satellite_detection_rf',
                        'vessel_mmsi': int(vessel_mmsi),
                        'vessel_name': vessel_name,
                        'satellite_id': sat_id,
                        'sensor_type': sensor_type,
                        'detection_confidence': confidence,
                        'lat': vessel_lat,
                        'lon': vessel_lon,
                        'latitude': vessel_lat,
                        'longitude': vessel_lon,
                        'metadata': {
                            'elevation_deg': float(elevation),
                            'cloud_cover_pct': float(cloud_cover),
                            'weather': weather,
                            'ais_visible': bool(ais_visible),
                            'satellite_altitude_km': float(sat_alt_km),
                            'satellite_velocity_km_s': float(velocity_km_s)
                        }
                    })

            # SAR: Tasked when AIS turns off (only detect dark vessels)
            elif sensor_type == 'SAR':
                # Only task SAR if vessel is dark AND hasn't been SAR-tasked recently
                if not ais_visible:
                    # Check if we should task SAR (within 6 hours of AIS off event)
                    last_sar_time = sar_tasked_vessels.get(vessel_mmsi, 0)
                    time_since_last_sar = ts_unix - last_sar_time

                    if time_since_last_sar > 3600:  # At least 1 hour between SAR taskings
                        in_footprint = is_vessel_in_sar_eo_footprint(
                            sat_lat, sat_lon, sat_alt_km, ground_track_heading,
                            vessel_lat, vessel_lon, specs['swath_width_km'],
                            along_track_length_km, specs['footprint_type']
                        )

                        if in_footprint:
                            elevation = calculate_elevation_angle(sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon)
                            if elevation < 30:
                                continue

                            confidence = compute_detection_confidence(sensor_type, elevation, ais_visible)
                            weather, cloud_cover = generate_weather_conditions(ts_dt)

                            detections.append({
                                'timestamp': ts_unix,
                                'event_type': 'satellite_detection_sar',
                                'vessel_mmsi': int(vessel_mmsi),
                                'vessel_name': vessel_name,
                                'satellite_id': sat_id,
                                'sensor_type': sensor_type,
                                'detection_confidence': confidence,
                                'lat': vessel_lat,
                                'lon': vessel_lon,
                                'latitude': vessel_lat,
                                'longitude': vessel_lon,
                                'metadata': {
                                    'elevation_deg': float(elevation),
                                    'cloud_cover_pct': float(cloud_cover),
                                    'weather': weather,
                                    'ais_visible': bool(ais_visible),
                                    'satellite_altitude_km': float(sat_alt_km),
                                    'satellite_velocity_km_s': float(velocity_km_s),
                                    'swath_width_km': float(specs['swath_width_km'])
                                }
                            })

                            sar_tasked_vessels[vessel_mmsi] = ts_unix

            # EO: Tasked after SAR confirmation (clear weather only)
            elif sensor_type == 'EO':
                # Only task EO if vessel is dark AND SAR has confirmed
                if not ais_visible and vessel_mmsi in sar_tasked_vessels:
                    # Check weather
                    weather, cloud_cover = generate_weather_conditions(ts_dt)

                    if cloud_cover > specs['max_cloud_cover']:
                        continue

                    # Check if we should task EO (within 12 hours of SAR)
                    last_eo_time = eo_tasked_vessels.get(vessel_mmsi, 0)
                    time_since_last_eo = ts_unix - last_eo_time

                    if time_since_last_eo > 7200:  # At least 2 hours between EO taskings
                        in_footprint = is_vessel_in_sar_eo_footprint(
                            sat_lat, sat_lon, sat_alt_km, ground_track_heading,
                            vessel_lat, vessel_lon, specs['swath_width_km'],
                            along_track_length_km, specs['footprint_type']
                        )

                        if in_footprint:
                            elevation = calculate_elevation_angle(sat_lat, sat_lon, sat_alt_km, vessel_lat, vessel_lon)
                            if elevation < 30:
                                continue

                            confidence = compute_detection_confidence(sensor_type, elevation, ais_visible)

                            detections.append({
                                'timestamp': ts_unix,
                                'event_type': 'satellite_detection_eo',
                                'vessel_mmsi': int(vessel_mmsi),
                                'vessel_name': vessel_name,
                                'satellite_id': sat_id,
                                'sensor_type': sensor_type,
                                'detection_confidence': confidence,
                                'lat': vessel_lat,
                                'lon': vessel_lon,
                                'latitude': vessel_lat,
                                'longitude': vessel_lon,
                                'metadata': {
                                    'elevation_deg': float(elevation),
                                    'cloud_cover_pct': float(cloud_cover),
                                    'weather': weather,
                                    'ais_visible': bool(ais_visible),
                                    'satellite_altitude_km': float(sat_alt_km),
                                    'satellite_velocity_km_s': float(velocity_km_s),
                                    'swath_width_km': float(specs['swath_width_km'])
                                }
                            })

                            eo_tasked_vessels[vessel_mmsi] = ts_unix

print(f"\n✅ Detection generation complete!")
print(f"   Total detections: {len(detections)}")

# Convert to DataFrame
detections_df = pd.DataFrame(detections)

if len(detections_df) > 0:
    print(f"\n📊 Detection Statistics:")
    print(f"   RF detections: {len(detections_df[detections_df['sensor_type'] == 'RF'])}")
    print(f"   SAR detections: {len(detections_df[detections_df['sensor_type'] == 'SAR'])}")
    print(f"   EO detections: {len(detections_df[detections_df['sensor_type'] == 'EO'])}")
    print(f"   Vessels detected: {detections_df['vessel_mmsi'].nunique()}")

    # Save detections
    output_file = gold_standard_dir / 'physics_based_detections.parquet'
    detections_df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved physics-based detections to: {output_file}")

    # Now combine with existing vessel behavior events
    existing_events = pd.read_parquet(gold_standard_dir / 'ground_truth_events_BACKUP.parquet'
                                      if (gold_standard_dir / 'ground_truth_events_BACKUP.parquet').exists()
                                      else gold_standard_dir / 'ground_truth_events.parquet')

    # Convert existing events to standard format
    combined_events = []
    for _, event in existing_events.iterrows():
        combined_events.append({
            'timestamp': pd.to_datetime(event['event_time']).timestamp() if 'event_time' in event else event.get('timestamp'),
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

    # Create backup if doesn't exist
    backup_file = gold_standard_dir / 'ground_truth_events_BACKUP.parquet'
    if not backup_file.exists():
        existing_events.to_parquet(backup_file, index=False)
        print(f"   Created backup: {backup_file}")

    # Save combined events
    combined_df.to_parquet(gold_standard_dir / 'ground_truth_events.parquet', index=False)
    print(f"   Updated: {gold_standard_dir / 'ground_truth_events.parquet'}")
    print(f"\n📦 Combined Events Summary:")
    print(f"   Total events: {len(combined_df)}")
    print(f"   Vessel behavior events: {len(existing_events)}")
    print(f"   Satellite detections: {len(detections_df)}")
else:
    print("\n⚠️  No detections generated - check vessel tracks and satellite coverage")

print("\n✅ Physics-based detection generation complete!")
