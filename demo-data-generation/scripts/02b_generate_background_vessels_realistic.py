#!/usr/bin/env python3
"""
Generate Background Vessels Using Vessel Simulation Service

Creates 498 background vessels with:
- Real Spire vessel metadata (IMO, call signs, flags, dimensions)
- Realistic port-to-port routing using Rust maritime router
- Proper AIS behavior and navigation

Timeline: July 17 20:45 - July 20 07:00 UTC (58.25 hours)
"""

import sys
import os
# Add vessel-simulation-service to path
vessel_sim_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'vessel-simulation-service')
sys.path.insert(0, vessel_sim_dir)

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import random
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Rust router
try:
    from maritime_router_rs import MaritimeRouter, RouterConfig
    logger.info("✅ Rust router loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Rust router: {e}")
    sys.exit(1)

# Load Spire vessel registry
VESSEL_REGISTRY_PATH = os.path.join(vessel_sim_dir, 'data', 'vessel_registry.parquet')
try:
    VESSEL_REGISTRY = pd.read_parquet(VESSEL_REGISTRY_PATH)
    logger.info(f"✅ Loaded vessel registry: {len(VESSEL_REGISTRY):,} vessels with real metadata")
except Exception as e:
    logger.error(f"❌ Failed to load vessel registry: {e}")
    VESSEL_REGISTRY = None

# Initialize Rust router
GEBCO_PATH = os.path.join(vessel_sim_dir, 'data', 'gebco_2025_n90.0_s0.0_w90.0_e180.0.tif')
config = RouterConfig.for_ports()
config.erosion_pixels = 0
config.max_iterations = 10_000_000

router = MaritimeRouter(
    gebco_path=GEBCO_PATH,
    min_lon=90.0,
    max_lon=180.0,
    min_lat=0.0,
    max_lat=90.0,
    resolution=0.00416,
    downsample_factor=10,
    config=config
)
logger.info("✅ Rust router initialized")

# Load ports
PORTS_CSV = os.path.join(vessel_sim_dir, 'data', 'Ports.csv')
ports_df = pd.read_csv(PORTS_CSV)
ports_df = ports_df[
    (ports_df['lon'] >= 90.0) &
    (ports_df['lon'] <= 180.0) &
    (ports_df['lat'] >= 0.0) &
    (ports_df['lat'] <= 90.0)
]

# Exclude problematic river/inland ports
EXCLUDED_PORTS = {
    'Shanghai (Pudong)', 'Shanghai (Yangshan)', 'Hai Phong', 'Saigon',
    'Dalian', 'Tianjin Xin Gang', 'Jakarta', 'Bangkok', 'Ho Chi Minh',
    'Guangzhou', 'Guangzhou (Nansha)', 'Hong Kong', 'Suzhou (Taicang)',
    'Jiangyin', 'Nanjing', 'Zhangjiagang'
}
ports_df = ports_df[~ports_df['portname'].isin(EXCLUDED_PORTS)]
ports_df = ports_df.nlargest(50, 'vessel_count_total')

logger.info(f"✅ Loaded {len(ports_df)} ports for routing")

# Track used vessels to avoid duplicates
USED_VESSEL_MMSIS = set()


def select_vessel_from_registry(vessel_type_str, vessel_id):
    """Select a random vessel from registry matching the desired type."""
    if VESSEL_REGISTRY is None:
        return None

    # Filter by vessel type
    type_vessels = VESSEL_REGISTRY[VESSEL_REGISTRY['vessel_type'] == vessel_type_str]

    # Remove already used vessels
    available_vessels = type_vessels[~type_vessels['mmsi'].isin(USED_VESSEL_MMSIS)]

    if len(available_vessels) == 0:
        logger.warning(f"No available {vessel_type_str} vessels in registry")
        return None

    # Select random vessel
    vessel = available_vessels.sample(n=1).iloc[0]
    USED_VESSEL_MMSIS.add(int(vessel['mmsi']))

    return {
        'mmsi': int(vessel['mmsi']),
        'vessel_name': vessel['name'],  # Column is 'name' not 'vessel_name'
        'imo': int(vessel['imo']) if pd.notna(vessel['imo']) else None,
        'call_sign': vessel['call_sign'] if pd.notna(vessel['call_sign']) else '',
        'flag': vessel['flag'] if pd.notna(vessel['flag']) else '',
        'ship_type_code': int(vessel['ship_type_code']) if pd.notna(vessel['ship_type_code']) else 70,
        'vessel_type': vessel_type_str,
        'length': float(vessel['length']) if pd.notna(vessel['length']) else 150.0,
        'width': float(vessel['width']) if pd.notna(vessel['width']) else 25.0,
        'draught': float(vessel['draught']) if pd.notna(vessel['draught']) else 10.0
    }


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def interpolate_route(waypoints, timestamps, speed_knots):
    """Interpolate position along route at 1-minute intervals."""
    from math import radians, sin, cos, sqrt, atan2, degrees

    positions = []

    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(waypoints)):
        d = haversine_distance(
            waypoints[i-1]['lat'], waypoints[i-1]['lon'],
            waypoints[i]['lat'], waypoints[i]['lon']
        )
        distances.append(distances[-1] + d)

    total_distance = distances[-1]
    speed_kmh = speed_knots * 1.852

    for ts in timestamps:
        time_elapsed = (ts - timestamps[0]).total_seconds() / 3600  # hours
        distance_traveled = speed_kmh * time_elapsed

        # If reached destination, stay there
        if distance_traveled >= total_distance:
            positions.append({
                'lat': waypoints[-1]['lat'],
                'lon': waypoints[-1]['lon'],
                'heading': 0,
                'sog': 0
            })
            continue

        # Find segment
        for i in range(len(distances) - 1):
            if distances[i] <= distance_traveled <= distances[i+1]:
                segment_progress = (distance_traveled - distances[i]) / (distances[i+1] - distances[i])
                lat = waypoints[i]['lat'] + segment_progress * (waypoints[i+1]['lat'] - waypoints[i]['lat'])
                lon = waypoints[i]['lon'] + segment_progress * (waypoints[i+1]['lon'] - waypoints[i]['lon'])

                # Calculate heading
                lat1, lon1 = radians(waypoints[i]['lat']), radians(waypoints[i]['lon'])
                lat2, lon2 = radians(waypoints[i+1]['lat']), radians(waypoints[i+1]['lon'])
                dlon = lon2 - lon1
                x = sin(dlon) * cos(lat2)
                y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                heading = (degrees(atan2(x, y)) + 360) % 360

                positions.append({
                    'lat': lat,
                    'lon': lon,
                    'heading': heading,
                    'sog': speed_knots
                })
                break

    return positions


def generate_vessel_voyage(vessel_id, vessel_type_config, timestamps):
    """Generate a single vessel voyage with routing."""

    # Select vessel from registry
    vessel_metadata = select_vessel_from_registry(vessel_type_config['type'], vessel_id)
    if vessel_metadata is None:
        # Fallback to synthetic
        vessel_metadata = {
            'mmsi': 400000000 + vessel_id,
            'vessel_name': f"{vessel_type_config['type'].upper()}_{vessel_id}",
            'imo': None,
            'call_sign': '',
            'flag': 'PA',
            'ship_type_code': 70,
            'vessel_type': vessel_type_config['type'],
            'length': 150.0,
            'width': 25.0,
            'draught': 10.0
        }

    # Select random origin and destination ports
    origin_port = ports_df.sample(n=1).iloc[0]
    dest_port = ports_df[ports_df['portname'] != origin_port['portname']].sample(n=1).iloc[0]

    # Get route using Rust router
    try:
        waypoints, metrics = router.route(
            start_lat=origin_port['lat'],
            start_lon=origin_port['lon'],
            end_lat=dest_port['lat'],
            end_lon=dest_port['lon'],
            auto_snap=True
        )

        if waypoints is None or len(waypoints) == 0:
            logger.warning(f"Vessel {vessel_id}: Routing failed {origin_port['portname']} -> {dest_port['portname']}")
            return []

        # Convert waypoints to dict format
        waypoints = [{'lat': wp[0], 'lon': wp[1]} for wp in waypoints]

    except Exception as e:
        logger.error(f"Vessel {vessel_id}: Router error: {e}")
        return []

    # Random speed within vessel type range
    speed_knots = random.uniform(vessel_type_config['speed_range'][0], vessel_type_config['speed_range'][1])

    # Interpolate positions
    positions = interpolate_route(waypoints, timestamps, speed_knots)

    # Build track records
    track = []
    for i, ts in enumerate(timestamps):
        if i >= len(positions):
            break

        track.append({
            'timestamp': ts.isoformat(),
            'mmsi': vessel_metadata['mmsi'],
            'imo': vessel_metadata['imo'],
            'name': vessel_metadata['vessel_name'],
            'call_sign': vessel_metadata['call_sign'],
            'flag': vessel_metadata['flag'],
            'ship_type_code': vessel_metadata['ship_type_code'],
            'vessel_type': vessel_metadata['vessel_type'],
            'length': vessel_metadata['length'],
            'width': vessel_metadata['width'],
            'draught': vessel_metadata['draught'],
            'lat': positions[i]['lat'],
            'lon': positions[i]['lon'],
            'speed_knots': positions[i]['sog'],
            'heading': positions[i]['heading'],
            'status': 'underway' if positions[i]['sog'] > 0 else 'moored',
            'ais_on': True
        })

    logger.info(f"  ✅ Vessel {vessel_id} ({vessel_metadata['vessel_name']}): {origin_port['portname']} -> {dest_port['portname']}, {len(track)} positions")
    return track


def main():
    """Generate all background vessels."""
    logger.info("=" * 70)
    logger.info("BACKGROUND VESSEL GENERATION (REALISTIC ROUTING)")
    logger.info("=" * 70)

    # Timeline
    start_time = datetime(2025, 7, 17, 20, 45, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 7, 20, 7, 0, 0, tzinfo=timezone.utc)

    # Generate timestamps (1 minute intervals)
    timestamps = []
    current = start_time
    while current <= end_time:
        timestamps.append(current)
        current += timedelta(seconds=60)

    logger.info(f"Period: {start_time.isoformat()} to {end_time.isoformat()}")
    logger.info(f"Time steps: {len(timestamps)} (1-minute resolution)")
    logger.info("")

    # Vessel type distribution
    vessel_types = {
        'cargo': {'type': 'general_cargo', 'count': 150, 'speed_range': (10, 18)},
        'tanker': {'type': 'tanker', 'count': 100, 'speed_range': (12, 16)},
        'container': {'type': 'container', 'count': 168, 'speed_range': (18, 24)},
        'bulk_carrier': {'type': 'bulk_carrier', 'count': 80, 'speed_range': (12, 15)}
    }

    logger.info("Generating background vessels:")
    for vtype, config in vessel_types.items():
        logger.info(f"  {config['type']}: {config['count']} vessels")
    logger.info("")

    # Generate all tracks
    all_tracks = []
    vessel_id = 1

    for vtype, type_config in vessel_types.items():
        logger.info(f"Generating {type_config['type']} tracks...")

        for i in range(type_config['count']):
            track = generate_vessel_voyage(vessel_id, type_config, timestamps)
            if track:
                all_tracks.extend(track)
            vessel_id += 1

    # Convert to DataFrame
    df = pd.DataFrame(all_tracks)

    logger.info("")
    logger.info(f"Total position reports: {len(df):,}")
    logger.info(f"Background vessels: {df['mmsi'].nunique()}")
    logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Export
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_path = output_dir / 'background_vessel_tracks.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Background vessel tracks exported: {output_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ BACKGROUND VESSEL GENERATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
