#!/usr/bin/env python3
"""
Generate Background Vessel Tracks for Demo Data

Creates 498 background vessels to provide realistic maritime context:
- Cargo ships transiting shipping lanes
- Fishing vessels operating in the region
- Tankers on various routes
- Container ships, bulk carriers, etc.

All vessels use 1-minute time resolution during dark period.
Region: East China Sea around 32.5°N, 126.0°E
Timeline: July 17 20:45 - July 20 07:00 UTC (58.25 hours)
"""

import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load simulation configuration."""
    config_dir = Path(__file__).parent.parent / 'config'

    with open(config_dir / 'satellites.yaml') as f:
        sat_config = yaml.safe_load(f)

    return sat_config['simulation']


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def bearing(lat1, lon1, lat2, lon2):
    """Calculate initial bearing from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    initial_bearing = np.arctan2(x, y)
    return (np.degrees(initial_bearing) + 360) % 360


def destination_point(lat, lon, bearing_deg, distance_km):
    """Calculate destination point given start point, bearing and distance."""
    R = 6371  # Earth radius in km

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    brng = np.radians(bearing_deg)
    d = distance_km

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) +
                     np.cos(lat1) * np.sin(d/R) * np.cos(brng))

    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d/R) * np.cos(lat1),
                             np.cos(d/R) - np.sin(lat1) * np.sin(lat2))

    return np.degrees(lat2), np.degrees(lon2)


def generate_vessel_config(vessel_id, vessel_type_config, region_center):
    """Generate configuration for a single background vessel."""

    # Generate unique MMSI (using 4xxxxx range for background vessels)
    mmsi = 400000000 + vessel_id

    # Random start position around region (within 500km)
    start_distance = np.random.uniform(50, 500)
    start_bearing = np.random.uniform(0, 360)
    start_lat, start_lon = destination_point(
        region_center['lat'],
        region_center['lon'],
        start_bearing,
        start_distance
    )

    # Random end position (different direction)
    end_bearing = (start_bearing + np.random.uniform(120, 240)) % 360
    end_distance = np.random.uniform(50, 500)
    end_lat, end_lon = destination_point(
        region_center['lat'],
        region_center['lon'],
        end_bearing,
        end_distance
    )

    # Speed based on vessel type
    speed_knots = np.random.uniform(
        vessel_type_config['speed_range'][0],
        vessel_type_config['speed_range'][1]
    )

    # Generate vessel name
    vessel_names = vessel_type_config.get('names', [f"{vessel_type_config['type']} {vessel_id}"])
    name = np.random.choice(vessel_names)

    # Random flag
    flags = ['CHN', 'JPN', 'KOR', 'PAN', 'LBR', 'MHL', 'SGP', 'HKG', 'TWN']
    flag = np.random.choice(flags)

    return {
        'mmsi': mmsi,
        'name': f"{name} {vessel_id}",
        'vessel_type': vessel_type_config['type'],
        'flag': flag,
        'start_lat': start_lat,
        'start_lon': start_lon,
        'end_lat': end_lat,
        'end_lon': end_lon,
        'speed_knots': speed_knots,
        'behavior': vessel_type_config.get('behavior', 'transit')
    }


def generate_transit_track(vessel_config, timestamps):
    """Generate track for vessel in transit from start to end."""

    track = []

    # Calculate route parameters
    start_lat = vessel_config['start_lat']
    start_lon = vessel_config['start_lon']
    end_lat = vessel_config['end_lat']
    end_lon = vessel_config['end_lon']

    total_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    route_bearing = bearing(start_lat, start_lon, end_lat, end_lon)

    # Speed in km/h
    speed_kmh = vessel_config['speed_knots'] * 1.852

    for ts in timestamps:
        # Calculate position based on time elapsed
        time_elapsed = (ts - timestamps[0]).total_seconds() / 3600  # hours
        distance_traveled = min(speed_kmh * time_elapsed, total_distance)

        if distance_traveled >= total_distance:
            # Reached destination, continue past it
            extra_distance = distance_traveled - total_distance
            lat, lon = destination_point(end_lat, end_lon, route_bearing, extra_distance)
        else:
            # Still in transit
            lat, lon = destination_point(start_lat, start_lon, route_bearing, distance_traveled)

        track.append({
            'timestamp': ts.isoformat(),
            'mmsi': vessel_config['mmsi'],
            'name': vessel_config['name'],
            'lat': lat,
            'lon': lon,
            'speed_knots': vessel_config['speed_knots'],
            'heading': route_bearing,
            'vessel_type': vessel_config['vessel_type'],
            'flag': vessel_config['flag'],
            'status': 'underway',
            'ais_on': True
        })

    return track


def generate_fishing_track(vessel_config, timestamps):
    """Generate track for fishing vessel (random walk pattern)."""

    track = []

    # Start position
    current_lat = vessel_config['start_lat']
    current_lon = vessel_config['start_lon']

    # Fishing vessels move slowly with random changes
    base_speed = vessel_config['speed_knots']

    for i, ts in enumerate(timestamps):
        # Every 30 minutes, change direction and speed
        if i % 30 == 0:
            current_heading = np.random.uniform(0, 360)
            current_speed = np.random.uniform(base_speed * 0.5, base_speed * 1.5)

        # Move vessel
        if i > 0:
            distance_km = (current_speed * 1.852) / 60  # distance in 1 minute
            current_lat, current_lon = destination_point(
                current_lat, current_lon, current_heading, distance_km
            )

        track.append({
            'timestamp': ts.isoformat(),
            'mmsi': vessel_config['mmsi'],
            'name': vessel_config['name'],
            'lat': current_lat,
            'lon': current_lon,
            'speed_knots': current_speed,
            'heading': current_heading,
            'vessel_type': vessel_config['vessel_type'],
            'flag': vessel_config['flag'],
            'status': 'fishing',
            'ais_on': True
        })

    return track


def main():
    """Generate all background vessel tracks."""
    logger.info("=" * 70)
    logger.info("BACKGROUND VESSEL TRACK GENERATION")
    logger.info("=" * 70)

    # Load configuration
    sim_config = load_config()

    # Parse times
    start_time = datetime.fromisoformat(sim_config['start_time'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(sim_config['end_time'].replace('Z', '+00:00'))

    # Generate timestamps (1 minute intervals)
    timestamps = []
    current = start_time
    while current <= end_time:
        timestamps.append(current)
        current += timedelta(seconds=60)

    logger.info(f"Simulation period: {sim_config['start_time']} to {sim_config['end_time']}")
    logger.info(f"Duration: {sim_config['duration_hours']} hours")
    logger.info(f"Time steps: {len(timestamps)} (1-minute resolution)")
    logger.info("")

    # Region center (STS transfer area)
    region_center = {'lat': 32.5, 'lon': 126.0}

    # Define vessel type distributions
    vessel_types = {
        'cargo': {
            'type': 'Cargo',
            'count': 150,
            'speed_range': (10, 18),
            'behavior': 'transit',
            'names': ['OCEAN CARRIER', 'PACIFIC TRADER', 'SEA MERCHANT', 'BLUE CARGO']
        },
        'tanker': {
            'type': 'Tanker',
            'count': 100,
            'speed_range': (12, 16),
            'behavior': 'transit',
            'names': ['OCEAN SPIRIT', 'PACIFIC PRIDE', 'SEA VENTURE', 'BLUE HORIZON']
        },
        'container': {
            'type': 'Container Ship',
            'count': 80,
            'speed_range': (18, 24),
            'behavior': 'transit',
            'names': ['MAERSK LINE', 'COSCO SHIP', 'EVERGREEN', 'OCEAN CONTAINER']
        },
        'bulk_carrier': {
            'type': 'Bulk Carrier',
            'count': 70,
            'speed_range': (12, 15),
            'behavior': 'transit',
            'names': ['BULK MASTER', 'OCEAN HAULER', 'SEA TITAN', 'PACIFIC BULK']
        },
        'fishing': {
            'type': 'Fishing',
            'count': 98,
            'speed_range': (2, 8),
            'behavior': 'fishing',
            'names': ['LUCKY STAR', 'OCEAN HARVEST', 'SEA BOUNTY', 'PACIFIC FISHER']
        }
    }

    logger.info("Generating background vessels:")
    for vtype, config in vessel_types.items():
        logger.info(f"  {config['type']}: {config['count']} vessels")
    logger.info("")

    # Generate all vessel tracks
    all_tracks = []
    vessel_id = 1

    for vtype, type_config in vessel_types.items():
        logger.info(f"Generating {type_config['type']} tracks...")

        for i in range(type_config['count']):
            vessel_config = generate_vessel_config(vessel_id, type_config, region_center)

            # Generate track based on behavior
            if type_config['behavior'] == 'fishing':
                track = generate_fishing_track(vessel_config, timestamps)
            else:
                track = generate_transit_track(vessel_config, timestamps)

            all_tracks.extend(track)
            vessel_id += 1

        logger.info(f"  ✅ Generated {type_config['count']} {type_config['type']} vessels")

    # Convert to DataFrame
    df = pd.DataFrame(all_tracks)

    logger.info("")
    logger.info(f"Total position reports: {len(df):,}")
    logger.info(f"Background vessels: {df['mmsi'].nunique()}")
    logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Export to parquet
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_path = output_dir / 'background_vessel_tracks.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Background vessel tracks exported: {output_path}")

    # Export summary
    summary = {
        'simulation_period': {
            'start': sim_config['start_time'],
            'end': sim_config['end_time'],
            'duration_hours': sim_config['duration_hours']
        },
        'vessel_types': {
            vtype: {
                'count': config['count'],
                'speed_range_knots': config['speed_range'],
                'behavior': config['behavior']
            }
            for vtype, config in vessel_types.items()
        },
        'statistics': {
            'total_vessels': df['mmsi'].nunique(),
            'total_positions': len(df),
            'region_center': region_center
        }
    }

    summary_path = output_dir / 'background_vessels_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Summary exported: {summary_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ BACKGROUND VESSEL GENERATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
