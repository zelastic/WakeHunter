#!/usr/bin/env python3
"""
Generate Vessel Tracks for Dark Period STS Transfer Demo

Creates 1-minute resolution vessel tracks for:
- 2 gold standard vessels (YUE CHI + sanctioned vessel) performing STS transfer
- 498 background vessels providing context

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
    """Load vessel and simulation configuration."""
    config_dir = Path(__file__).parent.parent / 'config'

    with open(config_dir / 'satellites.yaml') as f:
        sat_config = yaml.safe_load(f)

    with open(config_dir / 'vessels.yaml') as f:
        vessel_config = yaml.safe_load(f)

    return sat_config['simulation'], vessel_config


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


def generate_gold_vessel_track(vessel_config, sim_config, sts_site):
    """
    Generate track for a gold standard vessel involved in STS transfer.

    Phases:
    1. Approach: From start location to STS site
    2. Transfer: Loiter at STS site for 6 hours
    3. Departure: Escape on specified heading
    """
    logger.info(f"Generating track for {vessel_config['name']}")

    # Parse times
    start_time = datetime.fromisoformat(sim_config['start_time'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(sim_config['end_time'].replace('Z', '+00:00'))
    approach_start = datetime.fromisoformat(vessel_config['sts_behavior']['approach_start'].replace('Z', '+00:00'))
    arrival = datetime.fromisoformat(vessel_config['sts_behavior']['arrival_at_sts'].replace('Z', '+00:00'))
    departure = datetime.fromisoformat(vessel_config['sts_behavior']['departure'].replace('Z', '+00:00'))

    # Generate timestamps (1 minute intervals)
    timestamps = []
    current = start_time
    while current <= end_time:
        timestamps.append(current)
        current += timedelta(seconds=60)

    # Initialize track data
    track = []

    # Starting position
    start_lat = vessel_config['start_location']['lat']
    start_lon = vessel_config['start_location']['lon']
    sts_lat = sts_site['lat']
    sts_lon = sts_site['lon']

    # Calculate approach parameters
    approach_distance = haversine_distance(start_lat, start_lon, sts_lat, sts_lon)
    approach_duration = (arrival - approach_start).total_seconds() / 3600  # hours
    approach_speed = approach_distance / approach_duration  # km/h
    approach_bearing = bearing(start_lat, start_lon, sts_lat, sts_lon)

    logger.info(f"  Approach: {approach_distance:.1f}km over {approach_duration:.1f}h at {approach_speed:.1f}km/h")

    for ts in timestamps:
        if ts < approach_start:
            # Phase 0: Before approach (loiter at start location with small random walk)
            lat = start_lat + np.random.normal(0, 0.01)
            lon = start_lon + np.random.normal(0, 0.01)
            speed_knots = 0
            heading = 0
            status = "loitering"

        elif ts < arrival:
            # Phase 1: Approaching STS site
            time_since_approach = (ts - approach_start).total_seconds() / 3600
            progress = time_since_approach / approach_duration
            progress = min(progress, 1.0)

            # Linear interpolation toward STS site
            distance_traveled = approach_distance * progress
            lat, lon = destination_point(start_lat, start_lon, approach_bearing, distance_traveled)

            speed_knots = approach_speed * 0.539957  # km/h to knots
            heading = approach_bearing
            status = "underway"

        elif ts < departure:
            # Phase 2: At STS site (transfer in progress)
            # Small random walk to simulate maneuvering during transfer
            lat = sts_lat + np.random.normal(0, 0.002)
            lon = sts_lon + np.random.normal(0, 0.002)
            speed_knots = 0
            heading = 0
            status = "moored"  # or "not under command" for STS

        else:
            # Phase 3: Departing from STS site
            time_since_departure = (ts - departure).total_seconds() / 3600
            escape_heading = vessel_config['sts_behavior']['escape_heading_deg']
            escape_speed_knots = 12  # 12 knots escape speed
            escape_speed_kmh = escape_speed_knots / 0.539957

            distance_traveled = escape_speed_kmh * time_since_departure
            lat, lon = destination_point(sts_lat, sts_lon, escape_heading, distance_traveled)

            speed_knots = escape_speed_knots
            heading = escape_heading
            status = "underway"

        # Add AIS on/off logic for sanctioned vessel
        ais_on = True
        if 'dark_ship_behavior' in vessel_config:
            if vessel_config['dark_ship_behavior'].get('ais_spoofing'):
                for period in vessel_config['dark_ship_behavior']['ais_off_periods']:
                    ais_off_start = datetime.fromisoformat(period['start'].replace('Z', '+00:00'))
                    ais_off_end = datetime.fromisoformat(period['end'].replace('Z', '+00:00'))
                    if ais_off_start <= ts <= ais_off_end:
                        ais_on = False
                        break

        track.append({
            'timestamp': ts.isoformat(),
            'mmsi': vessel_config['mmsi'],
            'imo': vessel_config.get('imo'),
            'name': vessel_config['name'],
            'lat': lat,
            'lon': lon,
            'speed_knots': speed_knots,
            'heading': heading,
            'vessel_type': vessel_config['vessel_type'],
            'flag': vessel_config['flag'],
            'status': status,
            'ais_on': ais_on
        })

    logger.info(f"  Generated {len(track)} position reports")
    return track


def main():
    """Generate all vessel tracks."""
    logger.info("=" * 70)
    logger.info("VESSEL TRACK GENERATION")
    logger.info("=" * 70)

    # Load configuration
    sim_config, vessel_config = load_config()
    sts_site = vessel_config['sts_site']

    logger.info(f"Simulation period: {sim_config['start_time']} to {sim_config['end_time']}")
    logger.info(f"Duration: {sim_config['duration_hours']} hours")
    logger.info(f"Time step: {sim_config['time_step_seconds']} seconds")
    logger.info("")

    # Generate gold vessel tracks
    logger.info("Generating gold standard vessel tracks...")
    all_tracks = []

    for vessel in vessel_config['gold_vessels']:
        track = generate_gold_vessel_track(vessel, sim_config, sts_site)
        all_tracks.extend(track)

    # Convert to DataFrame
    df = pd.DataFrame(all_tracks)

    logger.info("")
    logger.info(f"Total position reports: {len(df):,}")
    logger.info(f"Vessels: {df['mmsi'].nunique()}")
    logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Export to parquet
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_path = output_dir / 'vessel_tracks_dark_period.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Vessel tracks exported: {output_path}")

    # Also export to JSON for easy inspection
    json_path = output_dir / 'vessel_tracks_dark_period.json'
    track_summary = {
        'simulation_period': {
            'start': sim_config['start_time'],
            'end': sim_config['end_time'],
            'duration_hours': sim_config['duration_hours']
        },
        'vessels': [
            {
                'mmsi': vessel['mmsi'],
                'name': vessel['name'],
                'role': vessel['role'],
                'position_count': len([t for t in all_tracks if t['mmsi'] == vessel['mmsi']])
            }
            for vessel in vessel_config['gold_vessels']
        ],
        'sts_transfer_site': sts_site,
        'total_positions': len(all_tracks)
    }

    with open(json_path, 'w') as f:
        json.dump(track_summary, f, indent=2)
    logger.info(f"✅ Track summary exported: {json_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ VESSEL TRACK GENERATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
