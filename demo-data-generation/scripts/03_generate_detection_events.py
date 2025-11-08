#!/usr/bin/env python3
"""
Generate Multi-Sensor Satellite Detection Events

Creates detection events when satellites pass over vessels during dark period:
- RF detections: Radio frequency emissions detected
- SAR detections: All-weather radar imaging (triggered by cloudy conditions)
- EO detections: Visual confirmation (triggered by clear conditions)

Timeline: July 17 20:45 - July 20 07:00 UTC (58.25 hours)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load satellite passes and vessel tracks."""
    output_dir = Path(__file__).parent.parent / 'outputs'

    # Load satellite passes
    with open(output_dir / 'satellite_passes_july2025.json') as f:
        all_passes = json.load(f)

    # Load vessel tracks
    vessel_tracks = pd.read_parquet(output_dir / 'vessel_tracks_dark_period.parquet')

    return all_passes, vessel_tracks


def filter_dark_period_passes(passes):
    """Filter passes to only those during dark period."""
    dark_start = datetime(2025, 7, 17, 20, 45, 0, tzinfo=timezone.utc)
    dark_end = datetime(2025, 7, 20, 7, 0, 0, tzinfo=timezone.utc)

    dark_passes = []
    for p in passes:
        pass_time = datetime.fromisoformat(p['max_elevation_time'].removesuffix('Z'))
        if dark_start <= pass_time <= dark_end:
            dark_passes.append(p)

    return dark_passes


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def get_weather_condition(timestamp):
    """
    Determine weather condition for weather-driven tasking narrative.

    Cloudy during approach -> triggers SAR imaging
    Clear during transfer -> triggers EO visual confirmation
    """
    ts = datetime.fromisoformat(timestamp.removesuffix('Z'))

    # July 17 21:00 - July 18 03:00: Cloudy (approach phase)
    # July 18 03:00 - July 18 09:00: Clearing (transfer phase)
    # July 18 09:00+: Clear (departure phase)

    cloudy_start = datetime(2025, 7, 17, 21, 0, 0, tzinfo=timezone.utc)
    clearing_start = datetime(2025, 7, 18, 3, 0, 0, tzinfo=timezone.utc)
    clear_start = datetime(2025, 7, 18, 9, 0, 0, tzinfo=timezone.utc)

    if ts < cloudy_start:
        return 'partly_cloudy', 30
    elif ts < clearing_start:
        return 'cloudy', 80
    elif ts < clear_start:
        return 'partly_cloudy', 40
    else:
        return 'clear', 10


def generate_rf_detection(satellite_pass, vessel_position):
    """Generate RF detection event."""
    weather, cloud_cover = get_weather_condition(satellite_pass['max_elevation_time'])

    return {
        'detection_id': f"rf_{satellite_pass['satellite_id']}_{satellite_pass['max_elevation_time']}_{vessel_position['mmsi']}",
        'timestamp': satellite_pass['max_elevation_time'],
        'sensor_type': 'RF',
        'satellite_id': satellite_pass['satellite_id'],
        'satellite_type': satellite_pass['satellite_type'],
        'elevation_deg': satellite_pass['max_elevation_deg'],
        'pass_duration_sec': satellite_pass['pass_duration_sec'],
        'vessel_mmsi': vessel_position['mmsi'],
        'vessel_name': vessel_position['name'],
        'vessel_lat': vessel_position['lat'],
        'vessel_lon': vessel_position['lon'],
        'vessel_ais_on': vessel_position['ais_on'],
        'detection_confidence': 0.95 if vessel_position['ais_on'] else 0.75,  # Lower confidence if AIS off
        'rf_signature': {
            'frequency_mhz': 162.0,  # AIS frequency
            'signal_strength_db': np.random.uniform(-80, -60),
            'detected': vessel_position['ais_on']
        },
        'weather': weather,
        'cloud_cover_pct': cloud_cover,
        'tasking_triggered': {
            'sar': weather == 'cloudy',  # Task SAR if cloudy
            'eo': weather == 'clear'      # Task EO if clear
        }
    }


def generate_sar_detection(satellite_pass, vessel_position):
    """Generate SAR detection event (all-weather imaging)."""
    weather, cloud_cover = get_weather_condition(satellite_pass['max_elevation_time'])

    # SAR detects regardless of weather or AIS status
    return {
        'detection_id': f"sar_{satellite_pass['satellite_id']}_{satellite_pass['max_elevation_time']}_{vessel_position['mmsi']}",
        'timestamp': satellite_pass['max_elevation_time'],
        'sensor_type': 'SAR',
        'satellite_id': satellite_pass['satellite_id'],
        'satellite_type': satellite_pass['satellite_type'],
        'elevation_deg': satellite_pass['max_elevation_deg'],
        'pass_duration_sec': satellite_pass['pass_duration_sec'],
        'vessel_mmsi': vessel_position['mmsi'],
        'vessel_name': vessel_position['name'],
        'vessel_lat': vessel_position['lat'],
        'vessel_lon': vessel_position['lon'],
        'vessel_ais_on': vessel_position['ais_on'],
        'detection_confidence': 0.92,  # High confidence (all-weather)
        'image_metadata': {
            'resolution_m': 3.0,
            'polarization': 'VV',
            'vessel_length_m': np.random.uniform(150, 200) if 'TANKER' in vessel_position.get('vessel_type', '').upper() else np.random.uniform(100, 150),
            'heading_deg': vessel_position['heading']
        },
        'weather': weather,
        'cloud_cover_pct': cloud_cover,
        'tasking_reason': 'RF anomaly detected' if not vessel_position['ais_on'] else 'Routine all-weather imaging'
    }


def generate_eo_detection(satellite_pass, vessel_position):
    """Generate EO detection event (visual confirmation)."""
    weather, cloud_cover = get_weather_condition(satellite_pass['max_elevation_time'])

    # EO only detects if weather is clear enough
    detected = cloud_cover < 50

    return {
        'detection_id': f"eo_{satellite_pass['satellite_id']}_{satellite_pass['max_elevation_time']}_{vessel_position['mmsi']}",
        'timestamp': satellite_pass['max_elevation_time'],
        'sensor_type': 'EO',
        'satellite_id': satellite_pass['satellite_id'],
        'satellite_type': satellite_pass['satellite_type'],
        'elevation_deg': satellite_pass['max_elevation_deg'],
        'pass_duration_sec': satellite_pass['pass_duration_sec'],
        'vessel_mmsi': vessel_position['mmsi'],
        'vessel_name': vessel_position['name'],
        'vessel_lat': vessel_position['lat'],
        'vessel_lon': vessel_position['lon'],
        'vessel_ais_on': vessel_position['ais_on'],
        'detection_confidence': 0.98 if detected else 0.0,
        'image_metadata': {
            'resolution_m': 0.31,
            'bands': ['panchromatic', 'multispectral'],
            'vessel_visible': detected,
            'visual_features': ['hull_visible', 'deck_equipment', 'transfer_equipment'] if detected else []
        },
        'weather': weather,
        'cloud_cover_pct': cloud_cover,
        'tasking_reason': 'Visual confirmation of SAR detection' if detected else 'Cloud cover too high'
    }


def main():
    """Generate all detection events."""
    logger.info("=" * 70)
    logger.info("SATELLITE DETECTION EVENT GENERATION")
    logger.info("=" * 70)

    # Load data
    all_passes, vessel_tracks = load_data()
    dark_passes = filter_dark_period_passes(all_passes)

    logger.info(f"Total satellite passes: {len(all_passes)}")
    logger.info(f"Dark period passes: {len(dark_passes)}")
    logger.info(f"Vessel position reports: {len(vessel_tracks):,}")
    logger.info("")

    # Generate detections
    detections = []

    for satellite_pass in dark_passes:
        pass_time = satellite_pass['max_elevation_time']
        sat_type = satellite_pass['satellite_type']

        # Find vessel positions at this time (within ±5 minutes for flexibility)
        pass_dt = datetime.fromisoformat(pass_time.removesuffix('Z'))
        time_window_start = (pass_dt - timedelta(minutes=5)).isoformat()
        time_window_end = (pass_dt + timedelta(minutes=5)).isoformat()

        nearby_vessels = vessel_tracks[
            (vessel_tracks['timestamp'] >= time_window_start) &
            (vessel_tracks['timestamp'] <= time_window_end)
        ]

        if len(nearby_vessels) == 0:
            continue

        # Get closest position for each vessel
        for mmsi in nearby_vessels['mmsi'].unique():
            vessel_positions = nearby_vessels[nearby_vessels['mmsi'] == mmsi]
            # Take the position closest to pass time
            vessel_position = vessel_positions.iloc[0].to_dict()

            # Generate detection based on sensor type
            if sat_type == 'RF':
                detection = generate_rf_detection(satellite_pass, vessel_position)
                detections.append(detection)
            elif sat_type == 'SAR':
                detection = generate_sar_detection(satellite_pass, vessel_position)
                detections.append(detection)
            elif sat_type == 'EO':
                detection = generate_eo_detection(satellite_pass, vessel_position)
                detections.append(detection)

    logger.info(f"Generated {len(detections)} detection events")

    # Count by sensor type
    rf_count = len([d for d in detections if d['sensor_type'] == 'RF'])
    sar_count = len([d for d in detections if d['sensor_type'] == 'SAR'])
    eo_count = len([d for d in detections if d['sensor_type'] == 'EO'])

    logger.info(f"  RF detections: {rf_count}")
    logger.info(f"  SAR detections: {sar_count}")
    logger.info(f"  EO detections: {eo_count}")
    logger.info("")

    # Export to JSON
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_path = output_dir / 'detection_events_dark_period.json'

    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'simulation_period': {
                    'start': '2025-07-17T20:45:00Z',
                    'end': '2025-07-20T07:00:00Z',
                    'duration_hours': 58.25
                },
                'total_detections': len(detections),
                'by_sensor': {
                    'RF': rf_count,
                    'SAR': sar_count,
                    'EO': eo_count
                }
            },
            'detections': detections
        }, f, indent=2)

    logger.info(f"✅ Detection events exported: {output_path}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ DETECTION EVENT GENERATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
