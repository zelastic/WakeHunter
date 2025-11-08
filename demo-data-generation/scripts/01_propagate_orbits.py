#!/usr/bin/env python3
"""
Satellite Orbital Propagation Script

Uses tensorgator CPU backend to propagate ELK constellation orbits over July 16-30, 2025.
Calculates visibility from STS transfer location and extracts satellite pass times.

Output:
- satellite_passes_july2025.json: Pass schedule (AOS/LOS times)
- satellite_positions_ecef.parquet: Position data for 3D visualization
"""

import sys
import numpy as np
import pandas as pd
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add tensorgator to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))

import tensorgator as tg
from tensorgator.coord_conv import lla_to_ecef
from tensorgator.visibility import batch_visibility_check

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load satellite constellation configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'satellites.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_constellation_array(config):
    """
    Build numpy array of orbital elements for entire constellation.

    Returns:
        constellation: Array of shape (num_sats, 6) with Keplerian elements [a, e, i, raan, argp, M]
        satellite_names: List of satellite names in same order
        satellite_types: List of sensor types (SAR/RF/EO)
    """
    constellation = []
    satellite_names = []
    satellite_types = []

    # Process each sensor type
    for sensor_type in ['sar', 'rf', 'eo']:
        for sat in config['constellation'][sensor_type]:
            elements = sat['orbital_elements']

            # Convert to radians for tensorgator
            orbital_params = [
                elements['semi_major_axis_m'],
                elements['eccentricity'],
                np.radians(elements['inclination_deg']),
                np.radians(elements['raan_deg']),
                np.radians(elements['arg_perigee_deg']),
                np.radians(elements['mean_anomaly_deg'])
            ]

            constellation.append(orbital_params)
            satellite_names.append(sat['name'])
            satellite_types.append(sensor_type.upper())

    return np.array(constellation), satellite_names, satellite_types


def propagate_constellation(constellation, config):
    """
    Propagate constellation using tensorgator CPU backend.

    Returns:
        positions: Array of shape (num_sats, num_times, 3) in ECEF coordinates
        times: Array of timestamps (seconds since start)
    """
    prop_config = config['propagation']

    # Calculate time array
    duration_seconds = prop_config['duration_days'] * 86400
    time_step = prop_config['time_step_seconds']
    times = np.arange(0, duration_seconds, time_step)

    logger.info(f"Propagating {len(constellation)} satellites over {len(times):,} timesteps")
    logger.info(f"Duration: {prop_config['duration_days']} days, Time step: {time_step}s")

    # Propagate using tensorgator CPU backend
    import time
    start = time.time()

    positions = tg.satellite_positions(
        times,
        constellation,
        backend='cpu',
        return_frame='ecef',
        input_type='kepler'
    )

    elapsed = time.time() - start
    logger.info(f"✅ Propagation complete in {elapsed:.2f}s")

    return positions, times


def calculate_visibility(positions, config):
    """
    Calculate visibility from STS location using tensorgator.

    Returns:
        visibility: Boolean array of shape (1, num_times) - True when any satellite is visible
        visibility_by_sat: Boolean array of shape (num_sats, num_times) - Per-satellite visibility
    """
    ground_station = config['ground_station']
    min_elevation = np.radians(config['propagation']['min_elevation_deg'])

    # Convert ground station to ECEF (lla_to_ecef expects degrees)
    lat_deg = ground_station['latitude']
    lon_deg = ground_station['longitude']
    alt_m = ground_station['altitude_m']

    ground_ecef = lla_to_ecef(lat_deg, lon_deg, alt_m)
    ground_points = np.array([ground_ecef])

    logger.info(f"Calculating visibility from {ground_station['name']}")
    logger.info(f"  Location: {ground_station['latitude']}°N, {ground_station['longitude']}°E")
    logger.info(f"  Min elevation: {config['propagation']['min_elevation_deg']}°")

    # Calculate visibility (any satellite visible)
    visibility = batch_visibility_check(positions, ground_points, min_elevation)

    # Calculate per-satellite visibility
    num_sats = positions.shape[0]
    num_times = positions.shape[1]
    visibility_by_sat = np.zeros((num_sats, num_times), dtype=bool)

    for sat_idx in range(num_sats):
        sat_positions = positions[sat_idx:sat_idx+1, :, :]  # Keep 3D shape
        vis = batch_visibility_check(sat_positions, ground_points, min_elevation)
        visibility_by_sat[sat_idx, :] = vis[0, :]

    logger.info(f"✅ Visibility calculated for {num_sats} satellites")

    return visibility, visibility_by_sat


def extract_passes(visibility_by_sat, times, satellite_names, satellite_types, config):
    """
    Extract satellite pass information from visibility timeline.

    A pass is defined as a continuous period of visibility (elevation > min threshold).

    Returns:
        passes: List of dictionaries with pass information
    """
    passes = []
    num_sats = visibility_by_sat.shape[0]
    time_step = config['propagation']['time_step_seconds']
    start_time = datetime.fromisoformat(config['propagation']['start_time'].replace('Z', '+00:00'))

    for sat_idx in range(num_sats):
        vis_timeline = visibility_by_sat[sat_idx, :]
        sat_name = satellite_names[sat_idx]
        sat_type = satellite_types[sat_idx]

        # Find transitions (visibility changes)
        in_pass = False
        pass_start_idx = None

        for t_idx in range(len(vis_timeline)):
            currently_visible = vis_timeline[t_idx]

            if currently_visible and not in_pass:
                # Start of pass (AOS - Acquisition of Signal)
                in_pass = True
                pass_start_idx = t_idx

            elif not currently_visible and in_pass:
                # End of pass (LOS - Loss of Signal)
                in_pass = False
                pass_end_idx = t_idx - 1

                # Calculate pass details
                pass_duration_sec = (pass_end_idx - pass_start_idx + 1) * time_step

                # Find max elevation point (approximate - would need elevation calculation)
                pass_mid_idx = (pass_start_idx + pass_end_idx) // 2

                # Convert times to ISO format (convert numpy types to Python float)
                aos_time = start_time + timedelta(seconds=float(times[pass_start_idx]))
                los_time = start_time + timedelta(seconds=float(times[pass_end_idx]))
                max_elev_time = start_time + timedelta(seconds=float(times[pass_mid_idx]))

                passes.append({
                    'satellite_id': sat_name,
                    'satellite_type': sat_type,
                    'aos_time': aos_time.isoformat() + 'Z',
                    'los_time': los_time.isoformat() + 'Z',
                    'max_elevation_time': max_elev_time.isoformat() + 'Z',
                    'max_elevation_deg': config['propagation']['min_elevation_deg'] + 15,  # Approximate
                    'pass_duration_sec': pass_duration_sec,
                    'aos_index': int(pass_start_idx),
                    'los_index': int(pass_end_idx)
                })

        # Handle pass still in progress at end of timeline
        if in_pass:
            pass_end_idx = len(vis_timeline) - 1
            pass_duration_sec = (pass_end_idx - pass_start_idx + 1) * time_step
            pass_mid_idx = (pass_start_idx + pass_end_idx) // 2

            aos_time = start_time + timedelta(seconds=float(times[pass_start_idx]))
            los_time = start_time + timedelta(seconds=float(times[pass_end_idx]))
            max_elev_time = start_time + timedelta(seconds=float(times[pass_mid_idx]))

            passes.append({
                'satellite_id': sat_name,
                'satellite_type': sat_type,
                'aos_time': aos_time.isoformat() + 'Z',
                'los_time': los_time.isoformat() + 'Z',
                'max_elevation_time': max_elev_time.isoformat() + 'Z',
                'max_elevation_deg': config['propagation']['min_elevation_deg'] + 15,
                'pass_duration_sec': pass_duration_sec,
                'aos_index': int(pass_start_idx),
                'los_index': int(pass_end_idx)
            })

    logger.info(f"✅ Extracted {len(passes)} satellite passes")

    # Log pass distribution
    for sat_type in ['SAR', 'RF', 'EO']:
        type_passes = [p for p in passes if p['satellite_type'] == sat_type]
        logger.info(f"   {sat_type}: {len(type_passes)} passes")

    return passes


def analyze_dark_period_coverage(passes, config):
    """
    Analyze satellite coverage during the dark period (July 17 20:45 - July 20 07:00).
    """
    from datetime import timezone
    dark_start = datetime(2025, 7, 17, 20, 45, 0, tzinfo=timezone.utc)
    dark_end = datetime(2025, 7, 20, 7, 0, 0, tzinfo=timezone.utc)

    logger.info("\n" + "="*70)
    logger.info("DARK PERIOD COVERAGE ANALYSIS")
    logger.info("="*70)
    logger.info(f"Dark Period: {dark_start.isoformat()} to {dark_end.isoformat()}")
    logger.info(f"Duration: {(dark_end - dark_start).total_seconds() / 3600:.1f} hours")

    # Filter passes during dark period
    dark_period_passes = []
    for p in passes:
        pass_time = datetime.fromisoformat(p['max_elevation_time'].replace('Z', ''))
        if dark_start <= pass_time <= dark_end:
            dark_period_passes.append(p)

    logger.info(f"\n✅ {len(dark_period_passes)} satellite passes during dark period:")

    for sat_type in ['RF', 'SAR', 'EO']:
        type_passes = [p for p in dark_period_passes if p['satellite_type'] == sat_type]
        logger.info(f"\n{sat_type} Passes ({len(type_passes)}):")
        for p in sorted(type_passes, key=lambda x: x['max_elevation_time']):
            logger.info(f"  {p['satellite_id']}: {p['max_elevation_time']} (duration: {p['pass_duration_sec']/60:.1f} min)")

    logger.info("="*70 + "\n")

    return dark_period_passes


def export_pass_schedule(passes, output_dir):
    """Export pass schedule to JSON."""
    output_path = output_dir / 'satellite_passes_july2025.json'

    with open(output_path, 'w') as f:
        json.dump(passes, f, indent=2)

    logger.info(f"✅ Pass schedule exported: {output_path}")


def export_position_data(positions, times, satellite_names, config, output_dir):
    """Export satellite positions for 3D visualization."""
    num_sats, num_times, _ = positions.shape
    start_time = datetime.fromisoformat(config['propagation']['start_time'].replace('Z', '+00:00'))

    # Create DataFrame with positions
    records = []
    for sat_idx in range(num_sats):
        for t_idx in range(num_times):
            timestamp = start_time + timedelta(seconds=float(times[t_idx]))
            x, y, z = positions[sat_idx, t_idx, :]

            records.append({
                'satellite_id': satellite_names[sat_idx],
                'timestamp': timestamp.isoformat() + 'Z',
                'x_ecef': x,
                'y_ecef': y,
                'z_ecef': z
            })

    df = pd.DataFrame(records)

    # Export as Parquet (compressed)
    output_path = output_dir / 'satellite_positions_ecef.parquet'
    df.to_parquet(output_path, index=False)

    logger.info(f"✅ Position data exported: {output_path} ({len(df):,} records)")


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("SATELLITE ORBITAL PROPAGATION")
    logger.info("="*70)

    # Setup paths
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    try:
        # Load configuration
        logger.info("Loading constellation configuration...")
        config = load_config()

        # Build constellation
        constellation, satellite_names, satellite_types = build_constellation_array(config)
        logger.info(f"✅ Built constellation: {len(satellite_names)} satellites")
        for sat_type in ['SAR', 'RF', 'EO']:
            count = sum(1 for t in satellite_types if t == sat_type)
            logger.info(f"   {sat_type}: {count} satellites")

        # Propagate orbits
        positions, times = propagate_constellation(constellation, config)

        # Calculate visibility
        visibility, visibility_by_sat = calculate_visibility(positions, config)

        # Extract passes
        passes = extract_passes(visibility_by_sat, times, satellite_names, satellite_types, config)

        # Analyze dark period coverage
        dark_passes = analyze_dark_period_coverage(passes, config)

        # Export data
        export_pass_schedule(passes, output_dir)
        export_position_data(positions, times, satellite_names, config, output_dir)

        logger.info("\n" + "="*70)
        logger.info("✅ ORBITAL PROPAGATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total passes: {len(passes)}")
        logger.info(f"Dark period passes: {len(dark_passes)}")
        logger.info(f"Outputs: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"❌ Error during orbital propagation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
