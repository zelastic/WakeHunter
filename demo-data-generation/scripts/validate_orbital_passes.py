#!/usr/bin/env python3
"""
Validate Satellite Orbital Passes

Creates visualizations and tests to verify:
1. Satellite passes are over correct location
2. Timing during dark period makes sense
3. Ground tracks are realistic
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add tensorgator to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'tensorgator'))
from tensorgator.coord_conv import ecef_to_lla


def load_pass_data():
    """Load satellite pass schedule."""
    output_dir = Path(__file__).parent.parent / 'outputs'
    with open(output_dir / 'satellite_passes_july2025.json') as f:
        return json.load(f)


def load_position_data():
    """Load satellite positions."""
    output_dir = Path(__file__).parent.parent / 'outputs'
    return pd.read_parquet(output_dir / 'satellite_positions_ecef.parquet')


def validate_pass_locations(passes, positions_df):
    """
    Verify that satellite passes are actually over the STS location (32.5°N, 126.0°E).
    """
    print("="*70)
    print("VALIDATING PASS LOCATIONS")
    print("="*70)

    target_lat = 32.5
    target_lon = 126.0
    max_distance_km = 250  # Swath width for SAR

    failures = []

    for p in passes[:10]:  # Check first 10 passes
        sat_id = p['satellite_id']
        pass_time = p['max_elevation_time']

        # Find position at max elevation time
        sat_positions = positions_df[
            (positions_df['satellite_id'] == sat_id) &
            (positions_df['timestamp'] == pass_time)
        ]

        if len(sat_positions) == 0:
            failures.append(f"  ❌ {sat_id} at {pass_time}: No position data found")
            continue

        pos = sat_positions.iloc[0]

        # Convert ECEF to LLA (ecef_to_lla returns degrees)
        position_ecef = np.array([pos['x_ecef'], pos['y_ecef'], pos['z_ecef']])
        lat_deg, lon_deg, alt_m = ecef_to_lla(position_ecef)
        alt_km = alt_m / 1000

        # Calculate distance from target
        dlat = lat_deg - target_lat
        dlon = lon_deg - target_lon
        distance_km = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(target_lat)))**2)

        if distance_km > max_distance_km:
            failures.append(f"  ❌ {sat_id} at {pass_time}: {distance_km:.1f}km from target (> {max_distance_km}km)")
        else:
            print(f"  ✅ {sat_id} at {pass_time}: {distance_km:.1f}km from target, alt={alt_km:.0f}km")

    if failures:
        print("\n⚠️  FAILURES:")
        for f in failures:
            print(f)
    else:
        print(f"\n✅ All passes validated within {max_distance_km}km of target")

    print()


def plot_ground_tracks(positions_df, passes):
    """
    Plot satellite ground tracks over the STS location.
    """
    print("="*70)
    print("GENERATING GROUND TRACK VISUALIZATION")
    print("="*70)

    # Convert all positions to LLA
    print("Converting ECEF to geographic coordinates...")
    lats = []
    lons = []
    sat_ids = []

    for _, row in positions_df.iterrows():
        position_ecef = np.array([row['x_ecef'], row['y_ecef'], row['z_ecef']])
        lat_deg, lon_deg, alt_m = ecef_to_lla(position_ecef)
        lats.append(lat_deg)
        lons.append(lon_deg)
        sat_ids.append(row['satellite_id'])

    positions_df['lat'] = lats
    positions_df['lon'] = lons

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot ground tracks (sample every 10th point for performance)
    colors = {
        'elk-1-sar': 'blue', 'elk-2-sar': 'cyan', 'elk-3-sar': 'lightblue',
        'elk-1-rf': 'red', 'elk-2-rf': 'orange', 'elk-3-rf': 'pink',
        'elk-1-eo': 'green', 'elk-2-eo': 'lime', 'elk-3-eo': 'darkgreen'
    }

    for sat_id, color in colors.items():
        sat_data = positions_df[positions_df['satellite_id'] == sat_id][::10]
        ax.plot(sat_data['lon'], sat_data['lat'],
                color=color, alpha=0.3, linewidth=0.5, label=sat_id)

    # Mark STS location
    ax.plot(126.0, 32.5, 'r*', markersize=20, label='STS Transfer Site', zorder=10)

    # Mark dark period passes
    dark_start = datetime(2025, 7, 17, 20, 45, 0, tzinfo=timezone.utc)
    dark_end = datetime(2025, 7, 20, 7, 0, 0, tzinfo=timezone.utc)

    for p in passes:
        pass_time = datetime.fromisoformat(p['max_elevation_time'].removesuffix('Z'))
        if dark_start <= pass_time <= dark_end:
            sat_id = p['satellite_id']
            pass_pos = positions_df[
                (positions_df['satellite_id'] == sat_id) &
                (positions_df['timestamp'] == p['max_elevation_time'])
            ]
            if len(pass_pos) > 0:
                pos = pass_pos.iloc[0]
                ax.plot(pos['lon'], pos['lat'], 'ko', markersize=3, zorder=5)

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Satellite Ground Tracks - July 16-30, 2025\n(Black dots = Dark Period Passes)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Set bounds around East China Sea
    ax.set_xlim(115, 135)
    ax.set_ylim(25, 40)

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_path = output_dir / 'ground_tracks_visualization.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ Ground track plot saved: {output_path}")
    print()


def plot_pass_timeline(passes):
    """
    Create timeline visualization of satellite passes during dark period.
    """
    print("="*70)
    print("GENERATING PASS TIMELINE")
    print("="*70)

    dark_start = datetime(2025, 7, 17, 20, 45, 0, tzinfo=timezone.utc)
    dark_end = datetime(2025, 7, 20, 7, 0, 0, tzinfo=timezone.utc)

    # Filter dark period passes
    dark_passes = []
    for p in passes:
        pass_time = datetime.fromisoformat(p['max_elevation_time'].removesuffix('Z'))
        if dark_start <= pass_time <= dark_end:
            dark_passes.append({
                'sat_id': p['satellite_id'],
                'sat_type': p['satellite_type'],
                'time': pass_time,
                'duration': p['pass_duration_sec'] / 60
            })

    # Create timeline plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by sensor type
    rf_passes = [p for p in dark_passes if p['sat_type'] == 'RF']
    sar_passes = [p for p in dark_passes if p['sat_type'] == 'SAR']
    eo_passes = [p for p in dark_passes if p['sat_type'] == 'EO']

    # Plot passes
    y_offset = {'RF': 3, 'SAR': 2, 'EO': 1}
    colors = {'RF': 'red', 'SAR': 'blue', 'EO': 'green'}

    for passes_list, sensor_type in [(rf_passes, 'RF'), (sar_passes, 'SAR'), (eo_passes, 'EO')]:
        for p in passes_list:
            hours_from_start = (p['time'] - dark_start).total_seconds() / 3600
            ax.barh(y_offset[sensor_type], p['duration']/60,
                   left=hours_from_start, height=0.8,
                   color=colors[sensor_type], alpha=0.6, edgecolor='black')

    # Mark STS transfer time (July 18 00:00 UTC)
    sts_time = datetime(2025, 7, 18, 0, 0, 0, tzinfo=timezone.utc)
    sts_hours = (sts_time - dark_start).total_seconds() / 3600
    ax.axvline(sts_hours, color='purple', linestyle='--', linewidth=2, label='STS Transfer Start')

    # Add shaded region for STS transfer (6 hours)
    ax.axvspan(sts_hours, sts_hours + 6, alpha=0.2, color='purple', label='STS Transfer (6h)')

    ax.set_xlabel('Hours Since Dark Period Start (July 17 20:45 UTC)')
    ax.set_ylabel('Sensor Type')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['EO', 'SAR', 'RF'])
    ax.set_title('Satellite Pass Timeline During Dark Period (July 17-20, 2025)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_path = output_dir / 'pass_timeline_visualization.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ Pass timeline plot saved: {output_path}")
    print()


def generate_summary_report(passes, positions_df):
    """Generate summary statistics report."""
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    dark_start = datetime(2025, 7, 17, 20, 45, 0, tzinfo=timezone.utc)
    dark_end = datetime(2025, 7, 20, 7, 0, 0, tzinfo=timezone.utc)

    # Count passes by type
    total_passes = len(passes)
    # Parse timestamps properly (remove Z suffix and parse)
    dark_passes = [p for p in passes
                   if dark_start <= datetime.fromisoformat(p['max_elevation_time'].removesuffix('Z')) <= dark_end]

    rf_total = len([p for p in passes if p['satellite_type'] == 'RF'])
    sar_total = len([p for p in passes if p['satellite_type'] == 'SAR'])
    eo_total = len([p for p in passes if p['satellite_type'] == 'EO'])

    rf_dark = len([p for p in dark_passes if p['satellite_type'] == 'RF'])
    sar_dark = len([p for p in dark_passes if p['satellite_type'] == 'SAR'])
    eo_dark = len([p for p in dark_passes if p['satellite_type'] == 'EO'])

    print(f"Total simulation period: July 16-30, 2025 (14 days)")
    print(f"Dark period: July 17 20:45 - July 20 07:00 (58.2 hours)")
    print()
    print(f"Total satellite passes: {total_passes}")
    print(f"  RF:  {rf_total} ({rf_dark} during dark period)")
    print(f"  SAR: {sar_total} ({sar_dark} during dark period)")
    print(f"  EO:  {eo_total} ({eo_dark} during dark period)")
    print()
    print(f"Dark period coverage: {len(dark_passes)} passes")
    print(f"  Average time between passes: {58.2 / len(dark_passes):.1f} hours")
    print()

    # Verify position data integrity
    print(f"Position records: {len(positions_df):,}")
    print(f"Unique satellites: {positions_df['satellite_id'].nunique()}")
    print(f"Time range: {positions_df['timestamp'].min()} to {positions_df['timestamp'].max()}")
    print()


def main():
    """Main validation function."""
    print("\n")
    print("="*70)
    print("ORBITAL PASS VALIDATION")
    print("="*70)
    print()

    # Load data
    passes = load_pass_data()
    positions_df = load_position_data()

    # Run validations
    validate_pass_locations(passes, positions_df)
    generate_summary_report(passes, positions_df)

    # Generate visualizations
    plot_ground_tracks(positions_df, passes)
    plot_pass_timeline(passes)

    print("="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70)
    print()
    print("Outputs:")
    print("  - ground_tracks_visualization.png")
    print("  - pass_timeline_visualization.png")
    print()


if __name__ == "__main__":
    main()
