#!/usr/bin/env python3
"""
Visualize satellite ground tracks with sensor footprints over one day
Shows coverage over the East China Sea during dark vessel period
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime, timedelta, timezone
from pathlib import Path

print("="*70)
print("VISUALIZING SATELLITE GROUND TRACKS WITH SENSOR FOOTPRINTS")
print("="*70)

# Load satellite positions
sat_positions = pd.read_parquet(Path(__file__).parent.parent / 'outputs' / 'satellite_positions_ecef.parquet')
sat_positions['timestamp_dt'] = pd.to_datetime(sat_positions['timestamp'])

# Select one day (July 18, 2025 - when vessels go dark)
target_date = datetime(2025, 7, 18, tzinfo=timezone.utc)
one_day_positions = sat_positions[
    (sat_positions['timestamp_dt'] >= target_date) &
    (sat_positions['timestamp_dt'] < target_date + timedelta(days=1))
]

print(f"\n✅ Loaded {len(one_day_positions):,} satellite positions for {target_date.date()}")

# Load vessel positions to show context
gold_standard_dir = Path('/Users/zski/PycharmProjects/ElasticShipVision/vessel-simulation-service/gold_standard_bad_vessel_sim_output')
vessel_tracks = pd.read_parquet(gold_standard_dir / 'ais_202507_dynamic.parquet')
if vessel_tracks['timestamp'].dtype == 'object':
    vessel_tracks['timestamp'] = pd.to_datetime(vessel_tracks['timestamp']).astype(int) / 1e9

# Filter vessels to same day
day_start = target_date.timestamp()
day_end = (target_date + timedelta(days=1)).timestamp()
vessels_one_day = vessel_tracks[
    (vessel_tracks['timestamp'] >= day_start) &
    (vessel_tracks['timestamp'] < day_end)
]

print(f"✅ Loaded {len(vessels_one_day)} vessel positions")

# Sensor specifications (swath widths in km converted to degrees at equator)
# 1 degree latitude ≈ 111 km
SENSOR_SPECS = {
    'RF': {'color': '#FFA500', 'swath_km': 800, 'label': 'RF (800km)'},
    'SAR': {'color': '#00FF00', 'swath_km': 700, 'label': 'SAR (700km)'},
    'EO': {'color': '#00BFFF', 'swath_km': 500, 'label': 'EO (500km)'}
}

# Create figure with subplots for each sensor type
fig = plt.figure(figsize=(20, 14))

# Define region of interest (East China Sea + buffer)
lon_min, lon_max = 120, 132
lat_min, lat_max = 28, 38

for idx, (sensor_type, specs) in enumerate(SENSOR_SPECS.items()):
    ax = plt.subplot(2, 2, idx + 1)

    # Plot coastlines (simple box for now)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title(f'{sensor_type} Satellites - Ground Track & Footprints\n{target_date.date()}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # Get satellites for this sensor type
    sensor_sats = one_day_positions[one_day_positions['sensor_type'] == sensor_type]

    # Plot each satellite's ground track
    for sat_id in sensor_sats['satellite_id'].unique():
        sat_data = sensor_sats[sensor_sats['satellite_id'] == sat_id].sort_values('timestamp_dt')

        # Plot full ground track (thin line)
        ax.plot(sat_data['longitude'], sat_data['latitude'],
               color=specs['color'], alpha=0.3, linewidth=0.5, linestyle='-')

        # Plot footprints at intervals (every 30 minutes)
        interval_minutes = 30
        footprint_positions = sat_data.iloc[::interval_minutes * 1]  # 1-minute resolution

        for _, pos in footprint_positions.iterrows():
            lat_rad = np.radians(pos['latitude'])

            if sensor_type == 'RF':
                # RF: Circular footprint (omnidirectional cone)
                radius_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                circle = Circle((pos['longitude'], pos['latitude']),
                              radius_deg,
                              facecolor=specs['color'],
                              alpha=0.1,
                              edgecolor=specs['color'],
                              linewidth=0.5)
                ax.add_patch(circle)

            elif sensor_type == 'SAR':
                # SAR: Rectangle perpendicular to ground track (side-looking)
                swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                along_track_deg = 50 / (111.0)  # 50km along-track

                # Get heading from ground_track_heading
                heading = pos['ground_track_heading']

                # Rectangle perpendicular to heading
                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                rect = Rectangle((pos['longitude'] - swath_width_deg/2, pos['latitude'] - along_track_deg/2),
                               swath_width_deg, along_track_deg,
                               facecolor=specs['color'], alpha=0.15,
                               edgecolor=specs['color'], linewidth=0.5)

                # Rotate rectangle to align with ground track
                t = Affine2D().rotate_deg_around(pos['longitude'], pos['latitude'], heading) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)

            elif sensor_type == 'EO':
                # EO: Narrow rectangle along ground track (push-broom)
                swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                along_track_deg = 100 / (111.0)  # 100km along-track (longer strip)

                heading = pos['ground_track_heading']

                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                # Narrow rectangle along heading
                rect = Rectangle((pos['longitude'] - swath_width_deg/2, pos['latitude'] - along_track_deg/2),
                               swath_width_deg, along_track_deg,
                               facecolor=specs['color'], alpha=0.15,
                               edgecolor=specs['color'], linewidth=0.5)

                t = Affine2D().rotate_deg_around(pos['longitude'], pos['latitude'], heading) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)

            # Mark satellite position
            ax.plot(pos['longitude'], pos['latitude'],
                   'o', color=specs['color'], markersize=2, alpha=0.6)

    # Plot vessel positions
    for mmsi in vessels_one_day['mmsi'].unique():
        vessel_data = vessels_one_day[vessels_one_day['mmsi'] == mmsi]
        vessel_name = vessel_data.iloc[0]['vessel_name']

        # Different colors for AIS on/off
        ais_on = vessel_data[vessel_data['ais_visible'] == True]
        ais_off = vessel_data[vessel_data['ais_visible'] == False]

        if len(ais_on) > 0:
            ax.plot(ais_on['longitude'], ais_on['latitude'],
                   'o', color='blue', markersize=3, alpha=0.5, label=f'{vessel_name} (AIS ON)')

        if len(ais_off) > 0:
            ax.plot(ais_off['longitude'], ais_off['latitude'],
                   'x', color='red', markersize=4, alpha=0.8, label=f'{vessel_name} (AIS OFF)')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             loc='upper right', fontsize=8, framealpha=0.9)

# Fourth subplot: All sensors combined
ax = plt.subplot(2, 2, 4)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitude (°E)', fontsize=10)
ax.set_ylabel('Latitude (°N)', fontsize=10)
ax.set_title(f'All Sensors - Combined Coverage\n{target_date.date()}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal')

# Plot all satellites with color coding
for sensor_type, specs in SENSOR_SPECS.items():
    sensor_sats = one_day_positions[one_day_positions['sensor_type'] == sensor_type]

    for sat_id in sensor_sats['satellite_id'].unique():
        sat_data = sensor_sats[sensor_sats['satellite_id'] == sat_id].sort_values('timestamp_dt')

        # Plot ground track
        ax.plot(sat_data['longitude'], sat_data['latitude'],
               color=specs['color'], alpha=0.2, linewidth=0.5,
               label=f"{sensor_type}" if sat_id == sensor_sats['satellite_id'].unique()[0] else "")

        # Plot footprints at longer intervals for combined view
        interval_minutes = 60
        footprint_positions = sat_data.iloc[::interval_minutes * 1]

        for _, pos in footprint_positions.iterrows():
            lat_rad = np.radians(pos['latitude'])

            if sensor_type == 'RF':
                radius_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                circle = Circle((pos['longitude'], pos['latitude']),
                              radius_deg,
                              facecolor=specs['color'],
                              alpha=0.05,
                              edgecolor=specs['color'],
                              linewidth=0.3)
                ax.add_patch(circle)

            elif sensor_type == 'SAR':
                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                along_track_deg = 50 / 111.0
                heading = pos['ground_track_heading']

                rect = Rectangle((pos['longitude'] - swath_width_deg/2, pos['latitude'] - along_track_deg/2),
                               swath_width_deg, along_track_deg,
                               facecolor=specs['color'], alpha=0.08,
                               edgecolor=specs['color'], linewidth=0.3)
                t = Affine2D().rotate_deg_around(pos['longitude'], pos['latitude'], heading) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)

            elif sensor_type == 'EO':
                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                along_track_deg = 100 / 111.0
                heading = pos['ground_track_heading']

                rect = Rectangle((pos['longitude'] - swath_width_deg/2, pos['latitude'] - along_track_deg/2),
                               swath_width_deg, along_track_deg,
                               facecolor=specs['color'], alpha=0.08,
                               edgecolor=specs['color'], linewidth=0.3)
                t = Affine2D().rotate_deg_around(pos['longitude'], pos['latitude'], heading) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)

# Plot vessels on combined view
for mmsi in vessels_one_day['mmsi'].unique():
    vessel_data = vessels_one_day[vessels_one_day['mmsi'] == mmsi]
    vessel_name = vessel_data.iloc[0]['vessel_name']

    ais_on = vessel_data[vessel_data['ais_visible'] == True]
    ais_off = vessel_data[vessel_data['ais_visible'] == False]

    if len(ais_on) > 0:
        ax.plot(ais_on['longitude'], ais_on['latitude'],
               'o', color='blue', markersize=3, alpha=0.5, label=f'{vessel_name} (AIS ON)')

    if len(ais_off) > 0:
        ax.plot(ais_off['longitude'], ais_off['latitude'],
               'x', color='red', markersize=4, alpha=0.8, label=f'{vessel_name} (AIS OFF)')

# Legend for combined view
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
         loc='upper right', fontsize=8, framealpha=0.9)

plt.suptitle('ELK Constellation Ground Tracks with Sensor Footprints\nEast China Sea - July 18, 2025',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / 'outputs'
output_file = output_dir / 'ground_tracks_with_footprints.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved visualization: {output_file}")

# Also create a focused view on vessel area
fig2, ax2 = plt.subplots(figsize=(14, 12))

# Zoom into vessel area
vessel_lon_min = vessels_one_day['longitude'].min() - 2
vessel_lon_max = vessels_one_day['longitude'].max() + 2
vessel_lat_min = vessels_one_day['latitude'].min() - 2
vessel_lat_max = vessels_one_day['latitude'].max() + 2

ax2.set_xlim(vessel_lon_min, vessel_lon_max)
ax2.set_ylim(vessel_lat_min, vessel_lat_max)
ax2.set_xlabel('Longitude (°E)', fontsize=12)
ax2.set_ylabel('Latitude (°N)', fontsize=12)
ax2.set_title(f'Satellite Coverage Over Dark Vessels\n{target_date.date()} - East China Sea',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal')

# Plot all satellite passes over this region
for sensor_type, specs in SENSOR_SPECS.items():
    sensor_sats = one_day_positions[one_day_positions['sensor_type'] == sensor_type]

    # Filter to positions that could see the vessels
    nearby_sats = sensor_sats[
        (sensor_sats['longitude'] >= vessel_lon_min - 10) &
        (sensor_sats['longitude'] <= vessel_lon_max + 10) &
        (sensor_sats['latitude'] >= vessel_lat_min - 10) &
        (sensor_sats['latitude'] <= vessel_lat_max + 10)
    ]

    for sat_id in nearby_sats['satellite_id'].unique():
        sat_data = nearby_sats[nearby_sats['satellite_id'] == sat_id].sort_values('timestamp_dt')

        if len(sat_data) == 0:
            continue

        # Plot ground track segment
        ax2.plot(sat_data['longitude'], sat_data['latitude'],
               color=specs['color'], alpha=0.4, linewidth=1.5,
               label=f"{sensor_type}" if sat_id == nearby_sats['satellite_id'].unique()[0] else "")

        # Plot footprints every 10 minutes in zoomed view
        interval_minutes = 10
        footprint_positions = sat_data.iloc[::interval_minutes * 1]

        for _, pos in footprint_positions.iterrows():
            lat_rad = np.radians(pos['latitude'])

            # RF: Circular footprint (omnidirectional cone)
            if sensor_type == 'RF':
                radius_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                circle = Circle((pos['longitude'], pos['latitude']),
                              radius_deg,
                              facecolor=specs['color'],
                              alpha=0.15,
                              edgecolor=specs['color'],
                              linewidth=1.0)
                ax2.add_patch(circle)

            # SAR: Rectangle perpendicular to ground track (side-looking radar)
            elif sensor_type == 'SAR':
                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                along_track_deg = 50 / (111.0)  # 50km along-track
                heading = pos['ground_track_heading']

                rect = Rectangle((pos['longitude'] - swath_width_deg/2, pos['latitude'] - along_track_deg/2),
                               swath_width_deg, along_track_deg,
                               facecolor=specs['color'], alpha=0.15,
                               edgecolor=specs['color'], linewidth=1.0)

                # Rotate rectangle to align with ground track
                t = Affine2D().rotate_deg_around(pos['longitude'], pos['latitude'], heading) + ax2.transData
                rect.set_transform(t)
                ax2.add_patch(rect)

            # EO: Narrow rectangle along ground track (push-broom scanner)
            elif sensor_type == 'EO':
                from matplotlib.patches import Rectangle
                from matplotlib.transforms import Affine2D

                swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
                along_track_deg = 100 / (111.0)  # 100km along-track (longer strip)
                heading = pos['ground_track_heading']

                rect = Rectangle((pos['longitude'] - swath_width_deg/2, pos['latitude'] - along_track_deg/2),
                               swath_width_deg, along_track_deg,
                               facecolor=specs['color'], alpha=0.15,
                               edgecolor=specs['color'], linewidth=1.0)

                # Rotate rectangle to align with ground track
                t = Affine2D().rotate_deg_around(pos['longitude'], pos['latitude'], heading) + ax2.transData
                rect.set_transform(t)
                ax2.add_patch(rect)

            # Mark satellite position
            ax2.plot(pos['longitude'], pos['latitude'],
                   'o', color=specs['color'], markersize=4, alpha=0.8)

# Plot vessels with larger markers
for mmsi in vessels_one_day['mmsi'].unique():
    vessel_data = vessels_one_day[vessels_one_day['mmsi'] == mmsi]
    vessel_name = vessel_data.iloc[0]['vessel_name']

    ais_on = vessel_data[vessel_data['ais_visible'] == True]
    ais_off = vessel_data[vessel_data['ais_visible'] == False]

    if len(ais_on) > 0:
        ax2.plot(ais_on['longitude'], ais_on['latitude'],
               'o', color='blue', markersize=5, alpha=0.6,
               label=f'{vessel_name} (AIS ON)', zorder=10)

    if len(ais_off) > 0:
        ax2.plot(ais_off['longitude'], ais_off['latitude'],
               'X', color='red', markersize=8, alpha=0.9,
               markeredgewidth=2, label=f'{vessel_name} (AIS OFF)', zorder=10)

# Legend
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(),
         loc='upper right', fontsize=10, framealpha=0.95)

# Add timestamp annotation
ax2.text(0.02, 0.02,
        f'Footprints shown every 10 minutes\nRF: 800km | SAR: 700km | EO: 500km',
        transform=ax2.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save zoomed figure
output_file2 = output_dir / 'ground_tracks_zoomed_vessels.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✅ Saved zoomed visualization: {output_file2}")

print("\n✅ Visualization complete!")
print(f"\nGenerated files:")
print(f"  1. {output_file}")
print(f"  2. {output_file2}")
