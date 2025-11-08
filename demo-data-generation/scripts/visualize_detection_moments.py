#!/usr/bin/env python3
"""
Visualize satellite footprints at the exact moments when detections occurred.
This shows that satellites were actually over the vessels when detections happened.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from pathlib import Path
from datetime import datetime

# Sensor specifications
SENSOR_SPECS = {
    'RF': {'swath_km': 800, 'color': 'orange'},
    'SAR': {'swath_km': 700, 'color': 'green'},
    'EO': {'swath_km': 500, 'color': 'cyan'}
}

# Load ground truth events (detections)
events_file = Path(__file__).parent.parent.parent / 'vessel-simulation-service/gold_standard_bad_vessel_sim_output/ground_truth_events.parquet'
events = pd.read_parquet(events_file)
detections = events[events['event_type'].str.contains('satellite_detection')].copy()
detections['timestamp_dt'] = pd.to_datetime(detections['timestamp'], unit='s')
detections['sensor_type'] = detections['sensor_type'].str.upper()

print(f"✅ Loaded {len(detections)} detections")

# Load satellite positions
sat_positions = pd.read_parquet(Path(__file__).parent.parent / 'outputs/satellite_positions_ecef.parquet')
sat_positions['timestamp_dt'] = pd.to_datetime(sat_positions['timestamp']).dt.tz_localize(None)

print(f"✅ Loaded {len(sat_positions)} satellite positions")

# Create figure
fig, ax = plt.subplots(figsize=(16, 14))

# Set up map bounds (zoom to vessel area)
vessel_lon_min = detections['longitude'].min() - 1
vessel_lon_max = detections['longitude'].max() + 1
vessel_lat_min = detections['latitude'].min() - 1
vessel_lat_max = detections['latitude'].max() + 1

ax.set_xlim(vessel_lon_min, vessel_lon_max)
ax.set_ylim(vessel_lat_min, vessel_lat_max)
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Satellite Footprints at Detection Moments\nShowing actual satellite coverage when detections occurred',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal')

# Plot detection locations as reference points
for vessel_name in detections['vessel_name'].unique():
    vessel_dets = detections[detections['vessel_name'] == vessel_name]
    ax.scatter(vessel_dets['longitude'], vessel_dets['latitude'],
              c='red', s=60, alpha=0.5, marker='x', linewidths=2,
              label=f'{vessel_name} (dark)', zorder=2)

# Plot satellite footprints at detection moments
print("\nProcessing detections:")
for idx, detection in detections.iterrows():
    # Find the satellite position at this exact detection time
    sat_id = detection['satellite_id']
    det_time = detection['timestamp_dt']
    sensor_type = detection['sensor_type']

    # Find closest satellite position (within 1 minute tolerance)
    sat_at_time = sat_positions[
        (sat_positions['satellite_id'] == sat_id) &
        (np.abs((sat_positions['timestamp_dt'] - det_time).dt.total_seconds()) < 60)
    ]

    if len(sat_at_time) == 0:
        print(f"⚠️  No satellite position found for {sat_id} at {det_time}")
        continue

    sat_pos = sat_at_time.iloc[0]
    specs = SENSOR_SPECS[sensor_type]

    print(f"  {det_time.strftime('%H:%M:%S')} | {sensor_type:3} | {sat_id:12} | "
          f"Sat: ({sat_pos['latitude']:.2f}°, {sat_pos['longitude']:.2f}°) | "
          f"Vessel: ({detection['latitude']:.2f}°, {detection['longitude']:.2f}°)")

    # Calculate footprint
    lat_rad = np.radians(sat_pos['latitude'])

    if sensor_type == 'RF':
        # Circular footprint
        radius_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
        circle = Circle((sat_pos['longitude'], sat_pos['latitude']),
                      radius_deg,
                      facecolor=specs['color'],
                      alpha=0.2,
                      edgecolor=specs['color'],
                      linewidth=2,
                      zorder=3)
        ax.add_patch(circle)

    elif sensor_type == 'SAR':
        # Rectangle perpendicular to ground track
        swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
        along_track_deg = 50 / 111.0  # 50km along-track
        heading = sat_pos['ground_track_heading']

        rect = Rectangle((sat_pos['longitude'] - swath_width_deg/2, sat_pos['latitude'] - along_track_deg/2),
                       swath_width_deg, along_track_deg,
                       facecolor=specs['color'], alpha=0.2,
                       edgecolor=specs['color'], linewidth=2, zorder=3)

        t = Affine2D().rotate_deg_around(sat_pos['longitude'], sat_pos['latitude'], heading) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    elif sensor_type == 'EO':
        # Narrow rectangle along ground track
        swath_width_deg = specs['swath_km'] / (111.0 * np.cos(lat_rad))
        along_track_deg = 100 / 111.0  # 100km along-track
        heading = sat_pos['ground_track_heading']

        rect = Rectangle((sat_pos['longitude'] - swath_width_deg/2, sat_pos['latitude'] - along_track_deg/2),
                       swath_width_deg, along_track_deg,
                       facecolor=specs['color'], alpha=0.2,
                       edgecolor=specs['color'], linewidth=2, zorder=3)

        t = Affine2D().rotate_deg_around(sat_pos['longitude'], sat_pos['latitude'], heading) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # Mark satellite position
    ax.plot(sat_pos['longitude'], sat_pos['latitude'],
           'o', color=specs['color'], markersize=10, markeredgecolor='black',
           markeredgewidth=1.5, zorder=4, label=f'{sensor_type}')

    # Mark detection location (vessel position at detection time)
    ax.plot(detection['longitude'], detection['latitude'],
           '*', color='yellow', markersize=15, markeredgecolor='black',
           markeredgewidth=1, zorder=5)

# Add timestamp annotations for each unique detection time
unique_times = detections.groupby('timestamp_dt').first().reset_index()
for idx, det in unique_times.iterrows():
    time_str = det['timestamp_dt'].strftime('%H:%M')
    ax.annotate(time_str,
               xy=(det['longitude'], det['latitude']),
               xytext=(10, 10), textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
               zorder=6)

# Clean up legend (remove duplicates)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
         loc='upper right', fontsize=10, framealpha=0.9)

# Add annotation explaining the visualization
ax.text(0.02, 0.02,
       'Yellow stars = detection locations\n'
       'Colored circles/rectangles = sensor footprints at detection time\n'
       'Colored dots = satellite positions at detection time',
       transform=ax.transAxes,
       fontsize=9, verticalalignment='bottom',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / 'outputs'
output_file = output_dir / 'detection_moments_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")

plt.show()
