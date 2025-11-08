#!/usr/bin/env python3
"""
Export satellite and vessel data to CZML for Cesium visualization.
Shows satellites at altitude with sensor cones and ground tracks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json

# Load data
print("Loading data...")
sat_positions = pd.read_parquet(Path(__file__).parent.parent / 'outputs/satellite_positions_ecef.parquet')
sat_positions['timestamp_dt'] = pd.to_datetime(sat_positions['timestamp']).dt.tz_localize(None)

events = pd.read_parquet(Path(__file__).parent.parent.parent / 'vessel-simulation-service/gold_standard_bad_vessel_sim_output/ground_truth_events.parquet')
events['timestamp_dt'] = pd.to_datetime(events['timestamp'], unit='s')

# Filter to dark period only (July 18, 04:00 - 09:00 UTC)
start_time = pd.Timestamp('2025-07-18 04:00:00')
end_time = pd.Timestamp('2025-07-18 09:00:00')

sat_filtered = sat_positions[(sat_positions['timestamp_dt'] >= start_time) &
                             (sat_positions['timestamp_dt'] <= end_time)]

print(f"✅ Filtered to {len(sat_filtered)} satellite positions")

# Color map
SENSOR_COLORS = {
    'RF': {'rgba': [255, 165, 0, 150]},   # Orange
    'SAR': {'rgba': [0, 255, 0, 150]},    # Green
    'EO': {'rgba': [0, 255, 255, 150]}    # Cyan
}

def to_iso8601(dt):
    """Convert datetime to ISO8601 string"""
    if pd.isna(dt):
        return None
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Build CZML document
czml = []

# Document header
czml.append({
    "id": "document",
    "name": "ELK Satellite Coverage - Dark Vessels",
    "version": "1.0",
    "clock": {
        "interval": f"{to_iso8601(start_time)}/{to_iso8601(end_time)}",
        "currentTime": to_iso8601(start_time),
        "multiplier": 60,  # 60x speed
        "range": "LOOP_STOP",
        "step": "SYSTEM_CLOCK_MULTIPLIER"
    }
})

print("\nCreating satellite entities...")

# Create satellite tracks
for sat_id in sat_filtered['satellite_id'].unique():
    sat_data = sat_filtered[sat_filtered['satellite_id'] == sat_id].sort_values('timestamp_dt')
    sensor_type = sat_data.iloc[0]['sensor_type'].upper()
    color = SENSOR_COLORS[sensor_type]['rgba']

    # Build position array using cartographicDegrees: [time1, lon1, lat1, alt1, time2, lon2, lat2, alt2, ...]
    positions = []
    for _, row in sat_data.iterrows():
        epoch_seconds = (row['timestamp_dt'] - pd.Timestamp('1970-01-01')).total_seconds()
        positions.extend([
            epoch_seconds,
            row['longitude'],
            row['latitude'],
            row['altitude_m']  # Already in meters
        ])

    # Satellite entity
    czml.append({
        "id": sat_id,
        "name": f"{sat_id} ({sensor_type})",
        "availability": f"{to_iso8601(sat_data['timestamp_dt'].min())}/{to_iso8601(sat_data['timestamp_dt'].max())}",
        "position": {
            "epoch": to_iso8601(sat_data['timestamp_dt'].min()),
            "cartographicDegrees": positions,
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 1
        },
        "point": {
            "pixelSize": 20,
            "color": {
                "rgba": color
            },
            "outlineColor": {
                "rgba": [255, 255, 255, 255]
            },
            "outlineWidth": 3,
            "scaleByDistance": {
                "nearFarScalar": [1000, 2.0, 10000000, 0.5]
            },
            "disableDepthTestDistance": "Infinity"
        },
        "billboard": {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAACMSURBVHgBnZLBDYAwDAOdgTEYg1kYgTEYg1kYo5WQkPJCfHqS5cR2DoA0EZGZiIwxhmVZwN0BYK01lFJAKaWUUkoppZRSSimllFJKKf+llFJKKaWUUkopv5RSSimllFJK+S2llFJKKaWU8ltKKaWUUkoppfyWUkoppZRSym8ppZRSSiml/JZSSimllFL+pxfQKjYVX8yqrAAAAABJRU5ErkJggg==",
            "scale": 1.5,
            "horizontalOrigin": "CENTER",
            "verticalOrigin": "CENTER"
        },
        "label": {
            "text": f"{sensor_type}",
            "font": "11pt sans-serif",
            "fillColor": {
                "rgba": [255, 255, 255, 255]
            },
            "outlineColor": {
                "rgba": [0, 0, 0, 255]
            },
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {
                "cartesian2": [0, -12]
            }
        },
        "path": {
            "material": {
                "solidColor": {
                    "color": {
                        "rgba": color
                    }
                }
            },
            "width": 2,
            "leadTime": 0,
            "trailTime": 300,  # 5 minute trail
            "resolution": 60
        }
    })

    print(f"  Added {sat_id} with {len(sat_data)} positions")

print("\nCreating vessel entities...")

# Create vessel entities from detection events
detections = events[events['event_type'].str.contains('satellite_detection')]

for vessel_name in detections['vessel_name'].unique():
    vessel_dets = detections[detections['vessel_name'] == vessel_name].sort_values('timestamp')

    # Build position array using cartographicDegrees: [time1, lon1, lat1, alt1, time2, lon2, lat2, alt2, ...]
    positions = []
    for _, row in vessel_dets.iterrows():
        epoch_seconds = row['timestamp']
        positions.extend([
            epoch_seconds,
            row['longitude'],
            row['latitude'],
            0  # Vessels at sea level
        ])

    czml.append({
        "id": f"vessel_{vessel_name}",
        "name": vessel_name,
        "availability": f"{to_iso8601(vessel_dets['timestamp_dt'].min())}/{to_iso8601(vessel_dets['timestamp_dt'].max())}",
        "position": {
            "epoch": to_iso8601(vessel_dets['timestamp_dt'].min()),
            "cartographicDegrees": positions,
            "interpolationAlgorithm": "LINEAR"
        },
        "point": {
            "pixelSize": 25,
            "color": {
                "rgba": [255, 0, 0, 255]  # Red
            },
            "outlineColor": {
                "rgba": [255, 255, 255, 255]
            },
            "outlineWidth": 4,
            "scaleByDistance": {
                "nearFarScalar": [1000, 2.0, 10000000, 0.5]
            },
            "disableDepthTestDistance": "Infinity"
        },
        "label": {
            "text": vessel_name,
            "font": "12pt sans-serif",
            "fillColor": {
                "rgba": [255, 255, 255, 255]
            },
            "outlineColor": {
                "rgba": [0, 0, 0, 255]
            },
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {
                "cartesian2": [0, -15]
            }
        }
    })

    print(f"  Added vessel {vessel_name} with {len(vessel_dets)} detection points")

# Save CZML
output_file = Path(__file__).parent.parent / 'outputs/satellite_coverage.czml'
with open(output_file, 'w') as f:
    json.dump(czml, f, indent=2)

print(f"\n✅ Saved CZML: {output_file}")
print(f"   Satellites: {len(sat_filtered['satellite_id'].unique())}")
print(f"   Vessels: {len(detections['vessel_name'].unique())}")
print(f"   Time span: {start_time} to {end_time}")
