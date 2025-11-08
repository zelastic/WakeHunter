
# Demo Data Generation Summary Report

## Dataset Overview

**Generation Date:** 2025-11-05 19:10:13 UTC

**Simulation Period:**
- Start: 2025-07-17T20:45:00Z
- End: 2025-07-20T07:00:00Z
- Duration: 58.25 hours

## Vessel Tracks

**Total Position Reports:** 6,992

**Vessels:**

- **SANCTIONED VESSEL** (MMSI: 412999999)
  - Type: Cargo
  - Positions: 3,496
  - AIS OFF: 916 positions (26.2%)
  - Lat range: 32.50° to 39.12°
  - Lon range: 125.99° to 134.92°

- **YUE CHI** (MMSI: 477105700)
  - Type: Tanker
  - Positions: 3,496
  - AIS OFF: 0 positions (0.0%)
  - Lat range: 25.34° to 32.51°
  - Lon range: 124.97° to 133.65°

## Satellite Detections

**Total Detections:** 96

**By Sensor Type:**
- RF: 44 detections
- SAR: 24 detections
- EO: 28 detections

**Unique Satellites:** 9

**Weather Conditions:**
- clear: 72 detections (75.0%)
- partly_cloudy: 14 detections (14.6%)
- cloudy: 10 detections (10.4%)

## Detection-Vessel Correlation

**Vessels Detected:**
- SANCTIONED VESSEL (MMSI: 412999999): 48 detections
- YUE CHI (MMSI: 477105700): 48 detections

## Data Quality

✅ All vessel positions have valid coordinates
✅ All detections match known vessels
✅ All detections within vessel track time range
✅ Multi-sensor correlation maintained
✅ Weather-driven tasking narrative consistent

## Visualizations Generated

1. `vessel_tracks_with_detections.png` - Vessel tracks and satellite detection overlay
2. `detection_timeline.png` - Temporal analysis of AIS status and detections
3. `detection_correlation.png` - Statistical analysis of detection patterns
4. `ground_tracks_visualization.png` - Satellite ground tracks
5. `pass_timeline_visualization.png` - Satellite pass timeline

## Next Steps

1. Export data to debug dashboard
2. Ingest into Elasticsearch
3. Validate STS detection narrative
4. Test analytical queries
