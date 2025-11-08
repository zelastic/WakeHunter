#!/usr/bin/env python3
"""
Fake Weather API for Demo Data

Serves cloud cover and weather condition data for the simulation period.
This provides the weather context that drives satellite tasking decisions.

Timeline: July 17 20:45 - July 20 07:00 UTC (58.25 hours)
Narrative:
- July 17 21:00 - July 18 03:00: Cloudy (approach phase) -> triggers SAR imaging
- July 18 03:00 - July 18 09:00: Clearing (transfer phase)
- July 18 09:00+: Clear (departure phase) -> triggers EO visual confirmation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Maritime Weather API",
    description="Fake weather API for WakeHunter demo data",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_weather_for_timestamp(timestamp_str: str):
    """
    Get weather conditions for a given timestamp.

    Weather Timeline (matches detection event generation):
    - July 17 21:00 - July 18 03:00: Cloudy (80% cloud cover) - approach phase
    - July 18 03:00 - July 18 09:00: Clearing (40% cloud cover) - transfer phase
    - July 18 09:00+: Clear (10% cloud cover) - departure phase
    """
    # Parse timestamp
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str.removesuffix('Z')

    ts = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

    # Define weather transition times
    cloudy_start = datetime(2025, 7, 17, 21, 0, 0, tzinfo=timezone.utc)
    clearing_start = datetime(2025, 7, 18, 3, 0, 0, tzinfo=timezone.utc)
    clear_start = datetime(2025, 7, 18, 9, 0, 0, tzinfo=timezone.utc)

    # Determine weather based on timestamp
    if ts < cloudy_start:
        condition = 'partly_cloudy'
        cloud_cover_pct = 30
        visibility_km = 15.0
        temperature_c = 26.0
    elif ts < clearing_start:
        condition = 'cloudy'
        cloud_cover_pct = 80
        visibility_km = 8.0
        temperature_c = 24.0
    elif ts < clear_start:
        condition = 'partly_cloudy'
        cloud_cover_pct = 40
        visibility_km = 12.0
        temperature_c = 25.0
    else:
        condition = 'clear'
        cloud_cover_pct = 10
        visibility_km = 25.0
        temperature_c = 28.0

    return {
        'timestamp': ts.isoformat() + 'Z',
        'location': {
            'lat': 32.5,
            'lon': 126.0,
            'name': 'East China Sea'
        },
        'weather': {
            'condition': condition,
            'cloud_cover_pct': cloud_cover_pct,
            'visibility_km': visibility_km,
            'temperature_c': temperature_c,
            'wind_speed_knots': 12,
            'wind_direction_deg': 180,
            'sea_state': 'moderate'
        },
        'satellite_tasking': {
            'sar_recommended': cloud_cover_pct > 50,  # SAR works in all weather
            'eo_recommended': cloud_cover_pct < 30,   # EO needs clear skies
            'rf_recommended': True  # RF always works
        }
    }


@app.get("/")
def root():
    """API root endpoint."""
    return {
        "service": "Maritime Weather API",
        "version": "1.0.0",
        "description": "Fake weather API for WakeHunter demo data",
        "endpoints": {
            "/weather": "Get weather for specific timestamp",
            "/weather/range": "Get weather for time range",
            "/health": "Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/weather")
def get_weather(
    timestamp: str = Query(..., description="ISO 8601 timestamp (e.g., 2025-07-18T00:00:00Z)"),
    lat: Optional[float] = Query(None, description="Latitude (ignored for demo)"),
    lon: Optional[float] = Query(None, description="Longitude (ignored for demo)")
):
    """
    Get weather conditions for a specific timestamp.

    Parameters:
    - timestamp: ISO 8601 timestamp (e.g., 2025-07-18T00:00:00Z)
    - lat: Latitude (optional, ignored for demo - always returns ECS weather)
    - lon: Longitude (optional, ignored for demo)

    Returns:
    - Weather conditions including cloud cover, visibility, temperature, etc.
    - Satellite tasking recommendations based on weather
    """
    try:
        weather_data = get_weather_for_timestamp(timestamp)
        return weather_data
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {str(e)}")


@app.get("/weather/range")
def get_weather_range(
    start_time: str = Query(..., description="Start timestamp (ISO 8601)"),
    end_time: str = Query(..., description="End timestamp (ISO 8601)"),
    interval_minutes: int = Query(60, description="Interval between data points in minutes"),
    lat: Optional[float] = Query(None, description="Latitude (ignored for demo)"),
    lon: Optional[float] = Query(None, description="Longitude (ignored for demo)")
):
    """
    Get weather conditions for a time range.

    Parameters:
    - start_time: Start timestamp (ISO 8601)
    - end_time: End timestamp (ISO 8601)
    - interval_minutes: Interval between data points (default: 60 minutes)
    - lat: Latitude (optional, ignored for demo)
    - lon: Longitude (optional, ignored for demo)

    Returns:
    - Array of weather conditions at specified intervals
    """
    try:
        # Parse timestamps
        start = datetime.fromisoformat(start_time.removesuffix('Z')).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_time.removesuffix('Z')).replace(tzinfo=timezone.utc)

        # Generate weather data for each interval
        weather_data = []
        current = start
        from datetime import timedelta

        while current <= end:
            ts_str = current.isoformat()
            weather = get_weather_for_timestamp(ts_str)
            weather_data.append(weather)
            current += timedelta(minutes=interval_minutes)

        return {
            'start_time': start_time,
            'end_time': end_time,
            'interval_minutes': interval_minutes,
            'count': len(weather_data),
            'data': weather_data
        }

    except Exception as e:
        logger.error(f"Error getting weather range: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")


@app.get("/weather/summary")
def get_weather_summary(
    start_time: str = Query("2025-07-17T20:45:00Z", description="Start timestamp"),
    end_time: str = Query("2025-07-20T07:00:00Z", description="End timestamp")
):
    """
    Get weather summary for the simulation period.

    Returns overview of weather conditions during the simulation.
    """
    return {
        'simulation_period': {
            'start': start_time,
            'end': end_time,
            'duration_hours': 58.25
        },
        'weather_timeline': [
            {
                'period': 'Early Approach',
                'time': '2025-07-17T20:45:00Z - 2025-07-17T21:00:00Z',
                'condition': 'partly_cloudy',
                'cloud_cover_pct': 30,
                'satellite_tasking': 'RF detection, limited EO'
            },
            {
                'period': 'Approach Phase',
                'time': '2025-07-17T21:00:00Z - 2025-07-18T03:00:00Z',
                'condition': 'cloudy',
                'cloud_cover_pct': 80,
                'satellite_tasking': 'RF detection triggers SAR imaging (all-weather)'
            },
            {
                'period': 'Transfer Phase',
                'time': '2025-07-18T03:00:00Z - 2025-07-18T09:00:00Z',
                'condition': 'clearing',
                'cloud_cover_pct': 40,
                'satellite_tasking': 'RF + SAR continued, some EO possible'
            },
            {
                'period': 'Departure Phase',
                'time': '2025-07-18T09:00:00Z - 2025-07-20T07:00:00Z',
                'condition': 'clear',
                'cloud_cover_pct': 10,
                'satellite_tasking': 'Full multi-sensor coverage: RF + SAR + EO visual confirmation'
            }
        ],
        'narrative': {
            'cloudy_conditions': 'Cloudy conditions during approach phase trigger SAR all-weather imaging',
            'clear_conditions': 'Clear conditions during departure enable EO visual confirmation of STS transfer',
            'rf_detection': 'RF sensors detect radio emissions throughout, providing initial alerts'
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Maritime Weather API on port 8004...")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
