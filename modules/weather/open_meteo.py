"""
=========================================================
Solar Forecasting Project
Open-Meteo Weather Client
=========================================================
Free weather-forecast feed (no API key) that supplies the
one thing the rest of the pipeline lacks: a FORWARD-LOOKING
estimate of sunlight for the rest of the day.

We use forecasted shortwave radiation (GHI, W/m2), NOT the
cloud-cover percentage - cloud_cover counts thin high cirrus
that barely dims the sun, whereas GHI is the actual usable
radiation (confirmed on 2026-07-14: cloud_cover said 100%
but GHI and the plant meter both showed a clear day).

Honesty for backtesting: past dates use the "historical
forecast" API (the forecast that was ISSUED at the time),
never the reanalysis/archive (which is the recorded truth
and would be lookahead cheating). Today/future use the live
forecast API. Responses are cached to disk per date so a
backtest hits the network once per day, not once per run.
=========================================================
"""

import json
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.config import settings
from utils.logger import get_logger


class OpenMeteoClient:

    HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):

        self.logger = get_logger()

        plant = settings["plant"]
        self.latitude = plant["latitude"]
        self.longitude = plant["longitude"]
        self.timezone = plant["timezone"]

        weather = settings.get("weather", {})
        self.enabled = weather.get("enabled", False)
        self.weather_weight = weather.get("weather_weight", 0.0)
        self.cache_dir = Path(weather.get("cache_dir", "data/weather/openmeteo_cache"))
        self.timeout = weather.get("timeout_seconds", 25)

    # --------------------------------------------------

    def _cache_path(self, date_str):
        return self.cache_dir / f"{date_str}.json"

    # --------------------------------------------------

    def fetch_day(self, date_str):
        """
        Hourly forecasted GHI / cloud cover / temperature for
        one day, as a DataFrame indexed by (tz-naive local)
        timestamp. Cached to disk. Returns None on failure so
        callers degrade to no-weather rather than crashing.
        """

        cache_path = self._cache_path(date_str)

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            data = self._request(date_str)
            if data is None:
                return None
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as file:
                json.dump(data, file)

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            return None

        return pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "ghi": hourly.get("shortwave_radiation", [np.nan] * len(times)),
            "cloud_cover": hourly.get("cloud_cover", [np.nan] * len(times)),
        })

    # --------------------------------------------------

    def _request(self, date_str):

        target = pd.Timestamp(date_str).date()
        today = date_cls.today()

        # Past days -> the forecast that was issued then (no lookahead).
        # Today/future -> the live forecast.
        url = self.HISTORICAL_URL if target < today else self.FORECAST_URL

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
            "timezone": self.timezone,
            "start_date": date_str,
            "end_date": date_str,
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except Exception as error:
            self.logger.warning(f"Open-Meteo request failed for {date_str}: {error}")
            return None

    # --------------------------------------------------

    def forecast_ghi_at(self, timestamps):
        """
        Forecasted GHI aligned to the given 15-minute
        timestamps (hourly forecast interpolated). Returns
        None if the forecast is unavailable.
        """

        timestamps = pd.DatetimeIndex(timestamps)

        dates = sorted({ts.strftime("%Y-%m-%d") for ts in timestamps})

        frames = [self.fetch_day(d) for d in dates]
        frames = [f for f in frames if f is not None]

        if not frames:
            return None

        hourly = pd.concat(frames, ignore_index=True)
        hourly = hourly.dropna(subset=["ghi"]).drop_duplicates("timestamp")

        if hourly.empty:
            return None

        series = hourly.set_index("timestamp")["ghi"].sort_index()

        # Interpolate the hourly forecast onto the 15-min blocks
        union = series.index.union(timestamps)
        interpolated = series.reindex(union).interpolate(method="time")

        return interpolated.reindex(timestamps).to_numpy()
