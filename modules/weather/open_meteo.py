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
    SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
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
        # A forecast model run is not usable at the exact instant it starts.
        # Keep a conservative buffer in historical backtests so they cannot
        # accidentally use a run that would still have been publishing.
        self.availability_lag_minutes = weather.get(
            "historical_availability_lag_minutes", 120
        )
        self.single_run_forecast_days = weather.get(
            "single_run_forecast_days", 3
        )

        # Walk-forward bias correction (validated 2026-07-27, see
        # tests/test_weather_bias_experiment.py and the audit that
        # motivated it: Open-Meteo over-forecast sunlight here on 18
        # of 21 days, +49% during the monsoon week).
        bias = weather.get("bias_correction", {})
        self.bias_enabled = bias.get("enabled", False)
        self.bias_window_days = bias.get("window_days", 5)
        self.bias_clip = tuple(bias.get("clip", [0.55, 1.15]))
        self.bias_min_days = bias.get("min_days", 3)
        self.bias_lookback_days = bias.get("lookback_days", 10)

        self._bias_ratio_cache = {}   # date -> ratio or None (memoized)

    # --------------------------------------------------

    def _cache_path(self, date_str, model_run=None):
        if model_run is None:
            return self.cache_dir / f"{date_str}.json"

        label = pd.Timestamp(model_run).strftime("%Y%m%dT%H%M")
        return self.cache_dir / f"{date_str}__run_{label}.json"

    # --------------------------------------------------

    def select_historical_model_run(self, as_of):
        """Return the latest 6-hourly UTC model cycle safely available.

        This is deliberately based on *when the forecast was made*, not the
        target timestamp being predicted.  The previous implementation used a
        stitched historical series whose later blocks could come from model
        updates issued after a simulated forecast run.
        """

        as_of = pd.Timestamp(as_of)

        if as_of.tzinfo is None:
            as_of = as_of.tz_localize(self.timezone)
        else:
            as_of = as_of.tz_convert(self.timezone)

        available_utc = (
            as_of.tz_convert("UTC")
            - pd.Timedelta(minutes=self.availability_lag_minutes)
        )

        return available_utc.floor("6h").tz_localize(None)

    # --------------------------------------------------

    def fetch_day(self, date_str, as_of=None):
        """
        Hourly forecasted GHI / cloud cover / temperature for
        one day, as a DataFrame indexed by (tz-naive local)
        timestamp.  For historical point-in-time evaluation, ``as_of`` selects
        one archived model run that was available before the forecast time;
        it never uses the stitched historical series. Cached to disk. Returns
        None on failure so callers degrade to no-weather rather than crashing.
        """

        target = pd.Timestamp(date_str).date()
        today = date_cls.today()

        # Historical evaluation needs a specific archived forecast issue time.
        # Live/current forecasts still use the regular live endpoint.
        model_run = None
        if as_of is not None and target < today:
            model_run = self.select_historical_model_run(as_of)

        cache_path = self._cache_path(date_str, model_run)

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            data = self._request(date_str, model_run=model_run)
            if data is None:
                return None
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as file:
                json.dump(data, file)

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            return None

        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "ghi": hourly.get("shortwave_radiation", [np.nan] * len(times)),
            "cloud_cover": hourly.get("cloud_cover", [np.nan] * len(times)),
        })

        # A single archived run returns its full horizon. Keep only the day
        # requested by the caller, including when the run began the evening
        # before the target date.
        return frame[frame["timestamp"].dt.date == target].reset_index(drop=True)

    # --------------------------------------------------

    def _request(self, date_str, model_run=None):

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
            "timezone": self.timezone,
        }

        if model_run is None:
            # Today/future: use the currently available live forecast.
            url = self.FORECAST_URL
            params["start_date"] = date_str
            params["end_date"] = date_str
        else:
            # Historical: preserve a single model run.  Open-Meteo's regular
            # historical endpoint stitches later runs into one series, which
            # is unsuitable for a point-in-time forecast backtest.
            url = self.SINGLE_RUN_URL
            params["run"] = pd.Timestamp(model_run).strftime("%Y-%m-%dT%H:%M")
            params["forecast_days"] = self.single_run_forecast_days

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except Exception as error:
            self.logger.warning(f"Open-Meteo request failed for {date_str}: {error}")
            return None

    # --------------------------------------------------

    def daily_bias_ratio(self, real_data, day):
        """
        Forecast/actual sunlight ratio for one finished day: the
        archived 06:45 forecast for that day vs the plant's measured
        GHI over its real daylight blocks. 1.0 = honest, 1.5 = the
        forecast promised 50% more sun than arrived. None when either
        side is missing. Memoized - a finished day's ratio never
        changes.
        """

        if day in self._bias_ratio_cache:
            return self._bias_ratio_cache[day]

        ratio = None

        group = real_data[real_data["timestamp"].dt.date == day]
        daylight = group[group["ghi_w_m2"] > 30].dropna(subset=["ghi_w_m2"])

        if len(daylight) >= 15:
            as_of = pd.Timestamp(day) + pd.Timedelta(hours=6, minutes=45)
            forecast = self.forecast_ghi_at(daylight["timestamp"], as_of=as_of)

            if forecast is not None:
                actual = daylight["ghi_w_m2"].to_numpy()
                valid = ~np.isnan(forecast)

                if valid.sum() >= 15 and actual[valid].sum() > 0:
                    ratio = float(forecast[valid].sum() / actual[valid].sum())

        self._bias_ratio_cache[day] = ratio

        return ratio

    # --------------------------------------------------

    def bias_factor(self, dataframe, run_time):
        """
        Walk-forward correction for today's weather signal: the
        inverse of the MEDIAN forecast/actual bias over the last
        `bias_window_days` finished days (median so one freak day
        cannot own the factor; clipped so the correction can never
        run away).

        Validated walk-forward on 16 unseen days (2026-07-27):
        +0.76 pts overall, +1.23 pts on monsoon days, 11/16 days
        better. Uses only information available at run_time.
        """

        if not self.bias_enabled:
            return 1.0

        if "ghi_w_m2" not in dataframe.columns:
            return 1.0

        real = dataframe
        if "is_real_measurement" in real.columns:
            real = real[real["is_real_measurement"].fillna(False)]

        today = pd.Timestamp(run_time).date()
        start = today - pd.Timedelta(days=self.bias_lookback_days)

        prior_days = sorted({
            d for d in real["timestamp"].dt.date
            if start <= d < today
        })

        ratios = [
            r for d in prior_days
            if (r := self.daily_bias_ratio(real, d)) is not None
        ]

        if len(ratios) < self.bias_min_days:
            return 1.0

        recent = ratios[-self.bias_window_days:]

        return float(np.clip(1.0 / np.median(recent), *self.bias_clip))

    # --------------------------------------------------

    def forecast_ghi_at(self, timestamps, as_of=None):
        """
        Forecasted GHI aligned to the given 15-minute
        timestamps (hourly forecast interpolated). Returns
        None if the forecast is unavailable.
        """

        timestamps = pd.DatetimeIndex(timestamps)

        dates = sorted({ts.strftime("%Y-%m-%d") for ts in timestamps})

        frames = [self.fetch_day(d, as_of=as_of) for d in dates]
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
