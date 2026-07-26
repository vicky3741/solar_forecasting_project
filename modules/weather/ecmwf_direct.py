"""
=========================================================
Solar Forecasting Project
ECMWF Direct Client (no middleman)
=========================================================
Downloads the ECMWF forecast straight from ECMWF's own
open-data servers, decodes the GRIB files locally, and
returns forecasted GHI for the plant - the same numbers we
currently receive via Open-Meteo, but with ECMWF as the only
external dependency.

Built 2026-07-27 because the sub-mentor asked us to move off
Open-Meteo. The 21-day bake-off had already shown ECMWF is
the right *forecast* (MAE 107.7 W/m2 vs 118.9 for the
best_match default); this removes the delivery middleman too.

WHAT ECMWF PUBLISHES, AND THE TWO TRAPS
---------------------------------------
1. Radiation is published as `ssrd` - surface solar radiation
   downwards - ACCUMULATED from the start of the run, in
   J/m2. It is not W/m2 and not per-step. Converting needs a
   difference between consecutive steps divided by the step
   seconds:
       W/m2 = (ssrd[t] - ssrd[t-1]) / (seconds between them)
   Using ssrd raw would produce numbers that climb all day
   and are wrong by a factor of thousands.
2. The open-data feed is issued on a 6-hourly cycle but is
   NOT instantly available; a run appears roughly 7 hours
   after its nominal time. `latest_available_run` enforces
   that, so a live run never asks for a file that does not
   exist yet.

Grid: 0.25 degrees. The plant does not sit on a grid point,
so the four surrounding points are bilinearly interpolated.

Honest limitation, and why Open-Meteo stays installed:
ECMWF's open-data server keeps only the last few days of
runs. It therefore CANNOT support the historical backtests
that every change in this project is validated with. The
Open-Meteo client remains the backtesting source; this class
is for live forecasting.
=========================================================
"""

import bisect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from utils.logger import get_logger


# ECMWF publishes 3-hourly steps out to 144h on the open feed.
_STEP_HOURS = 3

# A run is nominally issued at 00/06/12/18 UTC but takes hours to
# publish. Anything younger than this is assumed not on the server yet.
_PUBLISH_LAG_HOURS = 7


class ECMWFDirectClient:

    def __init__(self):

        self.logger = get_logger()

        plant = settings["plant"]
        self.latitude = plant["latitude"]
        self.longitude = plant["longitude"]
        self.timezone = plant["timezone"]

        direct = settings.get("weather", {}).get("ecmwf_direct", {})

        self.enabled = direct.get("enabled", False)
        self.stream = direct.get("stream", "oper")
        self.model = direct.get("model", "ifs")
        self.forecast_hours = direct.get("forecast_hours", 48)
        self.cache_dir = Path(direct.get("cache_dir", "data/weather/ecmwf_direct"))

        self._client = None

    # --------------------------------------------------

    @property
    def client(self):

        if self._client is None:
            from ecmwf.opendata import Client
            self._client = Client(source="ecmwf", model=self.model)

        return self._client

    # --------------------------------------------------

    def latest_available_run(self, as_of=None):
        """
        The most recent run actually on the server.

        ASKS the server rather than assuming a publication lag. A
        fixed 7-hour guess produced a 404 on first use (it asked for
        the 12z run when only 06z had been published), and lag varies
        by cycle and load. `Client.latest` resolves what genuinely
        exists; the arithmetic guess is only the offline fallback.
        """

        try:
            return pd.Timestamp(
                self.client.latest(
                    param="ssrd", type="fc", step=_STEP_HOURS, stream=self.stream
                )
            ).tz_localize("UTC")

        except Exception as error:
            self.logger.warning(
                f"ECMWF latest-run lookup failed ({error}) - falling back to "
                f"now minus {_PUBLISH_LAG_HOURS}h"
            )

            now = pd.Timestamp(as_of or datetime.now(timezone.utc))
            now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

            return (now - pd.Timedelta(hours=_PUBLISH_LAG_HOURS)).floor("6h")

    # --------------------------------------------------

    def download_run(self, run_time):
        """
        Fetches accumulated `ssrd` for every step of one run into a
        local GRIB file. Cached - a run's contents never change once
        published. Returns the path, or None on failure.
        """

        run_time = pd.Timestamp(run_time)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / f"ssrd_{run_time:%Y%m%dT%H}.grib2"

        if target.exists() and target.stat().st_size > 0:
            return target

        steps = list(range(0, self.forecast_hours + 1, _STEP_HOURS))

        try:
            self.client.retrieve(
                date=run_time.strftime("%Y%m%d"),
                time=int(run_time.strftime("%H")),
                stream=self.stream,
                type="fc",
                param="ssrd",
                step=steps,
                target=str(target),
            )
        except Exception as error:
            self.logger.warning(f"ECMWF direct download failed ({error})")
            target.unlink(missing_ok=True)
            return None

        return target if target.exists() and target.stat().st_size else None

    # --------------------------------------------------

    def _interpolate_point(self, values, lats, lons):
        """
        Bilinear interpolation of a regular lat/lon grid at the plant.
        ECMWF grids run north-to-south, which the sorting here handles.
        """

        lat_axis = np.unique(lats)
        lon_axis = np.unique(lons)

        grid = values.reshape(len(lat_axis), len(lon_axis))

        # np.unique sorts ascending; if the source ran north->south the
        # rows must be flipped to match.
        if lats[0] > lats[-1]:
            grid = grid[::-1] if grid.shape[0] == len(lat_axis) else grid

        i = np.clip(bisect.bisect_left(lat_axis, self.latitude), 1, len(lat_axis) - 1)
        j = np.clip(bisect.bisect_left(lon_axis, self.longitude), 1, len(lon_axis) - 1)

        lat0, lat1 = lat_axis[i - 1], lat_axis[i]
        lon0, lon1 = lon_axis[j - 1], lon_axis[j]

        wy = 0.0 if lat1 == lat0 else (self.latitude - lat0) / (lat1 - lat0)
        wx = 0.0 if lon1 == lon0 else (self.longitude - lon0) / (lon1 - lon0)

        return float(
            grid[i - 1, j - 1] * (1 - wy) * (1 - wx)
            + grid[i - 1, j] * (1 - wy) * wx
            + grid[i, j - 1] * wy * (1 - wx)
            + grid[i, j] * wy * wx
        )

    # --------------------------------------------------

    def read_ghi_series(self, grib_path, run_time):
        """
        Decodes one run's GRIB into a (timestamp, ghi_w_m2) frame in
        plant-local time.

        ssrd is accumulated J/m2 since the run start, so the average
        power over each interval is the DIFFERENCE between consecutive
        accumulations divided by the interval length. The first step
        has nothing before it and is dropped.
        """

        import eccodes

        accumulated = []

        with open(grib_path, "rb") as file:
            while True:
                message = eccodes.codes_grib_new_from_file(file)
                if message is None:
                    break
                try:
                    step_hours = eccodes.codes_get(message, "step")
                    values = eccodes.codes_get_values(message)
                    lats = eccodes.codes_get_array(message, "latitudes")
                    lons = eccodes.codes_get_array(message, "longitudes")
                    accumulated.append(
                        (int(step_hours), self._interpolate_point(values, lats, lons))
                    )
                finally:
                    eccodes.codes_release(message)

        if len(accumulated) < 2:
            return None

        accumulated.sort()

        run_utc = pd.Timestamp(run_time)
        if run_utc.tzinfo is None:
            run_utc = run_utc.tz_localize("UTC")

        rows = []
        for (step_a, joules_a), (step_b, joules_b) in zip(accumulated, accumulated[1:]):

            seconds = (step_b - step_a) * 3600
            if seconds <= 0:
                continue

            watts = max(0.0, (joules_b - joules_a) / seconds)

            # Stamp the interval at its MIDPOINT: the value is an
            # average over the window, not an instant at its end.
            midpoint = run_utc + pd.Timedelta(hours=(step_a + step_b) / 2)

            rows.append({
                "timestamp": midpoint.tz_convert(self.timezone).tz_localize(None),
                "ghi": watts,
            })

        return pd.DataFrame(rows) if rows else None

    # --------------------------------------------------

    def forecast_ghi_at(self, timestamps, as_of=None):
        """
        Forecasted GHI at the given 15-minute timestamps, straight from
        ECMWF. Same signature as OpenMeteoClient.forecast_ghi_at, so
        the two are interchangeable. Returns None on any failure, so
        callers fall back rather than crash.
        """

        if not self.enabled:
            return None

        timestamps = pd.DatetimeIndex(timestamps)

        run_time = self.latest_available_run(as_of)

        grib_path = self.download_run(run_time)
        if grib_path is None:
            return None

        try:
            series = self.read_ghi_series(grib_path, run_time)
        except Exception as error:
            self.logger.warning(f"ECMWF GRIB decode failed ({error})")
            return None

        if series is None or series.empty:
            return None

        indexed = series.drop_duplicates("timestamp").set_index("timestamp")["ghi"].sort_index()

        # 3-hourly forecast -> our 15-minute blocks.
        # limit_area="inside" is essential: pandas' interpolate
        # forward-fills trailing gaps by default, so any block past the
        # last downloaded step silently inherited the last value. That
        # produced a dead-flat 487.9 W/m2 all afternoon AND through the
        # night on first test - a constant that looks plausible in a
        # table and is completely wrong. Outside the run's horizon we
        # must return NaN and let the caller fall back.
        union = indexed.index.union(timestamps)
        interpolated = (
            indexed.reindex(union)
            .interpolate(method="time", limit_area="inside")
        )

        result = interpolated.reindex(timestamps).to_numpy()

        if np.isnan(result).all():
            return None

        covered = np.count_nonzero(~np.isnan(result)) / len(result)

        if covered < 0.9:
            self.logger.warning(
                f"ECMWF run {run_time:%Y-%m-%d %HZ} covers only "
                f"{covered:.0%} of the requested blocks - falling back"
            )
            return None

        return result
