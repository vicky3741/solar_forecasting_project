"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Per-Block Feature Table
=========================================================
Turns everything we know at run time into ONE tidy table,
one row per 15-minute block, which is the file the LLM
fusion step reads (modules/fusion/llm_scheduler.py).

This is the "output file that also works like an input
file" in the new architecture. It keeps the shape of the
reference features log - block, time, layer features,
motion features, solar-geometry features - but replaces
the parts of it that were not measuring anything.

WHAT CHANGED FROM THE REFERENCE LOG, AND WHY
--------------------------------------------
The reference log described Windy through whole-frame pixel
statistics: clouds_avg_hue_deg, solarpower_avg_brightness,
and so on. Those cannot work, for reasons measured on
2026-08-09:

  * they average thousands of km of map that is not the
    plant, plus Windy's own UI chrome;
  * hue is circular, so an arithmetic mean of it is simply
    the wrong operation;
  * one clip produces one row, so every block downstream
    received IDENTICAL features - in the reference log
    blocks 47-54 were byte-identical and the resulting
    forecast collapsed to 1.9918 * sin(solar elevation).

modules/capture/windy_scraper.py now reads the real numbers
at the plant coordinates instead, so this table carries
windy_solarpower_w_m2, windy_clouds_pct, windy_rain_mm and
friends as actual physical values.

THE INTERPOLATION RULE (this is the important part)
---------------------------------------------------
Free-tier Windy is 3-HOURLY. The schedule is 15-minutely.
So Windy values must be spread across blocks somehow, and
the obvious way is wrong: linearly interpolating raw W/m2
between 08:30 and 11:30 draws a straight line through a
curve that is anything but straight near sunrise.

So the interpolation happens in CLEAR-SKY INDEX space:

    kt          = windy_solarpower / clearsky_ghi   (at each Windy step)
    kt_block    = linear interpolation of kt        (smooth, no daily shape)
    windy_block = kt_block * clearsky_ghi_block     (exact solar geometry)

kt is a cloudiness fraction with no sunrise/sunset shape of
its own, so interpolating it is safe; the shape then comes
back from pvlib, which is exact. This is the same reasoning
the existing predictor uses to blend in kt space.

Every interpolated row is FLAGGED. `windy_is_measured` says
whether that block sat on a real Windy step, and
`windy_gap_minutes` says how far the nearest real step was.
The LLM is told to trust a measured block more than one
interpolated two hours away from anything - which is exactly
the information the reference log threw away when it copied
one row across eight blocks without saying so.
=========================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from utils.logger import get_logger


# Compass letters Windy prints next to wind speed -> degrees the
# wind is coming FROM (meteorological convention).
_COMPASS_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

# Which scraped overlay becomes which column, and its unit.
_LAYER_COLUMNS = {
    "solarpower": "windy_solarpower_w_m2",
    "clouds": "windy_clouds_pct",
    "lclouds": "windy_lclouds_pct",
    "mclouds": "windy_mclouds_pct",
    "hclouds": "windy_hclouds_pct",
    "rain": "windy_rain_mm",
    "wind": "windy_wind_kt",
}


def epoch_seconds(values):
    """
    Seconds since the epoch for a datetime series, whatever its
    resolution.

    NOT `.astype("int64") / 1e9`. pandas 2.x keeps the resolution it
    parsed, and Windy's ISO strings come back as datetime64[us], so
    that expression returns MICROseconds divided by a billion - every
    interval 1000x too small. It fails silently and plausibly: the
    first version of this module reported a nearest-Windy-reading gap
    of 29770 minutes and interpolated a flat kt of 0.41 across the
    whole day, which is exactly the "every block identical" failure
    this pipeline exists to avoid.

    Subtracting a Timestamp and dividing by a Timedelta is
    resolution-independent, so it cannot drift when pandas changes
    what it hands back.
    """

    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))

    return (
        (index - pd.Timestamp("1970-01-01", tz="UTC"))
        / pd.Timedelta(seconds=1)
    ).to_numpy(dtype=float)


class WindyFeatureBuilder:

    def __init__(self):

        self.logger = get_logger()

        self.clearsky = ClearSkyModel()

        plant = settings["plant"]

        self.timezone = plant["timezone"]
        self.capacity_mw = plant["capacity_mw"]
        self.plant_tag = plant.get("code", "SIRMOUR")

        self.interval_minutes = settings["forecast"]["interval_minutes"]

        scrape = settings.get("windy_scrape", {})

        self.values_dir = Path(scrape.get("output_dir", "data/windy/values"))

        self.output_dir = Path(
            scrape.get("features_dir", "data/windy/features")
        )

    # --------------------------------------------------

    def latest_values_file(self, run_time=None):
        """
        The newest scraped-values file at or before run_time.

        Bounded by run_time on purpose: a backtest that picked up a
        file scraped later in the day would be reading the future,
        and every accuracy number after that would be fiction.
        """

        if not self.values_dir.exists():
            return None

        candidates = []

        for path in self.values_dir.glob("*_values.json"):

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                scraped_at = pd.Timestamp(payload["scraped_at"])
            except Exception:
                continue

            # The scraper stamps local wall-clock time with no zone.
            # run_time is tz-aware, and comparing the two raises - so
            # the naive side is localized rather than the aware side
            # stripped, which would shift every comparison by 5.5h.
            if scraped_at.tz is None:
                scraped_at = scraped_at.tz_localize(self.timezone)

            if run_time is not None:

                limit = pd.Timestamp(run_time)

                if limit.tz is None:
                    limit = limit.tz_localize(self.timezone)

                if scraped_at > limit:
                    continue

            candidates.append((scraped_at, path))

        if not candidates:
            return None

        candidates.sort()

        return candidates[-1][1]

    # --------------------------------------------------

    def block_index(self, timestamps):
        """
        Block number 1-96 within the day. Block 1 is 00:00-00:15,
        so 11:15 is block 46 - the same numbering the reference
        features log and the grid operator's schedule use.
        """

        return (
            timestamps.hour * 4
            + timestamps.minute // (60 // (60 // self.interval_minutes))
            + 1
        )

    # --------------------------------------------------

    def build_time_grid(self, day):
        """
        Every 15-minute block of one calendar day, as tz-aware local
        timestamps plus the solar-geometry and clear-sky columns that
        depend only on the clock and the site.
        """

        day = pd.Timestamp(day).normalize()

        timestamps = pd.date_range(
            start=day,
            periods=24 * 60 // self.interval_minutes,
            freq=f"{self.interval_minutes}min",
            tz=self.timezone,
        )

        irradiance = self.clearsky.get_poa_irradiance(timestamps)
        position = self.clearsky.location.get_solarposition(timestamps)

        frame = pd.DataFrame({
            "timestamp": timestamps,
            "block": timestamps.hour * 4 + timestamps.minute // 15 + 1,
            "time": timestamps.strftime("%H:%M"),

            # --- calendar / clock features (kept from the reference log) ---
            "day_of_year": timestamps.dayofyear,
            "month": timestamps.month,
            "hour": timestamps.hour,
            "minute_of_day": timestamps.hour * 60 + timestamps.minute,

            # --- solar geometry ---
            "solar_elevation_deg": position["apparent_elevation"].to_numpy(),
            "solar_azimuth_deg": position["azimuth"].to_numpy(),

            # --- exact physics, from pvlib ---
            "clearsky_ghi_w_m2": irradiance["ghi"].to_numpy(),
            "clearsky_poa_w_m2": irradiance["poa_global"].to_numpy(),
        })

        frame["is_daylight"] = (frame["clearsky_ghi_w_m2"] > 20).astype(int)

        frame["clearsky_power_mw"] = (
            frame["clearsky_poa_w_m2"] / 1000
            * self.capacity_mw
            * self.clearsky.performance_ratio
        ).clip(lower=0)

        return frame

    # --------------------------------------------------

    def layer_series(self, payload, overlay):
        """
        One scraped overlay as a tidy (timestamp, value) frame in
        local time, dropping unreadable and stale readings.

        A layer the scraper marked `stale` is discarded outright:
        stale means every step returned the same number because
        Windy served one data file for the whole sweep, so using it
        would paint a flat line across the day and call it a
        forecast.
        """

        layer = payload.get("layers", {}).get(overlay)

        if layer is None:
            return None

        if layer.get("stale"):
            self.logger.warning(
                f"Windy layer '{overlay}' was marked stale by the scraper "
                "- excluded from the feature table"
            )
            return None

        rows = []

        for reading in layer.get("readings", []):

            if reading.get("value") is None:
                continue

            rows.append({
                "timestamp": pd.Timestamp(reading["timestamp"]).tz_convert(
                    self.timezone
                ),
                "value": float(reading["value"]),
                "direction": reading.get("direction"),
            })

        if not rows:
            return None

        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------

    def interpolate_solarpower(self, grid, series):
        """
        Spreads Windy's 3-hourly solar power across 15-minute blocks
        THROUGH clear-sky index space (see the module docstring), so
        the daily shape comes from pvlib rather than from a straight
        line drawn between two Windy steps.
        """

        # kt at each Windy step, using the clear-sky GHI at that exact
        # moment rather than the block's - the steps do not land on
        # block boundaries.
        step_clearsky = self.clearsky.get_clearsky_irradiance(
            series["timestamp"]
        )["ghi"].to_numpy()

        kt = np.divide(
            series["value"].to_numpy(),
            step_clearsky,
            out=np.full(len(series), np.nan),
            where=step_clearsky > 20,
        )

        known = ~np.isnan(kt)

        if not known.any():
            # Every Windy step landed at night, so no ratio is defined.
            # Nothing usable - say so rather than inventing a kt of 1.
            return None, None

        step_seconds = epoch_seconds(series["timestamp"])
        grid_seconds = epoch_seconds(grid["timestamp"])

        kt_block = np.interp(
            grid_seconds,
            step_seconds[known],
            kt[known],
        )

        kt_block = np.clip(kt_block, 0.0, 1.2)

        power = kt_block * grid["clearsky_ghi_w_m2"].to_numpy()

        return kt_block, power

    # --------------------------------------------------

    def interpolate_plain(self, grid, series):
        """
        Straight time interpolation, for the layers that carry no
        solar shape of their own (cloud %, rain mm, wind kt).
        """

        step_seconds = epoch_seconds(series["timestamp"])
        grid_seconds = epoch_seconds(grid["timestamp"])

        return np.interp(
            grid_seconds,
            step_seconds,
            series["value"].to_numpy(),
        )

    # --------------------------------------------------

    def attach_windy(self, grid, payload):
        """
        Adds every scraped layer to the block grid, plus the two
        honesty columns that say how real each row's Windy data is.
        """

        reference = None

        for overlay, column in _LAYER_COLUMNS.items():

            series = self.layer_series(payload, overlay)

            if series is None:
                grid[column] = np.nan
                continue

            if reference is None:
                reference = series

            if overlay == "solarpower":

                kt_block, power = self.interpolate_solarpower(grid, series)

                if power is None:
                    grid[column] = np.nan
                    grid["windy_kt"] = np.nan
                else:
                    grid[column] = power
                    grid["windy_kt"] = kt_block

            else:
                grid[column] = self.interpolate_plain(grid, series)

            if overlay == "wind":
                grid["windy_wind_from_deg"] = self.wind_direction(grid, series)

        # --- how far is each block from a REAL Windy reading? ---
        if reference is not None:

            step_seconds = epoch_seconds(reference["timestamp"])
            grid_seconds = epoch_seconds(grid["timestamp"])

            gap = np.min(
                np.abs(grid_seconds[:, None] - step_seconds[None, :]),
                axis=1,
            ) / 60.0

            grid["windy_gap_minutes"] = np.round(gap, 1)
            grid["windy_is_measured"] = (gap <= self.interval_minutes / 2).astype(int)

        else:
            grid["windy_gap_minutes"] = np.nan
            grid["windy_is_measured"] = 0

        return grid

    # --------------------------------------------------

    def wind_direction(self, grid, series):
        """
        Wind bearing per block, carried forward from the nearest
        Windy step rather than interpolated - averaging compass
        bearings numerically is the same circular-mean mistake that
        made the reference log's hue columns meaningless (the mean
        of 350 and 10 degrees is 0, not 180).
        """

        degrees = series["direction"].map(
            lambda d: _COMPASS_DEGREES.get(str(d).upper()) if d else None
        )

        if degrees.isna().all():
            return np.nan

        step_seconds = epoch_seconds(series["timestamp"])
        grid_seconds = epoch_seconds(grid["timestamp"])

        nearest = np.argmin(
            np.abs(grid_seconds[:, None] - step_seconds[None, :]),
            axis=1,
        )

        return degrees.to_numpy()[nearest]

    # --------------------------------------------------

    def attach_actuals(self, grid, meter, run_time):
        """
        Adds today's measured generation for blocks that have already
        happened, and marks which side of run_time each block is on.

        Blocks at or before run_time carry `actual_power_mw`; blocks
        after it are the ones being forecast and are left blank. The
        LLM reads both - what today has actually done so far is the
        strongest single clue about the rest of it.
        """

        grid["is_past"] = (grid["timestamp"] <= run_time).astype(int)

        grid["actual_power_mw"] = np.nan
        grid["actual_ghi_w_m2"] = np.nan
        grid["actual_kt"] = np.nan

        if meter is None or meter.empty:
            return grid

        meter = meter.copy()

        if meter["timestamp"].dt.tz is None:
            meter["timestamp"] = meter["timestamp"].dt.tz_localize(self.timezone)
        else:
            meter["timestamp"] = meter["timestamp"].dt.tz_convert(self.timezone)

        meter = meter[meter["timestamp"] <= run_time]

        columns = ["timestamp", "active_power_kw"]

        if "ghi_w_m2" in meter.columns:
            columns.append("ghi_w_m2")

        merged = grid.merge(
            meter[columns], on="timestamp", how="left", suffixes=("", "_meter")
        )

        merged["actual_power_mw"] = merged["active_power_kw"] / 1000.0

        if "ghi_w_m2" in merged.columns:
            merged["actual_ghi_w_m2"] = merged["ghi_w_m2"]
            merged["actual_kt"] = np.clip(
                np.divide(
                    merged["ghi_w_m2"].to_numpy(dtype=float),
                    merged["clearsky_ghi_w_m2"].to_numpy(dtype=float),
                    out=np.full(len(merged), np.nan),
                    where=merged["clearsky_ghi_w_m2"].to_numpy() > 20,
                ),
                0, 1.2,
            )

        return merged.drop(columns=["active_power_kw", "ghi_w_m2"], errors="ignore")

    # --------------------------------------------------

    def build(self, run_time=None, meter=None, values_path=None):
        """
        Builds the full per-block feature table for run_time's day.

        Returns (frame, values_path). `values_path` is None when no
        scraped file was available - the table is still returned, with
        the Windy columns blank, so a caller can see exactly what was
        and was not known.
        """

        run_time = pd.Timestamp(run_time or pd.Timestamp.now())

        if run_time.tz is None:
            run_time = run_time.tz_localize(self.timezone)

        run_time = run_time.floor(f"{self.interval_minutes}min")

        grid = self.build_time_grid(run_time.date())

        if values_path is None:
            values_path = self.latest_values_file(run_time)

        if values_path is None:
            self.logger.warning(
                "No scraped Windy values found - the feature table will have "
                "empty Windy columns. Run: python -m modules.capture.windy_scraper"
            )
            for column in _LAYER_COLUMNS.values():
                grid[column] = np.nan
            grid["windy_kt"] = np.nan
            grid["windy_gap_minutes"] = np.nan
            grid["windy_is_measured"] = 0

        else:
            payload = json.loads(Path(values_path).read_text(encoding="utf-8"))
            grid = self.attach_windy(grid, payload)

        grid = self.attach_actuals(grid, meter, run_time)

        grid.insert(0, "plant", self.plant_tag)
        grid["run_time"] = run_time

        return grid, values_path

    # --------------------------------------------------

    def save(self, frame, run_time):

        self.output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.plant_tag}_{pd.Timestamp(run_time).strftime('%Y-%m-%d_%H-%M')}"

        path = self.output_dir / f"{stem}_features.csv"

        frame.to_csv(path, index=False)

        self.logger.info(f"Feature table saved: {path}")

        return path


# --------------------------------------------------

def load_meter_history():
    """
    Every daily raw meter CSV, preprocessed and concatenated - the
    same per-day treatment the existing orchestrator uses (each day
    is resampled on its own so nothing interpolates across the
    overnight gap).

    Deliberately does NOT go through Orchestrator: constructing that
    builds HybridPredictor, which loads the Chronos weights (~450 MB
    and several seconds) purely to read some CSVs. The new pipeline
    has no Chronos in it, so it must not pay for one.
    """

    from modules.preprocessing.preprocess import DataPreprocessor

    folder = Path(settings["paths"]["historical_data"])

    files = sorted(folder.glob("*.csv"))

    if not files:
        return None

    preprocessor = DataPreprocessor()

    days = [preprocessor.preprocess(file_path=path) for path in files]

    frame = pd.concat(days, ignore_index=True)

    return frame.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------

def main():
    """
    `python -m modules.preprocessing.windy_features` - builds and
    prints today's feature table so it can be inspected directly.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build the per-block feature table")
    parser.add_argument("--run-time", default=None, help="e.g. 2026-08-09 11:15")
    parser.add_argument("--no-meter", action="store_true",
                        help="skip loading meter history")

    args = parser.parse_args()

    builder = WindyFeatureBuilder()

    meter = None

    if not args.no_meter:
        try:
            meter = load_meter_history()
        except Exception as error:
            print(f"(meter history unavailable: {error})")

    frame, values_path = builder.build(run_time=args.run_time, meter=meter)

    print(f"windy values: {values_path}")
    print(f"rows        : {len(frame)}")

    daylight = frame[frame["is_daylight"] == 1]

    columns = [
        "block", "time", "solar_elevation_deg", "clearsky_power_mw",
        "windy_kt", "windy_solarpower_w_m2", "windy_clouds_pct",
        "windy_lclouds_pct", "windy_rain_mm",
        "windy_gap_minutes", "windy_is_measured", "actual_power_mw",
    ]

    columns = [c for c in columns if c in daylight.columns]

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(daylight[columns].to_string(index=False, float_format="%.2f"))

    builder.save(frame, frame["run_time"].iloc[0])

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
