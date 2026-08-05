"""
=========================================================
Solar Forecasting Project
Satellite GHI vs Current Weather Source
=========================================================
Mentor's concern: Open-Meteo/ECMWF is a numerical weather
model, ~25-28 km grid - too coarse for a single plant's
local cloud. This tests a real fix candidate before
touching the pipeline: Open-Meteo's SATELLITE Radiation API
(https://satellite-api.open-meteo.com) - actual geostationary
satellite-measured GHI (Himawari over India), ~2.5-5 km,
10-30 min refresh. Free, no key, same provider family as
what we already use - zero new signup friction.

This is a COMPARISON ONLY. Nothing is wired into the blend.
Per the mentor brief: build the test, show the result, THEN
decide.

Compares, per day with real meter data:
  1. MEASURED plant GHI (ground truth)
  2. Satellite GHI (this candidate)
  3. Our current forecast weather_kt path, for reference

Run:  python -m tests.test_satellite_ghi_experiment
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.config import settings
from modules.preprocessing.preprocess import DataPreprocessor

LAT = settings["plant"]["latitude"]
LON = settings["plant"]["longitude"]
TZ = settings["plant"]["timezone"]

SAT_URL = "https://satellite-api.open-meteo.com/v1/archive"


def load_meter_data():

    pre = DataPreprocessor()

    frames = [
        pre.preprocess(
            file_path=path, required_columns=["TimeStamp"],
            timestamp_column="TimeStamp",
        )
        for path in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp").reset_index(drop=True)
    )


def fetch_satellite_ghi(start_date, end_date):
    """One archive call covering the whole date range - cheap, no key."""

    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": "shortwave_radiation",
        "timezone": TZ,
    }

    response = requests.get(SAT_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()["hourly"]

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "sat_ghi": data["shortwave_radiation"],
    })


def main():

    print("=" * 70)
    print("SATELLITE GHI vs MEASURED - candidate check")
    print("=" * 70)

    meter = load_meter_data()
    real = meter[meter["is_real_measurement"].fillna(False)][
        ["timestamp", "ghi_w_m2"]
    ].dropna()

    if real.empty:
        print("No real measured GHI found - nothing to compare.")
        return

    start = real["timestamp"].min().strftime("%Y-%m-%d")
    end = real["timestamp"].max().strftime("%Y-%m-%d")
    print(f"date range: {start} .. {end}  ({real['timestamp'].dt.date.nunique()} days)")

    print("Fetching satellite GHI archive (one call, free, no key)...")
    sat = fetch_satellite_ghi(start, end)

    # Satellite data is hourly; meter is 15-min. Compare at the hourly
    # timestamps satellite actually has, nearest meter reading within 10 min.
    merged = pd.merge_asof(
        sat.sort_values("timestamp"),
        real.sort_values("timestamp"),
        on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("10min"),
    ).dropna()

    if merged.empty:
        print("No overlapping timestamps after alignment - check timezone.")
        return

    # Daylight only - both should read ~0 at night, and night ratios blow up.
    day = merged[merged["ghi_w_m2"] > 20]

    error = day["sat_ghi"] - day["ghi_w_m2"]
    pct_error = (error / day["ghi_w_m2"].clip(lower=20)) * 100

    print()
    print(f"daylight points compared: {len(day)}")
    print(f"mean absolute error       : {error.abs().mean():.1f} W/m2")
    print(f"mean signed error         : {error.mean():+.1f} W/m2 "
          f"({'over' if error.mean() > 0 else 'under'}-estimates)")
    print(f"mean absolute %% error     : {pct_error.abs().mean():.1f}%%")
    print(f"correlation (sat vs real) : {day['sat_ghi'].corr(day['ghi_w_m2']):.3f}")

    out = Path("outputs/reports/satellite_ghi_experiment.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    day.to_csv(out, index=False)
    print()
    print(f"Saved per-point comparison: {out}")
    print()
    print("This is measurement accuracy only (satellite vs meter). Forecast")
    print("skill (predicting hours ahead) is a separate, follow-up question -")
    print("this run only answers 'is satellite data even worth pursuing'.")


if __name__ == "__main__":
    main()
