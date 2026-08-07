"""
=========================================================
Solar Forecasting Project
Clear-Sky Model Smoke Test
=========================================================
Computes the clear-sky curve for one processed day and
sanity-checks it against the actual meter GHI/POA readings.
=========================================================
"""

from pathlib import Path

import pandas as pd

from modules.forecasting.clearsky import ClearSkyModel
from utils.file_manager import processed_data_path


def main():

    processed_file = processed_data_path()

    dataframe = pd.read_csv(processed_file, parse_dates=["timestamp"])

    # Pick one day to inspect
    day = dataframe[
        dataframe["timestamp"].dt.date.astype(str) == "2026-07-10"
    ].reset_index(drop=True)

    print("=" * 60)
    print("Clear-Sky Model Test - 2026-07-10")
    print("=" * 60)

    model = ClearSkyModel()

    # Expected clear-sky generation curve (no historical data needed)
    # reset_index so this aligns positionally with `day`'s RangeIndex
    # instead of pvlib's tz-aware DatetimeIndex
    expected = model.estimate_clearsky_generation(
        day["timestamp"]
    ).reset_index(drop=True)

    print("\nExpected clear-sky curve (first 5 rows)")
    print(expected[["ghi", "poa_global", "expected_power_kw"]].head())

    # Clear-sky index vs actual GHI
    with_kt = model.compute_clear_sky_index(
        day,
        ghi_column="ghi_w_m2",
        timestamp_column="timestamp"
    )

    comparison = pd.DataFrame({
        "timestamp": day["timestamp"],
        "actual_ghi": day["ghi_w_m2"],
        "clear_sky_ghi": with_kt["clear_sky_ghi"],
        "clear_sky_index": with_kt["clear_sky_index"],
        "actual_power_kw": day["active_power_kw"],
        "expected_clearsky_power_kw": expected["expected_power_kw"]
    })

    print("\nActual vs Clear-Sky (midday sample)")
    print(comparison.iloc[20:28])

    print("\nMean clear-sky index (daylight hours):")
    print(with_kt["clear_sky_index"].mean())


if __name__ == "__main__":
    main()
