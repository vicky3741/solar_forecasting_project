"""
=========================================================
Solar Forecasting Project
Hybrid Predictor Smoke Test
=========================================================
Simulates a 12:45 forecast run on 2026-07-10 using only
data available up to that point in time (no lookahead),
then compares the forecast against what actually happened
that day, since it is historical data.
=========================================================
"""

from pathlib import Path

import pandas as pd

from modules.forecasting.predictor import HybridPredictor
from utils.file_manager import processed_data_path


def main():

    processed_file = processed_data_path()

    dataframe = pd.read_csv(processed_file, parse_dates=["timestamp"])

    run_time = pd.Timestamp("2026-07-10 12:45:00")

    print("=" * 60)
    print(f"Hybrid Forecast Run - simulated run_time {run_time}")
    print("=" * 60)

    predictor = HybridPredictor()

    forecast = predictor.predict(dataframe, run_time)

    print(f"\nChronos weight used : {forecast.attrs['chronos_weight_used']:.3f}")
    print(f"Clear-sky index now : {forecast.attrs['clear_sky_index_now']:.3f}")

    print("\nForecast (first 10 rows)")
    print(forecast.head(10))

    # Compare against what actually happened (historical data)
    actual = dataframe[
        (dataframe["timestamp"] > run_time)
        & (dataframe["timestamp"].dt.date == run_time.date())
    ][["timestamp", "active_power_kw"]]

    comparison = forecast.merge(actual, on="timestamp", how="left")

    print("\nForecast vs Actual")
    print(comparison[[
        "timestamp",
        "chronos_forecast_kw",
        "physics_forecast_kw",
        "final_forecast_kw",
        "active_power_kw"
    ]])

    mae = (comparison["final_forecast_kw"] - comparison["active_power_kw"]).abs().mean()

    print(f"\nMean Absolute Error (final forecast vs actual): {mae:.2f} kW")


if __name__ == "__main__":
    main()
