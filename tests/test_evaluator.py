"""
=========================================================
Solar Forecasting Project
Evaluator Smoke Test
=========================================================
Runs the same simulated 12:45 forecast on 2026-07-10 used
in test_predictor.py, then scores it against actual
generation and against Enercast side by side.
=========================================================
"""

from pathlib import Path

import pandas as pd

from modules.forecasting.predictor import HybridPredictor
from modules.evaluation.evaluator import Evaluator
from utils.file_manager import processed_data_path


def main():

    processed_file = processed_data_path()

    dataframe = pd.read_csv(processed_file, parse_dates=["timestamp"])

    run_time = pd.Timestamp("2026-07-10 12:45:00")

    print("=" * 60)
    print(f"Evaluation - simulated run_time {run_time}")
    print("=" * 60)

    predictor = HybridPredictor()

    forecast = predictor.predict(dataframe, run_time)

    evaluator = Evaluator()

    comparison, report = evaluator.evaluate(forecast, dataframe)

    print("\nComparison table")
    print(comparison[[
        "timestamp",
        "final_forecast_kw",
        "enercast_kw",
        "active_power_kw"
    ]])

    print("\n--- Our Forecast ---")
    for key, value in report["our_forecast"].items():
        print(f"{key:20s}: {value:.2f}")

    if "enercast" in report:
        print("\n--- Enercast ---")
        for key, value in report["enercast"].items():
            print(f"{key:20s}: {value:.2f}")


if __name__ == "__main__":
    main()
