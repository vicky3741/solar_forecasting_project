"""
=========================================================
Solar Forecasting Project
Orchestrator Smoke Test
=========================================================
Runs one full forecast cycle at a specific historical
run_time (2026-07-10 15:45, the last official run of that
day, so this also exercises the end-of-day validation
step) and checks the output files were written correctly.
=========================================================
"""

from pathlib import Path

import pandas as pd

from modules.orchestrator.pipeline import Orchestrator
from utils import file_manager


def main():

    orchestrator = Orchestrator()

    run_time = pd.Timestamp("2026-07-10 15:45:00")

    print("=" * 60)
    print(f"Orchestrator run - {run_time}")
    print("=" * 60)

    forecast = orchestrator.run(run_time)

    print("\nForecast (first 5 rows)")
    print(forecast.head())

    forecast_path = Path("outputs/forecasts/2026-07-10_15-45.csv")
    archive_path = Path("outputs/forecasts/archive.csv")
    schedule_path = Path("outputs/schedules/current_final_schedule.csv")
    validation_path = Path("outputs/reports/2026-07-10_end_of_day_validation.json")

    print(f"\nForecast file exists   : {forecast_path.exists()}")
    print(f"Archive file exists    : {archive_path.exists()}")
    print(f"Schedule file exists   : {schedule_path.exists()}")
    print(f"Validation file exists : {validation_path.exists()}")

    schedule = file_manager.load_dataframe(schedule_path, parse_dates=["timestamp"])
    print(f"\nCurrent schedule rows  : {len(schedule)}")
    print(schedule.tail(10))

    validation = file_manager.load_json(validation_path)
    print("\nEnd-of-day validation report:")
    print(validation)


if __name__ == "__main__":
    main()
