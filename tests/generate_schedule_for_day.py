"""
=========================================================
Solar Forecasting Project
Day Schedule Generator (mentor's evaluation workflow)
=========================================================
Produces the deliverable described in "Schedule Generation
Workflow for Model Evaluation": ONE complete reconstructed
day schedule, built exactly as it would have been published
in real time, then compared against actual meter data.

For each of the 7 scheduling times the model is given only
what existed at that instant:
  * the Windy clip STORED for that time (local capture or
    the S3 feed; none = that run honestly runs no-vision),
  * meter data up to block T only - no lookahead,
  * historical generation and the weather forecast that was
    available then.
It then schedules from block T to the end of the day, and
only the blocks still in the FUTURE are rewritten - blocks
already passed keep the value an earlier run gave them.

The result is a block-by-block schedule showing which run
is responsible for every block, scored against real measured
meter readings only (never gap-filled ones).

Run:  python -m tests.generate_schedule_for_day [YYYY-MM-DD]
=========================================================
"""

import sys
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.forecasting.case_based_correction import CaseBasedCorrector
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.preprocessing.preprocess import DataPreprocessor
from modules.evaluation import metrics
from tests.test_schedule_reconstruction import find_video_for, vision_features_for


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]
PLANT_NAME = settings["plant"]["name"]

DEFAULT_DAY = "2026-07-25"


def block_number(timestamp):
    """
    Indian scheduling block number: the day is split into 96
    fifteen-minute blocks, block 1 being 00:00-00:15.
    """

    return timestamp.hour * 4 + timestamp.minute // 15 + 1


def load_data():

    pre = DataPreprocessor()

    frames = [
        pre.preprocess(
            file_path=path,
            required_columns=["TimeStamp"],
            timestamp_column="TimeStamp",
        )
        for path in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]

    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def build_schedule(data, day, provider):
    """
    Walks the 7 scheduling times in order, rewriting only future
    blocks. Returns the final schedule with, for every block, the
    run that produced it and whether that run had a Windy clip.
    """

    predictor = HybridPredictor()
    fusion = FeatureFusion()
    corrector = CaseBasedCorrector()

    if corrector.available:
        corrector.load()

    schedule = {}          # timestamp -> dict(value, source run, had vision)
    run_log = []

    for run_str in RUN_TIMES:

        hour, minute = map(int, run_str.split(":"))
        run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

        video = find_video_for(day, run_str)
        features = vision_features_for(provider, video) if provider else None

        # Only data up to block T - compute_signals filters on run_time.
        signals = predictor.compute_signals(data, run_time)

        adjustment = (
            fusion.trend_adjustment_profile(features, signals["horizon_minutes"])
            if features else 0.0
        )

        forecast = predictor.blend_signals(signals, vision_adjustment=adjustment)

        # Same correction the live pipeline applies.
        if corrector.available:
            forecast = corrector.apply(forecast, run_time, signals["kt_now"])

        written = 0
        for timestamp, value in zip(forecast["timestamp"], forecast["final_forecast_kw"]):
            if timestamp > run_time:
                schedule[timestamp] = {
                    "scheduled_kw": float(value),
                    "scheduled_at": run_str,
                    "windy_video": video.name if video is not None else "(none)",
                    "vision_used": features is not None,
                }
                written += 1

        run_log.append({
            "scheduling_time": run_str,
            "windy_video_used": video.name if video is not None else "(none available)",
            "vision_signal": "yes" if features is not None else "no",
            "blocks_written": written,
            "weather_bias_factor": round(signals.get("weather_bias_factor", 1.0), 3),
        })

    rows = [
        {"timestamp": ts, **info}
        for ts, info in sorted(schedule.items())
    ]

    return pd.DataFrame(rows), pd.DataFrame(run_log)


def main():

    day_str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DAY
    day = pd.Timestamp(day_str).date()

    provider = settings["vision"].get("provider", "gemini")

    data = load_data()

    schedule, run_log = build_schedule(data, day, provider)

    if schedule.empty:
        print(f"No schedule could be built for {day}.")
        return

    # Actual generation - real measured blocks only.
    actual = data[data["timestamp"].dt.date == day]
    columns = ["timestamp", "active_power_kw"]
    if "is_real_measurement" in actual.columns:
        columns.append("is_real_measurement")
    actual = actual[columns]

    report = schedule.merge(actual, on="timestamp", how="left")

    report.insert(0, "block", report["timestamp"].map(block_number))
    report.insert(1, "block_time", report["timestamp"].dt.strftime("%H:%M"))

    report = report.rename(columns={"active_power_kw": "actual_kw"})

    if "is_real_measurement" in report.columns:
        report["actual_is_real"] = report["is_real_measurement"].fillna(False)
        report = report.drop(columns=["is_real_measurement"])
    else:
        report["actual_is_real"] = report["actual_kw"].notna()

    report["error_kw"] = (report["scheduled_kw"] - report["actual_kw"]).round(1)
    report["scheduled_kw"] = report["scheduled_kw"].round(1)
    report["actual_kw"] = report["actual_kw"].round(1)

    # ---- scoring: real measured blocks only ----
    scored = report[report["actual_is_real"] & report["actual_kw"].notna()]

    out_dir = Path("outputs/schedules")
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = out_dir / f"day_schedule_{day}.csv"
    report.to_csv(schedule_path, index=False)

    log_path = out_dir / f"day_schedule_{day}_run_log.csv"
    run_log.to_csv(log_path, index=False)

    # ---- printed summary ----
    print("=" * 84)
    print(f"RECONSTRUCTED DAY SCHEDULE - {PLANT_NAME} - {day}")
    print("=" * 84)
    print("Built per 'Schedule Generation Workflow for Model Evaluation':")
    print("each scheduling time uses only the Windy clip stored at that time and")
    print("meter data up to block T; only future blocks are rewritten.")
    print()

    print("HOW THE DAY WAS BUILT")
    print("-" * 84)
    print(run_log.to_string(index=False))
    print()

    print("SCHEDULE")
    print("-" * 84)
    display = report[[
        "block", "block_time", "scheduled_kw", "actual_kw", "error_kw", "scheduled_at"
    ]]
    print(display.to_string(index=False))
    print()

    if scored.empty:
        print("No real measured blocks available to score against yet.")
    else:
        deviation = metrics.average_percentage_deviation(
            scored["scheduled_kw"], scored["actual_kw"], CAPACITY_KW
        )
        mae = metrics.mean_absolute_error(scored["scheduled_kw"], scored["actual_kw"])
        rmse = metrics.root_mean_squared_error(scored["scheduled_kw"], scored["actual_kw"])

        print("=" * 84)
        print("ACCURACY vs ACTUAL METER DATA")
        print("=" * 84)
        print(f"  blocks scored (real measurements only) : {len(scored)}")
        print(f"  average percentage deviation           : {deviation:.2f}%")
        print(f"  mean absolute error                    : {mae:.1f} kW")
        print(f"  root mean squared error                : {rmse:.1f} kW")
        print(f"  scheduled energy                       : "
              f"{scored['scheduled_kw'].sum() * 0.25 / 1000:.3f} MWh")
        print(f"  actual energy                          : "
              f"{scored['actual_kw'].sum() * 0.25 / 1000:.3f} MWh")
        print(f"  average scheduled / actual             : "
              f"{scored['scheduled_kw'].mean()/1000:.3f} MW / "
              f"{scored['actual_kw'].mean()/1000:.3f} MW")

    print()
    print(f"Saved schedule : {schedule_path}")
    print(f"Saved run log  : {log_path}")


if __name__ == "__main__":
    main()
