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
from modules.forecasting.block_bias_correction import BlockBiasCorrector
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
    blocks.

    Returns three things:
      * the FINAL schedule - every block carrying the run that last
        wrote it (what would actually have been published),
      * the run log,
      * every INDIVIDUAL run's own complete schedule from its block T
        to the end of the day. The workflow document asks for a
        schedule to be generated at each scheduling time, so those
        seven schedules are deliverables in their own right, not just
        intermediate state - they show the forecast being revised as
        the day reveals itself.
    """

    predictor = HybridPredictor()
    fusion = FeatureFusion()
    corrector = CaseBasedCorrector()

    if corrector.available:
        corrector.load()

    # Learns only from day schedules BEFORE this day, so rebuilding an
    # old day never sees its own outcome (or any later day's).
    block_bias = BlockBiasCorrector().load(as_of=day)

    schedule = {}          # timestamp -> dict(value, source run, had vision)
    run_log = []
    per_run = []           # every run's own full schedule, block T -> end of day

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

        # Same corrections the live pipeline applies, in the same order.
        if corrector.available:
            forecast = corrector.apply(forecast, run_time, signals["kt_now"])

        if block_bias.available:
            forecast = block_bias.apply(forecast)

        written = 0
        for timestamp, value in zip(forecast["timestamp"], forecast["final_forecast_kw"]):
            if timestamp > run_time:

                # This run's own schedule for every remaining block -
                # kept before the merge overwrites anything.
                per_run.append({
                    "scheduling_time": run_str,
                    "timestamp": timestamp,
                    "scheduled_mw": round(float(value) / 1000, 4),
                })

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
            "block_bias_days": len(block_bias.days_used) if block_bias.available else 0,
        })

    rows = [
        {"timestamp": ts, **info}
        for ts, info in sorted(schedule.items())
    ]

    return pd.DataFrame(rows), pd.DataFrame(run_log), pd.DataFrame(per_run)


def main():

    day_str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DAY
    day = pd.Timestamp(day_str).date()

    provider = settings["vision"].get("provider", "gemini")

    data = load_data()

    schedule, run_log, per_run = build_schedule(data, day, provider)

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

    # Reported in MW throughout - the plant is rated in MW (5.1 MW) and
    # schedules are submitted in MW, so kW only invites conversion slips.
    report["scheduled_mw"] = (report["scheduled_kw"] / 1000).round(4)
    report["actual_mw"] = (report["actual_kw"] / 1000).round(4)
    report["error_mw"] = (report["scheduled_mw"] - report["actual_mw"]).round(4)
    report["capacity_utilisation_pct"] = (
        report["scheduled_mw"] / (CAPACITY_KW / 1000) * 100
    ).round(2)

    report = report.drop(columns=["scheduled_kw", "actual_kw"])

    # ---- scoring: real measured blocks only ----
    scored = report[report["actual_is_real"] & report["actual_mw"].notna()]

    out_dir = Path("outputs/schedules")
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = out_dir / f"day_schedule_{day}.csv"
    report.to_csv(schedule_path, index=False)

    log_path = out_dir / f"day_schedule_{day}_run_log.csv"
    run_log.to_csv(log_path, index=False)

    # ---- each scheduling time's OWN schedule, scored on its own ----
    per_run = per_run.merge(actual, on="timestamp", how="left")
    per_run = per_run.rename(columns={"active_power_kw": "actual_kw"})

    if "is_real_measurement" in per_run.columns:
        per_run["actual_is_real"] = per_run["is_real_measurement"].fillna(False)
        per_run = per_run.drop(columns=["is_real_measurement"])
    else:
        per_run["actual_is_real"] = per_run["actual_kw"].notna()

    per_run.insert(1, "block", per_run["timestamp"].map(block_number))
    per_run.insert(2, "block_time", per_run["timestamp"].dt.strftime("%H:%M"))
    per_run["actual_mw"] = (per_run["actual_kw"] / 1000).round(4)
    per_run["error_mw"] = (per_run["scheduled_mw"] - per_run["actual_mw"]).round(4)
    per_run = per_run.drop(columns=["actual_kw"])

    per_run_path = out_dir / f"day_schedule_{day}_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    capacity_mw_ = CAPACITY_KW / 1000

    per_run_scores = []
    for run_str, group in per_run.groupby("scheduling_time", sort=False):
        g = group[group["actual_is_real"] & group["actual_mw"].notna()]
        if g.empty:
            continue
        per_run_scores.append({
            "scheduling_time": run_str,
            "blocks_scheduled": len(group),
            "blocks_scored": len(g),
            "deviation_pct": round(metrics.average_percentage_deviation(
                g["scheduled_mw"], g["actual_mw"], capacity_mw_), 2),
            "mae_mw": round(metrics.mean_absolute_error(
                g["scheduled_mw"], g["actual_mw"]), 4),
            "rmse_mw": round(metrics.root_mean_squared_error(
                g["scheduled_mw"], g["actual_mw"]), 4),
            "scheduled_energy_mwh": round(group["scheduled_mw"].sum() * 0.25, 3),
        })

    per_run_scores = pd.DataFrame(per_run_scores)
    scores_path = out_dir / f"day_schedule_{day}_per_run_scores.csv"
    per_run_scores.to_csv(scores_path, index=False)

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

    print("EACH SCHEDULING TIME'S OWN SCHEDULE, SCORED SEPARATELY")
    print("-" * 84)
    print(per_run_scores.to_string(index=False))
    print()

    print("SCHEDULE (MW)")
    print("-" * 84)
    display = report[[
        "block", "block_time", "scheduled_mw", "actual_mw", "error_mw", "scheduled_at"
    ]]
    print(display.to_string(index=False))
    print()

    if scored.empty:
        print("No real measured blocks available to score against yet.")
        return

    capacity_mw = CAPACITY_KW / 1000

    deviation = metrics.average_percentage_deviation(
        scored["scheduled_mw"], scored["actual_mw"], capacity_mw
    )
    mae = metrics.mean_absolute_error(scored["scheduled_mw"], scored["actual_mw"])
    rmse = metrics.root_mean_squared_error(scored["scheduled_mw"], scored["actual_mw"])

    scheduled_energy = scored["scheduled_mw"].sum() * 0.25
    actual_energy = scored["actual_mw"].sum() * 0.25

    print("=" * 84)
    print(f"ACCURACY vs ACTUAL METER DATA   (plant capacity {capacity_mw} MW)")
    print("=" * 84)
    print(f"  blocks scored (real measurements only) : {len(scored)}")
    print(f"  average percentage deviation           : {deviation:.2f}%")
    print(f"  mean absolute error                    : {mae:.4f} MW")
    print(f"  root mean squared error                : {rmse:.4f} MW")
    print(f"  scheduled energy                       : {scheduled_energy:.3f} MWh")
    print(f"  actual energy                          : {actual_energy:.3f} MWh")
    print(f"  energy difference                      : "
          f"{scheduled_energy - actual_energy:+.3f} MWh "
          f"({(scheduled_energy/actual_energy - 1) * 100:+.1f}%)")
    print(f"  average scheduled / actual             : "
          f"{scored['scheduled_mw'].mean():.3f} MW / "
          f"{scored['actual_mw'].mean():.3f} MW")
    print(f"  peak scheduled / actual                : "
          f"{scored['scheduled_mw'].max():.3f} MW / "
          f"{scored['actual_mw'].max():.3f} MW")

    # Machine-readable summary beside the schedule, so the report
    # builder never has to re-derive these numbers.
    summary = pd.DataFrame([
        {"metric": "Plant", "value": PLANT_NAME},
        {"metric": "Plant capacity (MW)", "value": capacity_mw},
        {"metric": "Schedule date", "value": str(day)},
        {"metric": "Scheduling times", "value": ", ".join(RUN_TIMES)},
        {"metric": "Windy clips available", "value":
            f"{int((run_log['vision_signal'] == 'yes').sum())} of {len(RUN_TIMES)}"},
        {"metric": "Blocks scheduled", "value": len(report)},
        {"metric": "Blocks scored (real measurements only)", "value": len(scored)},
        {"metric": "Average percentage deviation (%)", "value": round(deviation, 2)},
        {"metric": "Mean absolute error (MW)", "value": round(mae, 4)},
        {"metric": "Root mean squared error (MW)", "value": round(rmse, 4)},
        {"metric": "Scheduled energy (MWh)", "value": round(scheduled_energy, 3)},
        {"metric": "Actual energy (MWh)", "value": round(actual_energy, 3)},
        {"metric": "Energy difference (MWh)", "value":
            round(scheduled_energy - actual_energy, 3)},
        {"metric": "Average scheduled (MW)", "value":
            round(scored["scheduled_mw"].mean(), 3)},
        {"metric": "Average actual (MW)", "value":
            round(scored["actual_mw"].mean(), 3)},
        {"metric": "Peak scheduled (MW)", "value":
            round(scored["scheduled_mw"].max(), 3)},
        {"metric": "Peak actual (MW)", "value":
            round(scored["actual_mw"].max(), 3)},
    ])

    summary_path = out_dir / f"day_schedule_{day}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Saved schedule     : {schedule_path}")
    print(f"Saved run log      : {log_path}")
    print(f"Saved summary      : {summary_path}")
    print(f"Saved per-run      : {per_run_path}")
    print(f"Saved per-run score: {scores_path}")


if __name__ == "__main__":
    main()
