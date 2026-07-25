"""
=========================================================
Solar Forecasting Project
Schedule Reconstruction (mentor's evaluation workflow)
=========================================================
Implements the workflow in "Schedule Generation Workflow
for Model Evaluation" exactly:

  * At each of the 7 scheduling times (06:45 ... 15:45) the
    model builds a schedule from block T to the END OF DAY,
    using ONLY what existed at that instant - the Windy
    video captured at that time and meter data up to block
    T (no lookahead).
  * Moving to the next scheduling time, only the FUTURE
    blocks are rewritten; blocks that have already passed
    keep the value they were given earlier.
  * The result is ONE reconstructed schedule for the whole
    day - what the system would really have published - and
    that is what gets compared against actual meter data.

Because each block keeps the value from the LAST run before
it, this is not the same as scoring each run over the whole
remaining day (which double-counts) nor scoring runs in
isolation. It is the real operational schedule.

To answer the mentor's question "what is the output if we
skip the Windy videos and use the rest of the parameters as
they are?", the same reconstruction is run three ways:
no-vision, Gemini, ChatGPT - side by side.

Run:  python -m tests.test_schedule_reconstruction
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.vision.vision_module import VisionModule
from modules.evaluation import metrics


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]

VIDEO_DIR = Path(settings["windy_capture"]["video_dir"])
CACHE_ROOT = Path("outputs/llm_compare")

TRIM_START_FRACTION = 0.15
TRIM_END_FRACTION = 0.70
TARGET_FRAMES = 6


def find_video_for(day, run_time_str):
    """
    The Windy clip captured AT this scheduling time (within a few
    minutes). Returns None when that capture is missing, so the run
    honestly falls back to no-vision rather than borrowing another
    time's video.
    """

    hour, minute = map(int, run_time_str.split(":"))
    target = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

    best = None
    for path in VIDEO_DIR.glob("*.webm"):
        captured = VisionModule.parse_video_time(path.name)
        if captured is None or captured.date() != day:
            continue
        gap = abs((pd.Timestamp(captured) - target).total_seconds())
        if gap <= 300 and (best is None or gap < best[0]):   # within 5 min
            best = (gap, path)

    return best[1] if best else None


def vision_features_for(provider, video_path):
    """Cached vision reading for one provider, or None."""

    if video_path is None or provider is None:
        return None

    settings["vision"]["provider"] = provider

    try:
        vision = VisionModule()
        out = vision.analyze_video(
            str(video_path), str(CACHE_ROOT / provider),
            target_frames=TARGET_FRAMES,
            start_fraction=TRIM_START_FRACTION,
            end_fraction=TRIM_END_FRACTION,
        )
        features = out["weather_features"]
        return None if "_error" in features else features
    except Exception:
        return None


def reconstruct_day(predictor, fusion, processed, day, provider):
    """
    Walks the 7 scheduling times in order, each time rewriting only
    the blocks that are still in the future. Returns the final
    schedule for the day plus a per-run record of which blocks each
    run contributed.
    """

    schedule = {}          # timestamp -> forecast kW (last writer wins)
    contributions = []

    for run_time_str in RUN_TIMES:

        hour, minute = map(int, run_time_str.split(":"))
        run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

        video = find_video_for(day, run_time_str)
        features = vision_features_for(provider, video) if provider else None

        # Only data up to block T - compute_signals filters on run_time,
        # so there is no lookahead here.
        signals = predictor.compute_signals(processed, run_time)

        adjustment = (
            fusion.trend_adjustment_profile(features, signals["horizon_minutes"])
            if features else 0.0
        )

        forecast = predictor.blend_signals(signals, vision_adjustment=adjustment)

        # Rewrite future blocks only; earlier blocks keep their values.
        written = 0
        for timestamp, value in zip(forecast["timestamp"], forecast["final_forecast_kw"]):
            if timestamp > run_time:
                schedule[timestamp] = value
                written += 1

        contributions.append({
            "run_time": run_time_str,
            "video": video.name if video else "(none)",
            "had_vision": features is not None,
            "blocks_written": written,
        })

    final = pd.DataFrame(
        {"timestamp": list(schedule.keys()), "final_forecast_kw": list(schedule.values())}
    ).sort_values("timestamp").reset_index(drop=True)

    return final, pd.DataFrame(contributions)


def score_schedule(final, actual):
    """Scores the reconstructed day against REAL measured blocks only."""

    merged = final.merge(actual, on="timestamp", how="inner")

    if "is_real_measurement" in merged.columns:
        merged = merged[merged["is_real_measurement"].fillna(False)]

    if merged.empty:
        return None

    forecast_kw = merged["final_forecast_kw"]
    actual_kw = merged["active_power_kw"]

    return {
        "blocks": len(merged),
        "deviation_pct": metrics.average_percentage_deviation(
            forecast_kw, actual_kw, CAPACITY_KW),
        "mae_kw": metrics.mean_absolute_error(forecast_kw, actual_kw),
        "rmse_kw": metrics.root_mean_squared_error(forecast_kw, actual_kw),
        "scheduled_mwh": forecast_kw.sum() * 0.25 / 1000,
        "actual_mwh": actual_kw.sum() * 0.25 / 1000,
        "avg_scheduled_mw": forecast_kw.mean() / 1000,
        "avg_actual_mw": actual_kw.mean() / 1000,
    }


def main():

    processed = pd.read_csv(
        "data/processed/processed_data.csv", parse_dates=["timestamp"]
    )

    actual_columns = ["timestamp", "active_power_kw"]
    if "is_real_measurement" in processed.columns:
        actual_columns.append("is_real_measurement")
    actual = processed[actual_columns]

    # Days that have BOTH videos at the scheduling times and meter data.
    video_days = sorted({
        VisionModule.parse_video_time(p.name).date()
        for p in VIDEO_DIR.glob("*.webm")
        if VisionModule.parse_video_time(p.name) is not None
    })
    meter_days = set(processed["timestamp"].dt.date)
    days = [d for d in video_days if d in meter_days]

    predictor = HybridPredictor()
    fusion = FeatureFusion()

    print("=" * 88)
    print("SCHEDULE RECONSTRUCTION - the mentor's evaluation workflow")
    print("=" * 88)
    print("Each scheduling time rewrites only the FUTURE blocks, so the day builds")
    print("into ONE final schedule - exactly what would have been published live.")
    print()

    all_rows = []

    for day in days:

        print("-" * 88)
        print(f"DAY {day}")

        # Which videos actually exist for this day's scheduling times
        available = [
            (rt, find_video_for(day, rt)) for rt in RUN_TIMES
        ]
        have = sum(1 for _, v in available if v is not None)
        print(f"  Windy captures available: {have} / {len(RUN_TIMES)} scheduling times")
        if have < len(RUN_TIMES):
            missing = [rt for rt, v in available if v is None]
            print(f"  missing at: {', '.join(missing)}")
        print()

        results = {}
        for label, provider in [("no vision", None),
                                ("Gemini", "gemini"),
                                ("ChatGPT", "openai")]:

            final, contributions = reconstruct_day(
                predictor, fusion, processed, day, provider
            )
            score = score_schedule(final, actual)

            if score is None:
                print(f"  {label:<10}: no real meter blocks to score against")
                continue

            results[label] = score

            print(f"  {label:<10}: deviation {score['deviation_pct']:6.2f}%   "
                  f"MAE {score['mae_kw']:7.1f} kW   "
                  f"scheduled {score['avg_scheduled_mw']:.3f} MW vs "
                  f"actual {score['avg_actual_mw']:.3f} MW")

            all_rows.append({
                "date": str(day),
                "approach": label,
                "blocks_scored": score["blocks"],
                "deviation_pct": round(score["deviation_pct"], 3),
                "mae_kw": round(score["mae_kw"], 1),
                "rmse_kw": round(score["rmse_kw"], 1),
                "avg_scheduled_mw": round(score["avg_scheduled_mw"], 3),
                "avg_actual_mw": round(score["avg_actual_mw"], 3),
                "scheduled_mwh": round(score["scheduled_mwh"], 3),
                "actual_mwh": round(score["actual_mwh"], 3),
            })

        if results:
            best = min(results, key=lambda k: results[k]["deviation_pct"])
            print(f"  -> closest to actual: {best}")

    if not all_rows:
        print("\nNo days could be reconstructed.")
        return

    table = pd.DataFrame(all_rows)

    out = Path("outputs/reports/schedule_reconstruction.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print()
    print("=" * 88)
    print("SUMMARY - average across all reconstructed days")
    print("=" * 88)
    summary = table.groupby("approach").agg(
        days=("date", "nunique"),
        avg_deviation_pct=("deviation_pct", "mean"),
        avg_mae_kw=("mae_kw", "mean"),
    ).sort_values("avg_deviation_pct").round(3)
    print(summary.to_string())

    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
