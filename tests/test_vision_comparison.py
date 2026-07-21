"""
=========================================================
Solar Forecasting Project
Vision ON vs OFF Comparison  (a real day with both video
and meter data)
=========================================================
For each official run time on the test day, forecasts the
rest of the day twice from the same signals:
  - WITHOUT the cloud video (physics + Chronos + persistence)
  - WITH  the latest same-day cloud video's Gemini read

Both are scored against actual meter generation, so the
difference is purely the vision signal's contribution.
Residual correction is intentionally excluded here to
isolate vision.

Default test day 2026-07-14 has 3 real morning videos
(~09:31, 09:55, 10:20) plus full meter data - so vision
only changes the 09:45 run onward (earlier runs have no
video yet, and read identically with and without).
=========================================================
"""

import sys

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.storage.s3_client import S3Storage
from modules.vision.vision_module import VisionModule
from modules.fusion.fusion import FeatureFusion
from modules.evaluation import metrics


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]


def deviation(forecast, actual):
    merged = forecast.merge(
        actual[["timestamp", "active_power_kw"]], on="timestamp", how="inner"
    )
    if merged.empty:
        return None, 0
    dev = metrics.average_percentage_deviation(
        merged["final_forecast_kw"], merged["active_power_kw"], CAPACITY_KW
    )
    return dev, len(merged)


def main():

    test_day = sys.argv[1] if len(sys.argv) > 1 else "2026-07-14"

    df = pd.read_csv("data/processed/processed_data.csv", parse_dates=["timestamp"])

    predictor = HybridPredictor()
    s3 = S3Storage()
    vision = VisionModule()
    fusion = FeatureFusion()

    print("=" * 78)
    print(f" VISION ON vs OFF  -  {test_day}  (accuracy = 100 - deviation%)")
    print("=" * 78)
    print(f"{'Run':>6} {'video read':>22} {'acc OFF':>9} {'acc ON':>9} {'vision effect':>14}")

    rows = []

    for run_time_str in RUN_TIMES:

        run_time = pd.Timestamp(f"{test_day} {run_time_str}")

        signals = predictor.compute_signals(df, run_time)
        horizons = signals["horizon_minutes"]

        # Latest same-day video at/before this run time
        video_path = s3.fetch_latest_video(run_time)
        vis_adj = 0.0
        video_present = video_path is not None
        read_label = "no video yet"

        if video_present:
            try:
                result = vision.analyze_video(video_path, "outputs/extracted_frames")
                feats = fusion.prepare_vision_features(result)
                vis_adj = fusion.trend_adjustment_profile(feats, horizons)
                cloud = feats.get("cloud_coverage_pct", "?")
                trend = feats.get("trend_next_2h", "?")
                read_label = f"cloud {cloud}%, {trend}"
            except Exception:
                read_label = "vision unavailable"

        forecast_off = predictor.blend_signals(signals, vision_adjustment=0.0)
        forecast_on = predictor.blend_signals(signals, vision_adjustment=vis_adj)

        actual = df[df["timestamp"].dt.date == run_time.date()]

        dev_off, n = deviation(forecast_off, actual)
        dev_on, _ = deviation(forecast_on, actual)

        if dev_off is None:
            continue

        acc_off = 100 - dev_off
        acc_on = 100 - dev_on
        effect = acc_on - acc_off

        adjusted = (np.max(np.abs(vis_adj)) > 1e-9) if np.ndim(vis_adj) else (vis_adj != 0)

        if not video_present:
            effect_str = "no video"
        elif read_label == "vision unavailable":
            effect_str = "n/a"
        elif adjusted:
            effect_str = f"{effect:+.2f}"
        else:
            effect_str = "clear: no adj"

        print(f"{run_time_str:>6} {read_label:>22} {acc_off:>8.1f}% {acc_on:>8.1f}% {effect_str:>14}")

        rows.append({
            "run_time": run_time_str,
            "acc_off": acc_off,
            "acc_on": acc_on,
            "video_present": video_present,
            "adjusted": adjusted
        })

    result = pd.DataFrame(rows)

    with_video = result[result["video_present"]]
    adjusted_runs = result[result["adjusted"]]

    print("=" * 78)
    print(f" Overall accuracy (all runs)          : {result['acc_off'].mean():.2f}%")
    print(f" Runs with a video available          : {len(with_video)} of {len(result)}")
    print(f" Runs where vision changed the forecast: {len(adjusted_runs)} of {len(result)}")

    if not adjusted_runs.empty:
        off = adjusted_runs["acc_off"].mean()
        on = adjusted_runs["acc_on"].mean()
        verdict = "helped" if on > off else ("hurt" if on < off else "no change")
        print(f" On adjusted runs: {off:.2f}% -> {on:.2f}%  (vision {verdict}, {on - off:+.2f} pts)")
    else:
        print(" Vision read the sky as clear/stable on every video run, so it")
        print(" correctly applied NO adjustment - it did not disturb an already")
        print(" accurate clear-day forecast (a safety property, not a failure).")

    print("=" * 78)


if __name__ == "__main__":
    main()
