"""
=========================================================
Solar Forecasting Project
LLM Honest Comparison: which model ACTUALLY helps?
=========================================================
tests/test_llm_comparison.py shows what each model SAYS
about the clouds. That alone never tells you which one is
right - two models can disagree and both be wrong, or both
right in different words.

This script answers the real question: for each of today's
runs, feed EACH model's cloud reading into the SAME forecast
pipeline (identical physics, weather, Chronos - only the
vision reading differs), then check which resulting forecast
is closer to what the plant ACTUALLY generated. That is the
same "compare against actual, not against each other"
standard used for every other signal in this project
(weather, case-based reasoning, the residual corrector).

Reads the vision results tests/test_llm_comparison.py already
cached under outputs/llm_compare/<provider>/. Run that first.

Run:  python -m tests.test_llm_honest_comparison
=========================================================
"""

import json
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.evaluation import metrics
from modules.storage.s3_client import S3Storage


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
COMPARE_DIR = Path("outputs/llm_compare")


def load_cached_vision(provider, video_name):
    """
    Reads the vision_result.json test_llm_comparison.py cached
    for one provider + video. Returns None if that provider was
    never successfully run (e.g. no key configured yet).
    """

    cache_file = COMPARE_DIR / provider / Path(video_name).stem / "vision_result.json"

    if not cache_file.exists():
        return None

    with open(cache_file, "r", encoding="utf-8") as file:
        cached = json.load(file)

    features = cached.get("weather_features", {})

    if "_error" in features:
        return None

    return features


def score_run(predictor, fusion, dataframe, run_time, vision_features, actual):
    """
    Builds the forecast for one run using the given vision
    reading (or None for a no-vision baseline), and returns its
    average % deviation against REAL actual blocks only.
    """

    signals = predictor.compute_signals(dataframe, run_time)

    vision_adjustment = fusion.trend_adjustment_profile(
        vision_features, signals["horizon_minutes"]
    ) if vision_features else 0.0

    forecast = predictor.blend_signals(signals, vision_adjustment=vision_adjustment)

    comparison = forecast.merge(actual, on="timestamp", how="inner")

    if "is_real_measurement" in comparison.columns:
        comparison = comparison[comparison["is_real_measurement"].fillna(False)]

    if comparison.empty:
        return None, 0

    deviation = metrics.average_percentage_deviation(
        comparison["final_forecast_kw"], comparison["active_power_kw"], CAPACITY_KW
    )

    return deviation, len(comparison)


def main():

    processed = pd.read_csv(
        "data/processed/processed_data.csv", parse_dates=["timestamp"]
    )

    actual_columns = ["timestamp", "active_power_kw"]
    if "is_real_measurement" in processed.columns:
        actual_columns.append("is_real_measurement")
    actual = processed[actual_columns]

    predictor = HybridPredictor()
    fusion = FeatureFusion()

    videos = sorted(
        Path(settings["windy_capture"]["video_dir"]).glob("SIRMOUR_2026-07-24*.webm")
    )

    gemini_dir_videos = [
        v for v in videos
        if (COMPARE_DIR / "gemini" / v.stem / "vision_result.json").exists()
    ]

    if not gemini_dir_videos:
        print("No cached comparisons found. Run tests/test_llm_comparison.py first.")
        return

    print("=" * 78)
    print("WHICH MODEL ACTUALLY HELPS - graded against REAL meter data")
    print("=" * 78)
    print(f"{'video':38s} {'no-vision':>10s} {'gemini':>10s} {'chatgpt':>10s} {'winner':>10s}")
    print("-" * 78)

    rows = []

    for video in gemini_dir_videos:

        video_time = S3Storage.parse_video_time(video.name)
        run_time = pd.Timestamp(video_time)

        gemini_features = load_cached_vision("gemini", video.name)
        openai_features = load_cached_vision("openai", video.name)

        base_dev, n = score_run(predictor, fusion, processed, run_time, None, actual)
        gem_dev, _ = score_run(predictor, fusion, processed, run_time, gemini_features, actual)
        gpt_dev, _ = score_run(predictor, fusion, processed, run_time, openai_features, actual) \
            if openai_features else (None, 0)

        if base_dev is None:
            print(f"{video.name:38s}  no real actual data yet for this run's blocks")
            continue

        candidates = {"no-vision": base_dev, "gemini": gem_dev}
        if gpt_dev is not None:
            candidates["chatgpt"] = gpt_dev

        winner = min(candidates, key=candidates.get)

        gpt_str = f"{gpt_dev:.2f}" if gpt_dev is not None else "-"

        print(f"{video.name:38s} {base_dev:10.2f} {gem_dev:10.2f} {gpt_str:>10s} {winner:>10s}")

        rows.append({
            "video": video.name, "blocks": n,
            "no_vision_pct": base_dev, "gemini_pct": gem_dev, "chatgpt_pct": gpt_dev,
            "winner": winner
        })

    print("-" * 78)

    if not rows:
        print("\nNo runs could be graded (no real actual data overlapping these run times yet).")
        return

    results = pd.DataFrame(rows)

    print("\nHow to read this:")
    print(" - Each column is that forecast's average %-of-capacity deviation from")
    print("   ACTUAL generation - lower is better, this is the real accuracy test.")
    print(" - 'no-vision' = physics+weather+Chronos alone, no cloud video input at")
    print("   all - the baseline both LLMs are trying to beat.")
    print(" - 'winner' = whichever column had the lowest error for that run.")

    out = Path("outputs/reports/llm_honest_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
