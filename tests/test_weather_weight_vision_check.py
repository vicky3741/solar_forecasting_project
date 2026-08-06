"""
=========================================================
Solar Forecasting Project
Weather-Weight Vision-On Check
=========================================================
The vision-OFF retune (test_weather_weight_retune.py) found
weather_weight=0.65 costing real accuracy against a
leave-one-day-out optimum near 0.20. That result has to be
re-checked with vision ON before it ships, because the
vision adjustment feeds the SAME persistence side of the
blend that weather_weight trades against.

ZERO EXTRA GEMINI CALLS: real vision results are already
cached on disk from building the daily reports (2026-07-27
through 2026-08-03 have near-complete 7/7 coverage). This
reads those cache files DIRECTLY, bypassing
VisionModule.analyze_video's cache-version check - the
PROMPT_VERSION bump for cloud_field_structure would
otherwise treat every one of them as stale and re-call
Gemini. The fields this needs (trend_next_2h, trend_2h_to_4h,
minutes_until_change, confidence) existed before that bump,
so reading the old cache directly is exactly as valid as a
fresh call for this purpose.

Same linear-interpolation trick as the other weight
experiments: for a FIXED vision_adjustment, blend_signals'
output is linear in weather_weight, so two probe blends
(weight 0 and weight 1) are enough to reconstruct the whole
curve. Vision runs on FEWER days than the full 34-day set
(only where a real cached video exists) - that's an honest
smaller sample, not a shortcut.

Run:  python -m tests.test_weather_weight_vision_check
=========================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.evaluation import metrics
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.preprocessing.preprocess import DataPreprocessor
from tests.test_schedule_reconstruction import find_video_for

CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]
CACHE_ROOT = Path("outputs/llm_compare/gemini")

PROBE_PR = 0.30
W_GRID = np.round(np.arange(0.0, 0.951, 0.025), 4)
SHIPPED = 0.65
CANDIDATES = [0.20, 0.25, 0.30, 0.50]


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

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def cached_vision_features(day, run_time_str):
    """
    Real Gemini features for this (day, run) if a video was captured
    AND already analyzed - read straight from disk, ignoring
    PROMPT_VERSION, so this never triggers a new API call. None when
    no video or no cache exists, matching the real no-vision fallback.
    """

    video = find_video_for(day, run_time_str)

    if video is None:
        return None, None

    cache_file = CACHE_ROOT / Path(video).stem / "vision_result.json"

    if not cache_file.exists():
        return video.name, None

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    features = cached.get("weather_features")

    if not features or "_error" in features:
        return video.name, None

    return video.name, features


def collapse_days(data, days, actual):
    """Same collapse as the other experiments, but with REAL vision."""

    predictor = HybridPredictor()
    fusion = FeatureFusion()

    collapsed = {}
    coverage = {}

    for day_index, day in enumerate(days, start=1):

        schedule = {}
        had_vision = 0

        for run_str in RUN_TIMES:

            hour, minute = map(int, run_str.split(":"))
            run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            signals = predictor.compute_signals(data, run_time)

            if signals.get("weather_kt") is None:
                schedule = None
                break

            _, features = cached_vision_features(day, run_str)

            if features is not None:
                had_vision += 1
                adjustment = fusion.trend_adjustment_profile(
                    features, signals["horizon_minutes"]
                )
            else:
                adjustment = 0.0

            probe0 = predictor.blend_signals(
                signals, performance_ratio=PROBE_PR,
                weather_weight=0.0, vision_adjustment=adjustment
            )
            probe1 = predictor.blend_signals(
                signals, performance_ratio=PROBE_PR,
                weather_weight=1.0, vision_adjustment=adjustment
            )

            if max(probe0["final_forecast_kw"].max(),
                   probe1["final_forecast_kw"].max()) >= CAPACITY_KW:
                raise AssertionError(
                    "Capacity clipping bound while building the probe - "
                    "lower PROBE_PR."
                )

            f0 = probe0["final_forecast_kw"].to_numpy() / PROBE_PR
            f1 = probe1["final_forecast_kw"].to_numpy() / PROBE_PR

            timestamps = pd.DatetimeIndex(probe0["timestamp"])

            for ts, a, b in zip(timestamps, f0, f1):
                if ts > run_time:
                    schedule[ts] = (a, b)

        if not schedule:
            continue

        keep = [t for t in sorted(schedule) if t in actual.index]

        if not keep:
            continue

        measured = actual.reindex(keep).to_numpy(dtype=float)
        good = ~np.isnan(measured)

        if not good.any():
            continue

        keep = list(np.array(keep)[good])

        collapsed[day] = {
            "f0": np.array([schedule[t][0] for t in keep]),
            "f1": np.array([schedule[t][1] for t in keep]),
            "actual": measured[good],
        }
        coverage[day] = had_vision

    return collapsed, coverage


def deviation_for(day_data, w, performance_ratio):

    blended = w * day_data["f1"] + (1.0 - w) * day_data["f0"]
    forecast = np.clip(blended * performance_ratio, 0, CAPACITY_KW)

    return metrics.average_percentage_deviation(
        forecast, day_data["actual"], CAPACITY_KW
    )


def main():

    data = load_data()

    actual = data[data["is_real_measurement"].fillna(False)]
    actual = actual.set_index("timestamp")["active_power_kw"]

    days = sorted(data["timestamp"].dt.date.unique())

    print("=" * 72)
    print("WEATHER-WEIGHT CHECK - VISION ON (using cached Gemini results only)")
    print("=" * 72)
    print("Reading cached vision results - zero new Gemini calls.")
    print()

    collapsed, coverage = collapse_days(data, days, actual)

    scored = sorted(collapsed)
    vision_days = [d for d in scored if coverage.get(d, 0) > 0]

    print(f"days usable at all       : {len(scored)}")
    print(f"days with >=1 real vision run : {len(vision_days)}")
    for d in vision_days:
        print(f"  {d}: {coverage[d]}/{len(RUN_TIMES)} runs had real vision")

    if len(vision_days) < 5:
        print("\nToo few vision-covered days for a trustworthy leave-one-day-out "
              "read. Reporting direct comparison only, not LOO.")

    pr = settings["clearsky"]["performance_ratio"]

    print()
    print(f"Scoring over the {len(vision_days)} vision-covered days (PR {pr})...")

    table = {(w, d): deviation_for(collapsed[d], w, pr)
             for w in W_GRID for d in vision_days}

    curve = {w: float(np.mean([table[(w, d)] for d in vision_days])) for w in W_GRID}
    best_w = min(curve, key=curve.get)

    print()
    print("  weight   mean deviation")
    marks = {SHIPPED: "shipped"}
    for c in CANDIDATES:
        marks[c] = "candidate"
    for w in W_GRID:
        tag = ""
        if abs(w - best_w) < 1e-9:
            tag = "  <- best"
        for mw, label in marks.items():
            if abs(w - mw) < 1e-9:
                tag += f"  <- {label} ({mw})"
        if tag:
            print(f"   {w:0.3f}      {curve[w]:6.3f}%{tag}")

    if len(vision_days) >= 5:
        loo_ship, loo_best = [], []
        picks = []
        for held_out in vision_days:
            train = [d for d in vision_days if d != held_out]
            pick = min(W_GRID, key=lambda w: np.mean([table[(w, d)] for d in train]))
            picks.append(pick)
            loo_ship.append(table[(SHIPPED, held_out)])
            loo_best.append(table[(pick, held_out)])

        m_ship = float(np.mean(loo_ship))
        m_best = float(np.mean(loo_best))
        better = sum(1 for a, b in zip(loo_ship, loo_best) if b < a)

        print()
        print("-" * 72)
        print("LEAVE-ONE-DAY-OUT, vision-covered days only")
        print("-" * 72)
        print(f"  shipped 0.65          : {m_ship:.3f}%")
        print(f"  tuned (median {np.median(picks):.2f})  : {m_best:.3f}%   "
              f"({m_ship - m_best:+.3f} pts)")
        print(f"  days improved         : {better}/{len(vision_days)}")

    print()
    print(f"Curve minimum with vision ON: weight={best_w:.3f}, "
          f"deviation={curve[best_w]:.3f}%")
    print(f"Curve minimum WITHOUT vision (previous run): weight~0.20")
    if abs(best_w - 0.20) <= 0.075:
        print("VERDICT: vision-on optimum lands close to the vision-off result "
              "(~0.20-0.30). The earlier finding holds with vision included.")
    else:
        print(f"VERDICT: vision shifts the optimum meaningfully "
              f"(vision-off ~0.20 -> vision-on {best_w:.2f}). Use the vision-on "
              "number, not the vision-off one.")


if __name__ == "__main__":
    main()
