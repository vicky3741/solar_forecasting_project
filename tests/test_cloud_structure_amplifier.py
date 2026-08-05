"""
=========================================================
Solar Forecasting Project
Cloud-Structure Vision Amplifier - Sanity Check
=========================================================
cloud_field_structure (solid/patchy/broken) exists in real
cached Gemini results for exactly 2 days so far: 2026-08-03
and 2026-08-04. That is NOT enough for a trustworthy leave-
one-day-out verdict - it is a sanity check only, to confirm
the mechanism (modules/fusion/fusion.py's
structure_multiplier) does something sensible before more
data accumulates. DO NOT ship based on this result alone.

Compares, on those 2 real days, at the CURRENT live weight
(0.25): today's default (no amplification) vs the candidate
(patchy x1.5, broken x2.0), scored on real DSM Rs penalty.

Run:  python -m tests.test_cloud_structure_amplifier
=========================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.preprocessing.preprocess import DataPreprocessor
from tests.test_schedule_reconstruction import find_video_for
from tests.test_weather_weight_dsm_retune import dsm_penalty_rs

CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]
CACHE_ROOT = Path("outputs/llm_compare/gemini")
CURRENT_WEIGHT = 0.25

DAYS = [pd.Timestamp("2026-08-03").date(), pd.Timestamp("2026-08-04").date()]


def load_data():

    pre = DataPreprocessor()

    frames = [
        pre.preprocess(
            file_path=path, required_columns=["TimeStamp"],
            timestamp_column="TimeStamp",
        )
        for path in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp").reset_index(drop=True)
    )


def cached_features(day, run_time_str):

    video = find_video_for(day, run_time_str)
    if video is None:
        return None

    cache_file = CACHE_ROOT / Path(video).stem / "vision_result.json"
    if not cache_file.exists():
        return None

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    features = cached.get("weather_features")

    if not features or "_error" in features:
        return None

    return features


def build_day(predictor, fusion, data, actual, day, amplify):

    schedule = {}

    for run_str in RUN_TIMES:

        hour, minute = map(int, run_str.split(":"))
        run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

        signals = predictor.compute_signals(data, run_time)
        features = cached_features(day, run_str)

        adjustment = 0.0
        if features is not None:
            adjustment = fusion.trend_adjustment_profile(
                features, signals["horizon_minutes"],
                structure_multiplier=amplify,
            )

        forecast = predictor.blend_signals(
            signals, weather_weight=CURRENT_WEIGHT, vision_adjustment=adjustment
        )

        for ts, kw in zip(forecast["timestamp"], forecast["final_forecast_kw"]):
            if ts > run_time:
                schedule[ts] = kw

    keep = [t for t in sorted(schedule) if t in actual.index]
    measured = actual.reindex(keep).to_numpy(dtype=float)
    good = ~np.isnan(measured)

    forecast_kw = np.array([schedule[t] for t in keep])[good]
    actual_kw = measured[good]

    deviation_mw = (actual_kw - forecast_kw) / 1000.0

    return float(dsm_penalty_rs(deviation_mw).sum()), int(good.sum())


def main():

    data = load_data()
    actual = data[data["is_real_measurement"].fillna(False)]
    actual = actual.set_index("timestamp")["active_power_kw"]

    predictor = HybridPredictor()
    fusion = FeatureFusion()

    print("=" * 70)
    print("CLOUD-STRUCTURE AMPLIFIER - SANITY CHECK (n=2 days, NOT a verdict)")
    print("=" * 70)

    rows = []
    for day in DAYS:

        structures = [cached_features(day, rt) for rt in RUN_TIMES]
        structures = [f.get("cloud_field_structure") for f in structures if f]
        print(f"\n{day}  structures seen: {structures}")

        base_rs, n = build_day(predictor, fusion, data, actual, day, amplify=False)
        amp_rs, _ = build_day(predictor, fusion, data, actual, day, amplify=True)

        print(f"  blocks scored          : {n}")
        print(f"  penalty WITHOUT amplify : Rs {base_rs:.1f}")
        print(f"  penalty WITH amplify    : Rs {amp_rs:.1f}")
        print(f"  difference              : {base_rs - amp_rs:+.1f} Rs "
              f"({'better' if amp_rs < base_rs else 'worse'})")

        rows.append({"day": str(day), "blocks": n,
                     "penalty_no_amplify_rs": round(base_rs, 1),
                     "penalty_amplify_rs": round(amp_rs, 1)})

    total_base = sum(r["penalty_no_amplify_rs"] for r in rows)
    total_amp = sum(r["penalty_amplify_rs"] for r in rows)

    print()
    print("-" * 70)
    print(f"TOTAL across both days: without {total_base:.1f} Rs, "
          f"with {total_amp:.1f} Rs  ({total_base - total_amp:+.1f} Rs)")
    print()
    print("n=2 - this is a mechanism check, NOT a leave-one-day-out result.")
    print("Do not change predictor.py/pipeline.py defaults from this alone.")
    print("Re-run once ~5-7 real cloud_field_structure days exist.")

    out = Path("outputs/reports/cloud_structure_amplifier_sanity.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
