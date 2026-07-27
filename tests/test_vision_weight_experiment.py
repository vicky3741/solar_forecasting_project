"""
=========================================================
Solar Forecasting Project
Vision Signal Test - 7/7 coverage days only
=========================================================
Does the Windy vision signal at its SHIPPED strength (0.15)
actually improve the forecast?

SELF-CONTAINED BY DESIGN. This file reproduces the blend
arithmetic locally instead of adding switches to
modules/forecasting/predictor.py, so the production
pipeline carries no experiment scaffolding and running this
can never change what the server publishes.

Only days with a Windy clip at ALL SEVEN scheduling times
are used. Partial days are excluded on purpose: a missing
clip makes that run fall back to no-vision, which quietly
scores as the baseline and drags every comparison towards
"no difference".

Two placements are compared, both at strength 0.15:

  as shipped - vision scales the persistence term, and is
      then diluted by the Chronos split (x0.8) and the
      weather split (x0.35), so ~28% of it survives.
  undiluted  - vision scales the blended kt instead, so it
      keeps its full stated size.

Scored against REAL measured meter blocks only.

Run:  python -m tests.test_vision_weight_experiment
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.preprocessing.preprocess import DataPreprocessor
from modules.evaluation import metrics
from tests.test_schedule_reconstruction import find_video_for, vision_features_for


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]
STRENGTH = 0.15                      # the shipped value, the only one tested here
MAX_KT = 1.2


def load_data():
    pre = DataPreprocessor()
    frames = [
        pre.preprocess(file_path=p, required_columns=["TimeStamp"],
                       timestamp_column="TimeStamp")
        for p in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def blend(predictor, signals, adjustment, placement):
    """
    Local copy of predictor.blend_signals' kt arithmetic, with the
    vision adjustment applied at one of two points. Deliberately not a
    call into the predictor with extra switches - see the docstring.
    """

    alpha = predictor.chronos_weight * signals["context_ratio"]
    weather_weight = predictor.weather_weight

    persistence_adj = adjustment if placement == "as shipped" else 0.0

    kt_persistence = np.clip(
        signals["kt_now"] * (1 + persistence_adj), 0, MAX_KT
    )

    base_kt = (
        alpha * signals["chronos_kt_forecast"] + (1 - alpha) * kt_persistence
    )

    weather_kt = signals.get("weather_kt")
    if weather_kt is not None and weather_weight > 0:
        final_kt = weather_weight * weather_kt + (1 - weather_weight) * base_kt
    else:
        final_kt = base_kt

    if placement == "undiluted":
        final_kt = np.clip(final_kt * (1 + adjustment), 0, MAX_KT)

    clearsky_kw = (
        signals["poa_curve"] / 1000 * CAPACITY_KW * predictor.performance_ratio
    )

    return pd.DataFrame({
        "timestamp": signals["forecast_timestamps"],
        "final_forecast_kw": np.clip(final_kt * clearsky_kw, 0, CAPACITY_KW),
    })


def main():

    data = load_data()

    real = (
        data[data["is_real_measurement"].fillna(False)]
        if "is_real_measurement" in data.columns else data
    )

    predictor = HybridPredictor()
    fusion = FeatureFusion()
    provider = settings["vision"].get("provider", "gemini")

    print("=" * 88)
    print(f"VISION SIGNAL TEST at shipped strength {STRENGTH} - 7/7 coverage days only")
    print("=" * 88)
    print(f"provider: {provider}")
    print()

    # --- find the days with a clip at every scheduling time ---
    full_days = []
    for day in sorted(set(real["timestamp"].dt.date)):
        clips = {rt: find_video_for(day, rt) for rt in RUN_TIMES}
        have = [rt for rt, v in clips.items() if v is not None]
        if len(have) == len(RUN_TIMES):
            full_days.append((day, clips))
        elif have:
            print(f"  skipping {day}: only {len(have)}/{len(RUN_TIMES)} clips")

    if not full_days:
        print("\nNo day has all seven clips yet.")
        return

    print(f"\ndays with 7/7 coverage: {', '.join(str(d) for d, _ in full_days)}")
    print()

    rows = []
    vision_failures = 0

    for day, clips in full_days:

        day_actual = real[real["timestamp"].dt.date == day][
            ["timestamp", "active_power_kw"]
        ]
        if day_actual.empty:
            continue

        for run_str in RUN_TIMES:

            hour, minute = map(int, run_str.split(":"))
            run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            features = vision_features_for(provider, clips[run_str])
            if not features:
                vision_failures += 1
                print(f"  !! {day} {run_str}: vision read FAILED - run excluded")
                continue

            signals = predictor.compute_signals(data, run_time)

            adjustment = fusion.trend_adjustment_profile(
                features, signals["horizon_minutes"], max_adjustment=STRENGTH
            )

            for label, placement in (
                ("vision off", None),
                ("vision as shipped", "as shipped"),
                ("vision undiluted", "undiluted"),
            ):
                forecast = blend(
                    predictor, signals,
                    0.0 if placement is None else adjustment,
                    placement or "as shipped",
                )

                merged = forecast.merge(day_actual, on="timestamp", how="inner")
                if len(merged) < 8:
                    continue

                rows.append({
                    "date": str(day),
                    "run_time": run_str,
                    "variant": label,
                    "blocks": len(merged),
                    "deviation_pct": metrics.average_percentage_deviation(
                        merged["final_forecast_kw"],
                        merged["active_power_kw"],
                        CAPACITY_KW,
                    ),
                    "mae_kw": metrics.mean_absolute_error(
                        merged["final_forecast_kw"], merged["active_power_kw"]
                    ),
                })

    if not rows:
        print("Nothing could be scored.")
        return

    table = pd.DataFrame(rows)
    out = Path("outputs/reports/vision_signal_test.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(out, index=False)

    print()
    print("=" * 88)
    print("PER DAY (average across that day's seven runs)")
    print("=" * 88)
    per_day = table.groupby(["date", "variant"])["deviation_pct"].mean().unstack()
    per_day = per_day[["vision off", "vision as shipped", "vision undiluted"]]
    per_day["shipped helps"] = (
        per_day["vision off"] - per_day["vision as shipped"]
    ).round(3)
    print(per_day.round(3).to_string())

    print()
    print("=" * 88)
    print("PER SCHEDULING TIME (average across days)")
    print("=" * 88)
    print(
        table.groupby(["run_time", "variant"])["deviation_pct"].mean()
        .unstack()[["vision off", "vision as shipped", "vision undiluted"]]
        .round(3).to_string()
    )

    print()
    print("=" * 88)
    print("OVERALL")
    print("=" * 88)
    overall = table.groupby("variant").agg(
        runs=("deviation_pct", "size"),
        avg_deviation_pct=("deviation_pct", "mean"),
        avg_mae_kw=("mae_kw", "mean"),
    ).round(3)
    print(overall.to_string())

    off = overall.loc["vision off", "avg_deviation_pct"]
    shipped = overall.loc["vision as shipped", "avg_deviation_pct"]
    undiluted = overall.loc["vision undiluted", "avg_deviation_pct"]

    print()
    print(f"vision as shipped vs off : {off - shipped:+.3f} pts")
    print(f"vision undiluted  vs off : {off - undiluted:+.3f} pts")
    if vision_failures:
        print(f"\nNOTE: {vision_failures} run(s) excluded - the vision model would "
              "not answer (free-tier busy).")

    helped = int((per_day["shipped helps"] > 0).sum())
    print(f"\ndays where the shipped setting helped: {helped}/{len(per_day)}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
