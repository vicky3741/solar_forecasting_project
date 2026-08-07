"""Honest future-day comparison for the forecast components.

The first seven available dates choose the weather blend and train the
optional residual correction.  The last six dates are untouched until final
scoring.  This is deliberately a temporal split: no later day can influence
an earlier deployment decision.

Historical weather is retrieved through OpenMeteoClient's single-run archive,
using a conservative publication delay configured in settings.yaml.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.config import settings
from modules.evaluation import metrics
from modules.evaluation.backtester import Backtester
from modules.forecasting.residual_correction import (
from utils.file_manager import processed_data_path
    FEATURES,
    NUM_BOOST_ROUND,
    TRAINING_PARAMS,
)


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
WEATHER_WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0]


def detail_at_weight(backtester, cache, weight):
    """Reblend cached point-in-time signals without touching future actuals."""

    frames = []

    for (date, run_time_str), cached in cache.items():
        forecast = backtester.predictor.blend_signals(
            cached["signals"],
            weather_weight=weight,
            vision_adjustment=cached["vision_adjustment"],
        )
        merged = forecast.merge(cached["actual"], on="timestamp", how="inner")

        if merged.empty:
            continue

        run_time = pd.Timestamp(f"{date} {run_time_str}")
        merged.insert(0, "date", str(date))
        merged.insert(1, "run_time", run_time_str)
        merged["horizon_min"] = (
            (merged["timestamp"] - run_time).dt.total_seconds() / 60
        )
        merged["block_hour"] = (
            merged["timestamp"].dt.hour + merged["timestamp"].dt.minute / 60
        )
        merged["kt_now"] = cached["signals"]["kt_now"]
        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def run_scores(detail):
    """Equal-weight each official forecast run, matching the project metric."""

    scored = detail.copy()
    scored["deviation_pct"] = metrics.percentage_deviation(
        scored["final_forecast_kw"], scored["active_power_kw"], CAPACITY_KW
    )

    per_run = scored.groupby(["date", "run_time"], as_index=False).agg(
        deviation_pct=("deviation_pct", "mean"),
        mae_kw=("final_forecast_kw", lambda x: 0.0),
    )

    # Calculate MAE without relying on an ambiguous groupby lambda index.
    mae = scored.assign(
        abs_error_kw=(scored["final_forecast_kw"] - scored["active_power_kw"]).abs()
    ).groupby(["date", "run_time"])["abs_error_kw"].mean()
    per_run["mae_kw"] = [mae[(d, r)] for d, r in zip(per_run.date, per_run.run_time)]

    return {
        "runs": len(per_run),
        "blocks": len(scored),
        "avg_deviation_pct": float(per_run["deviation_pct"].mean()),
        "avg_mae_kw": float(per_run["mae_kw"].mean()),
    }


def residual_correct(train, test):
    """Fit only on older days, then correct untouched future-day forecasts."""

    train = train.copy()
    test = test.copy()
    train["residual_kw"] = train["active_power_kw"] - train["final_forecast_kw"]

    train_set = lgb.Dataset(train[FEATURES], label=train["residual_kw"])
    model = lgb.train(TRAINING_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)

    test["final_forecast_kw"] = np.clip(
        test["final_forecast_kw"] + model.predict(test[FEATURES]),
        0,
        CAPACITY_KW,
    )
    return test


def main():
    processed = pd.read_csv(processed_data_path(), parse_dates=["timestamp"])
    backtester = Backtester()

    print("Running point-in-time signals. Historical weather uses archived single runs...")
    results, cache, _ = backtester.run(processed)

    dates = sorted(str(d) for d in processed["timestamp"].dt.date.unique())
    train_dates = set(dates[:7])
    test_dates = set(dates[7:])

    print(f"Selection/training dates: {min(train_dates)} to {max(train_dates)}")
    print(f"Untouched future test dates: {min(test_dates)} to {max(test_dates)}")

    weather_available = sum(
        cached["signals"].get("weather_kt") is not None for cached in cache.values()
    )
    print(f"Weather signal available: {weather_available}/{len(cache)} official runs")

    by_weight = {}
    for weight in WEATHER_WEIGHTS:
        detail = detail_at_weight(backtester, cache, weight)
        train = detail[detail["date"].isin(train_dates)]
        test = detail[detail["date"].isin(test_dates)]
        by_weight[weight] = (train, test, run_scores(train), run_scores(test))

    selected_weight = min(
        WEATHER_WEIGHTS,
        key=lambda weight: by_weight[weight][2]["avg_deviation_pct"],
    )

    base_test = by_weight[0.0][1]
    weather_test = by_weight[selected_weight][1]
    base_score = run_scores(base_test)
    weather_score = run_scores(weather_test)

    residual_test = residual_correct(
        by_weight[selected_weight][0], weather_test
    )
    residual_score = run_scores(residual_test)

    test_vision_runs = int(
        results[results["date"].astype(str).isin(test_dates)]["has_vision"].sum()
    )
    train_vision_runs = int(
        results[results["date"].astype(str).isin(train_dates)]["has_vision"].sum()
    )

    report = pd.DataFrame([
        {
            "component": "weather",
            "candidate": "weather off",
            "selected_on_past_days": True,
            **base_score,
        },
        {
            "component": "weather",
            "candidate": f"weather weight {selected_weight}",
            "selected_on_past_days": True,
            **weather_score,
        },
        {
            "component": "residual",
            "candidate": f"residual after weather weight {selected_weight}",
            "selected_on_past_days": True,
            **residual_score,
        },
    ])

    output = Path("outputs/reports/honest_future_day_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    print("\nFuture-day results (lower deviation is better)")
    print(report.round(3).to_string(index=False))
    print(
        f"\nWeather change on untouched days: "
        f"{base_score['avg_deviation_pct'] - weather_score['avg_deviation_pct']:+.3f} percentage points"
    )
    print(
        f"Residual change on untouched days: "
        f"{weather_score['avg_deviation_pct'] - residual_score['avg_deviation_pct']:+.3f} percentage points"
    )
    print(
        f"Vision coverage: {train_vision_runs} runs in training, "
        f"{test_vision_runs} runs in future test. "
        + ("A future-day vision comparison is possible." if test_vision_runs else
           "Vision verdict: INCONCLUSIVE (no held-out day has video coverage).")
    )
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
