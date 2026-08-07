"""
=========================================================
Solar Forecasting Project
Walk-Forward (Incremental) Residual Learning Experiment
=========================================================
Simulates real day-by-day deployment of the residual
correction, exactly as it would happen live:

  day 4 is corrected by a model trained on days 1-3,
  day 5 by a model trained on days 1-4,
  ...
  day 13 by a model trained on days 1-12.

Each day only ever learns from the PAST - never from the
future. This is stricter and more realistic than
leave-one-day-out (where a held-out early day is corrected
by a model that saw later days).

The first `WARMUP_DAYS` days are left uncorrected - with
almost no mistake history to learn from, a correction
would just add noise.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.forecasting.residual_correction import (
    FEATURES,
    TRAINING_PARAMS,
    NUM_BOOST_ROUND
)
from modules.evaluation import metrics
from tests.test_residual_experiment import compute_kt_now_lookup
from utils.file_manager import processed_data_path


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000

WARMUP_DAYS = 3


def main():

    detail = pd.read_csv(
        "outputs/reports/backtest_detail.csv",
        parse_dates=["timestamp"]
    )

    processed = pd.read_csv(
        processed_data_path(),
        parse_dates=["timestamp"]
    )

    clearsky = ClearSkyModel()

    print("Computing kt-at-run-time for every run...")
    kt_lookup = compute_kt_now_lookup(processed, clearsky)

    detail["date"] = detail["date"].astype(str)

    run_dt = pd.to_datetime(detail["date"] + " " + detail["run_time"])

    detail["horizon_min"] = (
        (detail["timestamp"] - run_dt).dt.total_seconds() / 60
    )
    detail["block_hour"] = (
        detail["timestamp"].dt.hour + detail["timestamp"].dt.minute / 60
    )
    detail["kt_now"] = [
        kt_lookup[(d, r)] for d, r in zip(detail["date"], detail["run_time"])
    ]
    detail["residual_kw"] = detail["active_power_kw"] - detail["final_forecast_kw"]

    days = sorted(detail["date"].unique())

    print(f"Dataset: {len(detail)} blocks, {len(days)} days, "
          f"warm-up = first {WARMUP_DAYS} days (uncorrected)\n")

    # ---------------- Walk forward, day by day ----------------

    rows = []

    for i, day in enumerate(days):

        test = detail[detail["date"] == day].copy()

        before = metrics.average_percentage_deviation(
            test["final_forecast_kw"], test["active_power_kw"], CAPACITY_KW
        )

        if i < WARMUP_DAYS:

            rows.append({
                "day": day,
                "training_days_available": i,
                "deviation_before_pct": before,
                "deviation_after_pct": before,
                "improvement_pct_points": 0.0,
                "corrected": "no (warm-up)"
            })

            continue

        train = detail[detail["date"].isin(days[:i])]

        train_set = lgb.Dataset(train[FEATURES], label=train["residual_kw"])

        model = lgb.train(TRAINING_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)

        corrected = np.clip(
            test["final_forecast_kw"] + model.predict(test[FEATURES]),
            0,
            CAPACITY_KW
        )

        after = metrics.average_percentage_deviation(
            corrected, test["active_power_kw"], CAPACITY_KW
        )

        rows.append({
            "day": day,
            "training_days_available": i,
            "deviation_before_pct": before,
            "deviation_after_pct": after,
            "improvement_pct_points": before - after,
            "corrected": "yes"
        })

    results = pd.DataFrame(rows)

    pd.set_option("display.width", 150)

    print("Day-by-day walk-forward results "
          "(each day corrected ONLY by past days' mistakes)\n")
    print(results.round(3).to_string(index=False))

    corrected_days = results[results["corrected"] == "yes"]

    before_avg = corrected_days["deviation_before_pct"].mean()
    after_avg = corrected_days["deviation_after_pct"].mean()
    helped = int((corrected_days["improvement_pct_points"] > 0).sum())

    print("\n================ VERDICT (corrected days only) ================")
    print(f"Days corrected                  : {len(corrected_days)} "
          f"(days {WARMUP_DAYS + 1}-{len(results)})")
    print(f"Avg deviation before correction : {before_avg:.3f} %")
    print(f"Avg deviation after correction  : {after_avg:.3f} %")
    print(f"Improvement                     : {before_avg - after_avg:+.3f} pct points")
    print(f"Days helped                     : {helped} / {len(corrected_days)}")

    # Does the correction get better as more days accumulate?
    first_half = corrected_days.head(len(corrected_days) // 2)
    second_half = corrected_days.tail(len(corrected_days) - len(corrected_days) // 2)

    print("\nLearning curve (is more history helping?)")
    print(f"  Early corrected days improvement : "
          f"{first_half['improvement_pct_points'].mean():+.3f} pct points")
    print(f"  Later corrected days improvement : "
          f"{second_half['improvement_pct_points'].mean():+.3f} pct points")

    output_path = Path("outputs/reports/walkforward_experiment.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
