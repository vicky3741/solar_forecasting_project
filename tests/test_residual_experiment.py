"""
=========================================================
Solar Forecasting Project
Residual Correction Experiment (LightGBM)
=========================================================
Tests whether a small LightGBM model can learn our
forecast's systematic mistakes and correct them.

Honesty rules baked in:
  - Leave-one-day-out: train on 12 days, test on the
    held-out 13th, repeat for every day. A day's correction
    is NEVER trained on that day's own data.
  - Tiny model, 4 features (block hour, forecast horizon,
    kt at run time, our forecast value) - all known at
    prediction time, zero future information.
  - The verdict is printed either way. If unseen-day
    deviation does not improve, the model does not ship.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.forecasting.residual_correction import (
    ResidualCorrector,
    FEATURES,
    TRAINING_PARAMS,
    NUM_BOOST_ROUND
)
from modules.evaluation import metrics


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000


def compute_kt_now_lookup(processed, clearsky):
    """
    kt at each (date, run_time) - the same 'current cloudiness'
    anchor the predictor uses, recomputed here per run. Uses
    only readings at/before the run time (no lookahead).
    """

    run_times = settings["forecast"]["run_times"]

    lookup = {}

    for date in sorted(processed["timestamp"].dt.date.unique()):

        today = processed[processed["timestamp"].dt.date == date]

        for run_time_str in run_times:

            hour, minute = map(int, run_time_str.split(":"))
            run_time = pd.Timestamp(date) + pd.Timedelta(hours=hour, minutes=minute)

            upto = today[today["timestamp"] <= run_time]

            with_kt = clearsky.compute_clear_sky_index(
                upto, ghi_column="ghi_w_m2", timestamp_column="timestamp"
            )

            recent = with_kt["clear_sky_index"].dropna().tail(4)

            lookup[(str(date), run_time_str)] = (
                float(recent.mean()) if not recent.empty else 1.0
            )

    return lookup


def main():

    detail = pd.read_csv(
        "outputs/reports/backtest_detail.csv",
        parse_dates=["timestamp"]
    )

    processed = pd.read_csv(
        "data/processed/processed_data.csv",
        parse_dates=["timestamp"]
    )

    clearsky = ClearSkyModel()

    print("Computing kt-at-run-time for every run (pvlib only, no Chronos)...")
    kt_lookup = compute_kt_now_lookup(processed, clearsky)

    # ---------------- Features and target ----------------

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

    features = FEATURES

    print(f"Dataset: {len(detail)} blocks across "
          f"{detail['date'].nunique()} days / {len(kt_lookup)} runs\n")

    # ---------------- Leave-one-day-out ----------------

    rows = []

    for held_out in sorted(detail["date"].unique()):

        train = detail[detail["date"] != held_out]
        test = detail[detail["date"] == held_out].copy()

        train_set = lgb.Dataset(
            train[features],
            label=train["residual_kw"]
        )

        model = lgb.train(TRAINING_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)

        predicted_residual = model.predict(test[features])

        test["corrected_kw"] = np.clip(
            test["final_forecast_kw"] + predicted_residual,
            0,
            CAPACITY_KW
        )

        before = metrics.average_percentage_deviation(
            test["final_forecast_kw"], test["active_power_kw"], CAPACITY_KW
        )
        after = metrics.average_percentage_deviation(
            test["corrected_kw"], test["active_power_kw"], CAPACITY_KW
        )

        rows.append({
            "held_out_day": held_out,
            "blocks": len(test),
            "deviation_before_pct": before,
            "deviation_after_pct": after,
            "improvement_pct_points": before - after
        })

    results = pd.DataFrame(rows)

    pd.set_option("display.width", 140)

    print("Leave-one-day-out results (positive improvement = correction helped)\n")
    print(results.round(3).to_string(index=False))

    overall_before = results["deviation_before_pct"].mean()
    overall_after = results["deviation_after_pct"].mean()
    days_helped = int((results["improvement_pct_points"] > 0).sum())

    print("\n================ VERDICT ================")
    print(f"Avg deviation before correction : {overall_before:.3f} %")
    print(f"Avg deviation after correction  : {overall_after:.3f} %")
    print(f"Improvement                     : {overall_before - overall_after:+.3f} pct points")
    print(f"Days helped                     : {days_helped} / {len(results)}")

    verdict_positive = (
        overall_after < overall_before and days_helped > len(results) / 2
    )

    if verdict_positive:
        print("\nRESULT: correction HELPS on unseen days - worth integrating.")
    else:
        print("\nRESULT: correction does NOT reliably help - do not ship it.")

    output_path = Path("outputs/reports/residual_experiment.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Train the production model on the FULL history - only
    # when the out-of-sample verdict justified shipping it.
    if verdict_positive:

        corrector = ResidualCorrector()

        model_path = corrector.train_and_save(
            detail[features],
            detail["residual_kw"]
        )

        print(f"Production model trained on all days and saved: {model_path}")


if __name__ == "__main__":
    main()
