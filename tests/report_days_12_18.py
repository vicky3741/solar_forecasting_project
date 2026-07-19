"""
=========================================================
Solar Forecasting Project
Days 12-18 Accuracy Report (out-of-sample, numbers only)
=========================================================
Mentor task: "Apply the developed AI model to this data and
do the prediction of meter data. Check accuracy of the
results and share the same on the group."

Produces a NUMBERS table (predicted vs actual meter
generation + accuracy %) for 2026-07-12 .. 2026-07-18.

Method / honesty:
  - Accuracy is averaged across all 7 official intraday runs
    per day (06:45 .. 15:45), matching how the system really
    operates - it re-forecasts 7x/day. Judging a single
    dawn run would be unrepresentative (at 06:45 there are
    only ~2 readings and dawn haze makes the estimate noisy).
  - Each run uses only data available at its run time
    (no lookahead).
  - The LightGBM residual correction is applied WALK-FORWARD:
    day D is corrected only by a model trained on days
    BEFORE D, and only once >= min_training_days of history
    exist - so days 12-18 are genuinely out-of-sample.
  - Predicted (MWh) per day = the intraday full-day energy
    estimate (actual generation so far + forecast for the
    rest), averaged across the 7 runs.
  - Accuracy % = 100 - mean(|predicted - actual|)/capacity.
=========================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.forecasting.residual_correction import (
    FEATURES, TRAINING_PARAMS, NUM_BOOST_ROUND
)
from tests.test_residual_experiment import compute_kt_now_lookup


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
MIN_TRAINING_DAYS = settings["residual_correction"]["min_training_days"]
RUN_TIMES = settings["forecast"]["run_times"]
REPORT_DAYS = [f"2026-07-{d}" for d in range(12, 19)]


def build_features(detail, kt_lookup):

    detail = detail.copy()
    detail["date"] = detail["date"].astype(str)
    run_dt = pd.to_datetime(detail["date"] + " " + detail["run_time"])

    detail["horizon_min"] = (detail["timestamp"] - run_dt).dt.total_seconds() / 60
    detail["block_hour"] = detail["timestamp"].dt.hour + detail["timestamp"].dt.minute / 60
    detail["kt_now"] = [
        kt_lookup[(d, r)] for d, r in zip(detail["date"], detail["run_time"])
    ]
    detail["residual_kw"] = detail["active_power_kw"] - detail["final_forecast_kw"]

    return detail


def train_past_only(detail, day):
    """
    LightGBM residual model trained only on days before `day`.
    Returns None when there is not yet enough history (the
    production min-training-days gate) - callers then apply no
    correction, exactly as production would.
    """

    past_days = [d for d in sorted(detail["date"].unique()) if d < day]

    if len(past_days) < MIN_TRAINING_DAYS:
        return None

    train = detail[detail["date"].isin(past_days)]
    train_set = lgb.Dataset(train[FEATURES], label=train["residual_kw"])

    return lgb.train(TRAINING_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)


def main():

    detail = pd.read_csv("outputs/reports/backtest_detail.csv", parse_dates=["timestamp"])
    processed = pd.read_csv("data/processed/processed_data.csv", parse_dates=["timestamp"])

    clearsky = ClearSkyModel()
    kt_lookup = compute_kt_now_lookup(processed, clearsky)
    detail = build_features(detail, kt_lookup)

    processed["date"] = processed["timestamp"].dt.date.astype(str)

    rows = []

    for day in REPORT_DAYS:

        model = train_past_only(detail, day)

        day_proc = processed[processed["date"] == day]
        actual_full_mwh = (day_proc["active_power_kw"].clip(lower=0) * 0.25).sum() / 1000

        run_energies = []
        run_deviations = []

        for run_time in RUN_TIMES:

            run_rows = detail[(detail["date"] == day) & (detail["run_time"] == run_time)]
            if run_rows.empty:
                continue

            correction = (
                model.predict(run_rows[FEATURES]) if model is not None else 0.0
            )
            predicted = np.clip(
                run_rows["final_forecast_kw"].to_numpy() + correction, 0, CAPACITY_KW
            )
            actual = run_rows["active_power_kw"].to_numpy()

            run_deviations.append(np.mean(np.abs(predicted - actual)) / CAPACITY_KW * 100)

            run_ts = pd.Timestamp(f"{day} {run_time}")
            morning_mwh = (
                day_proc[day_proc["timestamp"] <= run_ts]["active_power_kw"].clip(lower=0)
                * 0.25
            ).sum() / 1000
            forecast_mwh = predicted.sum() * 0.25 / 1000
            run_energies.append(morning_mwh + forecast_mwh)

        rows.append({
            "Date": day,
            "Predicted (MWh)": float(np.mean(run_energies)),
            "Actual (MWh)": actual_full_mwh,
            "Accuracy %": 100 - float(np.mean(run_deviations)),
            "Out-of-sample": "yes" if model is not None else "warm-up"
        })

    report = pd.DataFrame(rows)
    line = "=" * 70

    print("\n" + line)
    print(" SIRMOUR 5.1 MW  -  AI METER FORECAST vs ACTUAL   (Jul 12-18, 2026)")
    print(" Accuracy averaged across all 7 intraday runs/day - out-of-sample")
    print(line)
    print(report.to_string(
        index=False,
        formatters={
            "Predicted (MWh)": "{:.1f}".format,
            "Actual (MWh)": "{:.1f}".format,
            "Accuracy %": "{:.1f}".format
        }
    ))
    print(line)

    overall_acc = report["Accuracy %"].mean()
    total_pred = report["Predicted (MWh)"].sum()
    total_act = report["Actual (MWh)"].sum()
    energy_acc = 100 - abs(total_pred - total_act) / total_act * 100

    print(f" OVERALL forecast accuracy       : {overall_acc:.1f} %")
    print(f" Total predicted energy          : {total_pred:.1f} MWh")
    print(f" Total actual energy             : {total_act:.1f} MWh")
    print(f" Total-energy accuracy           : {energy_acc:.1f} %")
    print(line)

    report.to_csv("outputs/reports/days_12_18_accuracy.csv", index=False)
    print(" Saved: outputs/reports/days_12_18_accuracy.csv\n")


if __name__ == "__main__":
    main()
