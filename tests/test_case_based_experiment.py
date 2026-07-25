"""
=========================================================
Solar Forecasting Project
Case-Based Correction Experiment (k-NN retrieval)
=========================================================
Tests whether teammate Kushal's case-based-reasoning idea
- look up similar past situations and use how they actually
turned out - improves OUR forecast on unseen days.

Same honesty rules as the LightGBM residual experiment, so
the two are directly comparable:
  - Leave-one-day-out: a day's correction is never built
    from that day's own data.
  - Only real measured blocks (Part 2) are in the case
    store - never reconstructed ones.
  - The verdict prints either way; it ships only if it
    helps out-of-sample.

Run:  python -m tests.test_case_based_experiment
=========================================================
"""

from pathlib import Path

import pandas as pd

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.forecasting.case_based_correction import CaseBasedCorrector, FEATURES
from modules.evaluation import metrics
from tests.test_residual_experiment import compute_kt_now_lookup


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000


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

    print("Computing kt-at-run-time for every run...")
    kt_lookup = compute_kt_now_lookup(processed, clearsky)

    detail["date"] = detail["date"].astype(str)
    run_dt = pd.to_datetime(detail["date"] + " " + detail["run_time"])

    detail["horizon_min"] = (detail["timestamp"] - run_dt).dt.total_seconds() / 60
    detail["block_hour"] = (
        detail["timestamp"].dt.hour + detail["timestamp"].dt.minute / 60
    )
    detail["kt_now"] = [
        kt_lookup[(d, r)] for d, r in zip(detail["date"], detail["run_time"])
    ]
    detail["residual_kw"] = detail["active_power_kw"] - detail["final_forecast_kw"]

    print(f"Dataset: {len(detail)} REAL blocks across "
          f"{detail['date'].nunique()} days\n")

    corrector = CaseBasedCorrector()
    print(f"Case-based settings: k={corrector.k}, strength={corrector.strength}, "
          f"weights={corrector.weights}\n")

    rows = []

    for held_out in sorted(detail["date"].unique()):

        train = detail[detail["date"] != held_out]
        test = detail[detail["date"] == held_out].copy()

        corrector.fit(train[FEATURES], train["residual_kw"])

        # Same gentle-nudge damping the live corrector uses.
        predicted_residual = (
            corrector.predict_residual(test[FEATURES]) * corrector.strength
        )

        test["corrected_kw"] = (
            test["final_forecast_kw"] + predicted_residual
        ).clip(0, CAPACITY_KW)

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

    print("Leave-one-day-out (positive improvement = case-based helped)\n")
    print(results.round(3).to_string(index=False))

    overall_before = results["deviation_before_pct"].mean()
    overall_after = results["deviation_after_pct"].mean()
    days_helped = int((results["improvement_pct_points"] > 0).sum())

    print("\n================ VERDICT ================")
    print(f"Avg deviation before : {overall_before:.3f} %")
    print(f"Avg deviation after  : {overall_after:.3f} %")
    print(f"Improvement          : {overall_before - overall_after:+.3f} pct points")
    print(f"Days helped          : {days_helped} / {len(results)}")

    verdict_positive = (
        overall_after < overall_before and days_helped > len(results) / 2
    )

    output_path = Path("outputs/reports/case_based_experiment.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    if verdict_positive:
        print("\nRESULT: case-based reasoning HELPS on unseen days - worth integrating.")

        # Save the production case store on the FULL history - only
        # after the out-of-sample verdict justified shipping it.
        store_path = corrector.save_case_store(
            detail, detail["residual_kw"], detail["date"]
        )
        print(f"Case store saved on all {detail['date'].nunique()} days: {store_path}")
        print(f"(live corrector activates at >= {corrector.min_case_days} days)")
    else:
        print("\nRESULT: case-based reasoning does NOT reliably help - keep it off for now.")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
