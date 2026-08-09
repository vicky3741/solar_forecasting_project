"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Score the LLM schedule against actual generation
=========================================================
The only question that matters about the new pipeline:
is it better than what we already have?

This grades a saved LLM schedule (outputs/llm_schedules/
<PLANT>_<date>_<time>_schedule.csv) against REAL meter data
for the same blocks, and puts two baselines beside it:

  persistence  - hold the last measured clear-sky index flat
                 for the rest of the day. The honest "do
                 nothing clever" baseline; anything that
                 cannot beat this is not earning its
                 complexity.
  clear-sky    - assume no cloud at all. Not a serious
                 forecast, just the ceiling, so the numbers
                 have a scale.

Deviation is average absolute error as a PERCENTAGE OF PLANT
CAPACITY, which is the mentor's metric and what every other
signal in this project was tuned against - not MAPE, which
explodes near sunrise and sunset where the denominator goes
to zero.

Only blocks with a REAL measurement are scored. Interpolated
or filled meter rows are excluded, because scoring a forecast
against a number that was itself interpolated measures
nothing.

Run:  python -m tests.test_llm_approach_score
      python -m tests.test_llm_approach_score --run 2026-07-09_14-15
=========================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.preprocessing.windy_features import (
    epoch_seconds, load_meter_history
)


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]


def deviation_pct(forecast_mw, actual_mw):
    """
    Mean absolute error as a percentage of plant capacity.
    """

    return float(
        np.mean(np.abs(np.asarray(forecast_mw) - np.asarray(actual_mw)))
        / CAPACITY_MW * 100
    )


def load_schedules(folder, wanted=None):

    folder = Path(folder)

    if not folder.exists():
        return []

    paths = sorted(folder.glob("*_schedule.csv"))

    if wanted:
        paths = [p for p in paths if wanted in p.name]

    return paths


def score_one(path, meter, clearsky):
    """
    One saved schedule against actual. Returns None when none of its
    blocks have a real measurement yet.
    """

    schedule = pd.read_csv(path, parse_dates=["timestamp"])

    if schedule["timestamp"].dt.tz is None:
        schedule["timestamp"] = schedule["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        schedule["timestamp"] = schedule["timestamp"].dt.tz_convert(TIMEZONE)

    merged = schedule.merge(meter, on="timestamp", how="inner")

    if "is_real_measurement" in merged.columns:
        merged = merged[merged["is_real_measurement"].fillna(False)]

    merged = merged[merged["actual_mw"].notna()]

    if merged.empty:
        return None

    run_time = merged["timestamp"].min() - pd.Timedelta(minutes=15)

    # --- baseline 1: hold the last measured kt flat ---
    history = meter[
        (meter["timestamp"] < merged["timestamp"].min())
        & (meter["timestamp"].dt.date == merged["timestamp"].iloc[0].date())
        & meter["actual_mw"].notna()
    ].tail(4)

    if history.empty:
        persistence_kt = 1.0
    else:
        expected = clearsky.estimate_clearsky_generation(
            history["timestamp"]
        )["expected_power_kw"].to_numpy() / 1000

        ratio = np.divide(
            history["actual_mw"].to_numpy(), expected,
            out=np.full(len(history), np.nan), where=expected > 0.02,
        )
        persistence_kt = float(np.nanmean(ratio)) if np.isfinite(
            np.nanmean(ratio)
        ) else 1.0

    clearsky_mw = merged["clearsky_power_mw"].to_numpy()

    return {
        "run": path.name.replace("_schedule.csv", ""),
        "blocks": len(merged),
        "llm_pct": deviation_pct(merged["forecast_mw"], merged["actual_mw"]),
        "persistence_pct": deviation_pct(
            np.clip(persistence_kt * clearsky_mw, 0, CAPACITY_MW),
            merged["actual_mw"],
        ),
        "clearsky_pct": deviation_pct(clearsky_mw, merged["actual_mw"]),
        "persistence_kt": round(persistence_kt, 3),
        "actual_mwh": float(merged["actual_mw"].sum() * 0.25),
        "llm_mwh": float(merged["forecast_mw"].sum() * 0.25),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Score LLM schedules against actual generation"
    )
    parser.add_argument("--run", default=None,
                        help="score only schedules whose name contains this")
    parser.add_argument("--folder",
                        default=settings.get("fusion", {}).get(
                            "output_dir", "outputs/llm_schedules"))

    args = parser.parse_args()

    paths = load_schedules(args.folder, args.run)

    if not paths:
        print(f"No saved schedules in {args.folder}")
        return 0

    meter = load_meter_history()

    if meter is None or meter.empty:
        print("No meter history available - cannot score anything.")
        return 1

    meter = meter.copy()

    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        meter["timestamp"] = meter["timestamp"].dt.tz_convert(TIMEZONE)

    meter["actual_mw"] = meter["active_power_kw"] / 1000.0

    columns = ["timestamp", "actual_mw"]
    if "is_real_measurement" in meter.columns:
        columns.append("is_real_measurement")

    meter = meter[columns]

    clearsky = ClearSkyModel()

    rows = []

    for path in paths:

        result = score_one(path, meter, clearsky)

        if result is None:
            print(f"{path.name}: no real measured blocks yet - skipped")
            continue

        rows.append(result)

    if not rows:
        print("\nNothing could be scored. The saved schedules cover blocks "
              "with no real meter data.")
        return 0

    results = pd.DataFrame(rows)

    print("=" * 78)
    print("LLM SCHEDULE vs ACTUAL GENERATION")
    print("  deviation = mean absolute error as % of plant capacity, "
          "lower is better")
    print("=" * 78)

    print(results[[
        "run", "blocks", "llm_pct", "persistence_pct", "clearsky_pct"
    ]].to_string(index=False, float_format="%.2f"))

    print("-" * 78)
    print(f"{'MEAN':<28} {results['llm_pct'].mean():8.2f} "
          f"{results['persistence_pct'].mean():14.2f} "
          f"{results['clearsky_pct'].mean():13.2f}")

    beat = int((results["llm_pct"] < results["persistence_pct"]).sum())

    print()
    print(f"LLM beat plain persistence on {beat} of {len(results)} run(s).")

    if beat <= len(results) / 2:
        print("VERDICT: not yet earning its complexity. A flat hold of the "
              "last measured\n         cloudiness did as well or better.")
    else:
        print("VERDICT: better than persistence on this sample. Sample is "
              "small - do not\n         ship on it; rerun as more days land.")

    output = Path("outputs/reports/llm_approach_score.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
