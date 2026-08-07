"""
=========================================================
Solar Forecasting Project
Per-Plant Clear-Sky Calibration
=========================================================
Sweeps `clearsky.performance_ratio` for the plant SOLAR_PLANT
names, over a range wide enough to find an optimum rather
than run into the edge of a grid.

WHY THIS EXISTS
---------------
The performance ratio is the one tuned constant that is
genuinely PHYSICAL rather than statistical: it is how much of
the theoretical clear-sky irradiance this particular site
converts into metered AC power, and it differs per plant with
module technology, inverter sizing, soiling, DC:AC ratio and
losses. Sirmour's 0.80 was tuned on Sirmour.

A new plant inherits that 0.80 and, if its real ratio is
higher, under-forecasts every clear block of every day - a
bias no amount of blend-weight tuning can fix, because the
ceiling itself is set too low. Kasipet's and Bhupalpally's
first reconstructions came out 6% and 15% short on daily
energy, which is exactly that signature.

The default grid in tests/test_backtest.py stops at 0.85, so
a plant whose optimum is above it silently "wins" at the
boundary and looks tuned when it is not. This sweeps past it.

Only the performance ratio moves. chronos_weight and the
trend halflife are left at whatever the plant currently has,
because those were validated out-of-sample on Sirmour and one
in-sample grid is not evidence enough to overturn them.

Run:
    SOLAR_PLANT=kasipet python -m tests.tune_plant_clearsky
    SOLAR_PLANT=kasipet python -m tests.tune_plant_clearsky --min 0.7 --max 1.05
=========================================================
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config.config import settings
from modules.evaluation.backtester import Backtester
from utils.file_manager import processed_data_path, reports_path


def parse_args():

    parser = argparse.ArgumentParser(
        description="Sweep clearsky.performance_ratio for this plant."
    )
    parser.add_argument("--min", type=float, default=0.65)
    parser.add_argument("--max", type=float, default=1.05)
    parser.add_argument("--step", type=float, default=0.025)
    return parser.parse_args()


def main():

    args = parse_args()

    plant = settings["plant"]
    current = settings["clearsky"]["performance_ratio"]

    processed_file = processed_data_path()

    if not processed_file.exists():
        print(
            f"{processed_file} does not exist yet. Build it first:\n"
            f"    SOLAR_PLANT={plant['key']} python -m tests.test_preprocessing"
        )
        return 1

    dataframe = pd.read_csv(processed_file, parse_dates=["timestamp"])

    print("=" * 70)
    print(f"CLEAR-SKY CALIBRATION - {plant['name']} ({plant['capacity_mw']} MW)")
    print("=" * 70)
    print(f"current performance_ratio : {current}")
    print(f"sweeping                  : {args.min} to {args.max} "
          f"step {args.step}")
    print()
    print("Running the backtest once to cache every run's raw signals "
          "(no Chronos re-runs during the sweep)...")

    backtester = Backtester()
    _, cache, _ = backtester.run(dataframe)

    print(f"{len(cache)} runs cached.\n")

    candidates = [
        round(float(value), 4)
        for value in np.arange(args.min, args.max + 1e-9, args.step)
    ]

    grid, best = backtester.tune(
        cache,
        chronos_weight_candidates=[settings["hybrid_blend"]["chronos_weight"]],
        performance_ratio_candidates=candidates,
        trend_halflife_candidates=[
            settings["hybrid_blend"].get("trend_halflife_min", 0)
        ],
    )

    grid = grid.sort_values("performance_ratio").reset_index(drop=True)

    print("performance_ratio -> average % deviation")
    for _, row in grid.iterrows():
        marker = ""
        if row["performance_ratio"] == best["performance_ratio"]:
            marker = "   <-- best"
        if abs(row["performance_ratio"] - current) < 1e-9:
            marker += "   (current)"
        print(f"   {row['performance_ratio']:.3f}   "
              f"{row['avg_deviation_pct']:7.3f}{marker}")

    at_current = grid.loc[
        (grid["performance_ratio"] - current).abs().idxmin(),
        "avg_deviation_pct",
    ]

    print()
    print("=" * 70)
    print(f"best performance_ratio : {best['performance_ratio']:.3f}")
    print(f"deviation there        : {best['avg_deviation_pct']:.3f} %")
    print(f"deviation at current   : {at_current:.3f} %  "
          f"(pr {current})")
    print(f"improvement            : {at_current - best['avg_deviation_pct']:+.3f} "
          "pct points")

    edge = (
        best["performance_ratio"] <= candidates[0] + 1e-9
        or best["performance_ratio"] >= candidates[-1] - 1e-9
    )

    if edge:
        print()
        print("WARNING: the optimum sits at the EDGE of the swept range, so it "
              "is not really an optimum. Re-run with a wider --min/--max "
              "before trusting it.")

    output_path = reports_path("clearsky_calibration.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path}")
    print("This is an IN-SAMPLE fit over every available day. Set it in this "
          "plant's overlay under clearsky.performance_ratio, then re-run the "
          "day-schedule backfill so the reports reflect it.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
