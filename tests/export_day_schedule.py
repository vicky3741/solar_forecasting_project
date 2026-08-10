"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Export one day's final schedule for comparison
=========================================================
Writes the stitched day schedule in the format the reports
use - MW only, one row per block, with a per-block deviation
column - so it can be put beside another pipeline's schedule
for the same day without any reformatting.

The schedule is stitched the mentor's way: each run owns only
the blocks it really owned under this plant's freeze horizon,
so this is what the grid operator would actually have been
holding at the end of the day, not any single run's forecast.

Run:  python -m tests.export_day_schedule --day 2026-08-10
=========================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from modules.scheduling.effective_time import block_number
from tests.score_day_schedules import (
    FREEZE_BLOCKS, dsm_penalty, run_time_of, stitch_day
)


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]


def main():

    parser = argparse.ArgumentParser(
        description="Export one day's stitched schedule beside actual"
    )
    parser.add_argument("--day", required=True)
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    day = pd.Timestamp(args.day).date()

    paths = [
        p for p in sorted(Path(args.folder).glob("*_schedule.csv"))
        if run_time_of(p) is not None and run_time_of(p).date() == day
    ]

    if not paths:
        raise SystemExit(f"No saved schedules for {day} in {args.folder}")

    model = stitch_day(paths, "forecast_mw")
    anchor = stitch_day(paths, "anchor_mw")

    frame = pd.DataFrame({
        "timestamp": list(model),
        "scheduled_mw": list(model.values()),
    }).sort_values("timestamp")

    frame["anchor_mw"] = frame["timestamp"].map(anchor)
    frame["block"] = frame["timestamp"].map(block_number)
    frame["time"] = frame["timestamp"].dt.strftime("%H:%M")

    meter = load_meter_history()

    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(TIMEZONE)

    if "is_real_measurement" in meter.columns:
        meter = meter[meter["is_real_measurement"].fillna(False)]

    meter = meter.assign(actual_mw=meter["active_power_kw"] / 1000.0)

    frame = frame.merge(
        meter[["timestamp", "actual_mw"]], on="timestamp", how="left"
    )

    frame["deviation_mw"] = frame["actual_mw"] - frame["scheduled_mw"]
    frame["deviation_pct_of_capacity"] = (
        frame["deviation_mw"].abs() / CAPACITY_MW * 100
    )
    frame["penalty_rs"] = frame["deviation_mw"].map(
        lambda d: dsm_penalty(d) if pd.notna(d) else np.nan
    )

    columns = [
        "block", "time", "scheduled_mw", "actual_mw", "deviation_mw",
        "deviation_pct_of_capacity", "penalty_rs", "anchor_mw",
    ]

    frame = frame[columns].round(4)

    out = Path(
        args.out or
        Path.home() / "Downloads" /
        f"{settings['plant']['code']}_{day}_new_pipeline_schedule.csv"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    scored = frame.dropna(subset=["actual_mw"])

    print("=" * 74)
    print(f"FINAL SCHEDULE — {day}")
    print(f"  stitched from {len(paths)} run(s) under a "
          f"{FREEZE_BLOCKS}-block freeze horizon")
    print("=" * 74)

    print(frame.to_string(index=False, float_format="%.3f"))

    print("-" * 74)
    print(f"blocks scheduled : {len(frame)}")
    print(f"blocks scored    : {len(scored)}")
    print(f"scheduled energy : {frame['scheduled_mw'].sum() * 0.25:.3f} MWh")
    print(f"actual energy    : {scored['actual_mw'].sum() * 0.25:.3f} MWh")
    print(f"mean deviation   : "
          f"{scored['deviation_pct_of_capacity'].mean():.2f}% of capacity")
    print(f"blocks in free band (<=10%) : "
          f"{(scored['deviation_pct_of_capacity'] <= 10).mean() * 100:.1f}%")
    print(f"DSM penalty      : Rs {scored['penalty_rs'].sum():,.2f}")

    print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
