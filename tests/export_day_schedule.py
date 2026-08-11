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

To put this pipeline's day through the SAME DSM penalty report the old
pipeline's day goes through - same slabs, same moving penalty band, same
three charts - write the schedule in that report's own input layout and
point the report at it:

  python -m tests.export_day_schedule --day 2026-08-10 \
      --schedule-csv <main worktree>/outputs/schedules/day_schedule_2026-08-10.csv
  python -m tests.build_penalty_report 2026-08-10 --tag NewPipeline
  python -m tests.recalc outputs/reports/Schedule_vs_Meter_Penalty_2026-08-10_NewPipeline.xlsx

The --tag keeps the two pipelines' reports for one day from overwriting
each other, and labels the sheet so they cannot be mixed up.
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
    FREEZE_BLOCKS, dsm_penalty, run_time_of, stitch_day, stitch_day_with_runs
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
    parser.add_argument(
        "--schedule-csv", dest="schedule_csv", default=None,
        help="also write the stitched schedule in the day_schedule_<day>.csv "
             "layout tests/build_penalty_report.py reads, so this pipeline's "
             "day can be put through the same DSM penalty report, penalty "
             "bands and charts the old pipeline's day goes through",
    )

    args = parser.parse_args()

    day = pd.Timestamp(args.day).date()

    paths = [
        p for p in sorted(Path(args.folder).glob("*_schedule.csv"))
        if run_time_of(p) is not None and run_time_of(p).date() == day
    ]

    if not paths:
        raise SystemExit(f"No saved schedules for {day} in {args.folder}")

    written_by = stitch_day_with_runs(paths, "forecast_mw")
    model = {timestamp: value for timestamp, (value, _) in written_by.items()}
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

    if args.schedule_csv:

        # The penalty report merges on a NAIVE local timestamp, which is
        # what the meter preprocessor produces; handing it a tz-aware one
        # silently matches nothing and every block loses its actual.
        schedule = pd.DataFrame({
            "block": [block_number(t) for t in sorted(model)],
            "block_time": [t.strftime("%H:%M") for t in sorted(model)],
            "timestamp": [t.tz_localize(None) for t in sorted(model)],
            "scheduled_at": [written_by[t][1] for t in sorted(model)],
            "scheduled_mw": [round(model[t], 4) for t in sorted(model)],
            "anchor_mw": [round(anchor[t], 4) if t in anchor else None
                          for t in sorted(model)],
        })

        schedule_path = Path(args.schedule_csv)
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(schedule_path, index=False)

        print(f"\nSchedule for the penalty report: {schedule_path}")

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
