"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Score the way the mentor scores
=========================================================
Stitches each day's runs into ONE schedule the way the
production desk really builds it, then grades that against
actual generation and prices the DSM penalty.

WHY THIS REPLACES tests/test_llm_approach_score.py
--------------------------------------------------
That script graded every run's whole forward forecast to
19:00. The mentor's "Schedule Generation Workflow for Model
Evaluation" says the opposite:

    "The schedule values for the blocks that have already
     passed will remain unchanged. Only the remaining future
     blocks will be updated."

So the 09:45 run's opinion about 18:00 is overwritten five
times before evening and never reaches the grid. Grading it
punished the model for numbers it would never have
published - which is why the morning runs looked terrible
(7-20%) while the afternoon ones looked strong (1.5-4%).

Here each run only owns the blocks it would really have
owned, under this plant's freeze horizon from the mentor's
"Simple Effective Time Schedule Guide":

    Sirmour               6 blocks (90 min)
    Kothagudem / Kasipet  3 blocks (45 min)

    generated 11:15 -> engine block 46 -> freeze 46-51
                    -> new schedule effective from block 52

The first run of a day has nothing to freeze, so it writes
from its own engine block.

WHAT IT REPORTS, IN THE MENTOR'S ORDER
--------------------------------------
The "AI Schedule Accuracy Assessment" ranks models by, in
order: lowest scheduling penalty, highest accuracy, lowest
deviation, then consistency across days. Deviation-% was
what we had been reporting; it is third on that list. The
headline here is the penalty, in rupees.

DSM slabs, matching tests/build_penalty_report.py so the two
never disagree:

    0-10%   free
    10-15%  Rs 0.50 / kWh
    15-20%  Rs 0.75 / kWh
    20%+    Rs 1.00 / kWh

deviation % is measured against plant capacity, and one
block of 1 MW deviation is 250 kWh (0.25 h x 1000).

THE BASELINE IS STITCHED THE SAME WAY
-------------------------------------
Every saved schedule carries `anchor_mw` - damped
persistence, the meter and a clock, no AI and no Windy.
Stitching that identically gives a like-for-like "what if we
had not bothered" number on exactly the same blocks.

Run:  python -m tests.score_day_schedules
      python -m tests.score_day_schedules --folder outputs/llm_schedules_noweather
=========================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
# Aliased: this module's own `dsm_penalty()` function is the name every
# other script in this pipeline imports, and it must keep that name.
from modules.evaluation import dsm_penalty as dsm_penalty_module
from modules.preprocessing.windy_features import load_meter_history
from modules.scheduling.effective_time import block_number


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]
INTERVAL = settings["forecast"]["interval_minutes"]

FREEZE_BLOCKS = settings.get("schedule_rules", {}).get("freeze_blocks", 0)

BLOCK_ENERGY_FACTOR = dsm_penalty_module.BLOCK_ENERGY_FACTOR  # MW dev -> kWh

# The mentor's band table, from config (dsm.bands). Kept as a
# module-level name because tests/build_schedule_vs_meter_xlsx.py imports
# SLABS from here to print the slab table in its sheet.
SLABS = dsm_penalty_module.BANDS

_RUN_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})")


def dsm_penalty(deviation_mw):
    """
    Rupees for one block, by DSM band. `deviation_mw` is
    (actual - scheduled); only its size matters, not its sign.

    Every script in this pipeline prices a block through here, and since
    2026-08-14 this is a one-line call into
    modules/evaluation/dsm_penalty.py - the mentor's SIRMOUR penalty
    logic, shared with the old pipeline's penalty report. The arithmetic
    is unchanged (the same band-share-of-energy rule, checked against his
    worked example in tests/test_dsm_penalty.py); what changed is that
    the two pipelines can no longer drift apart on the band table.
    """

    return dsm_penalty_module.penalty_rs(deviation_mw)


def run_time_of(path):

    match = _RUN_STAMP.search(path.name)

    if not match:
        return None

    date, hour, minute = match.groups()

    return pd.Timestamp(f"{date} {hour}:{minute}", tz=TIMEZONE)


def stitch_day_with_runs(paths, column="forecast_mw"):
    """
    One day's runs -> {timestamp: (value, "HH:MM" of the run that wrote it)},
    applying the freeze horizon.

    Runs are applied oldest first. Each writes only from its EFFECTIVE
    START block onward - engine block plus the freeze horizon - so the
    blocks it was not allowed to touch keep whatever the earlier run
    put there. The first run of the day has nothing to freeze and
    writes from its own engine block.

    The run label is what the penalty report shows as "Scheduled at": it
    is the whole point of the freeze horizon that a block's value belongs
    to an earlier run than the one nearest it, so the report has to be
    able to say which.
    """

    day = {}

    for index, path in enumerate(sorted(paths, key=lambda p: run_time_of(p))):

        frame = pd.read_csv(path, parse_dates=["timestamp"])

        if column not in frame.columns or frame.empty:
            continue

        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert(TIMEZONE)

        run_time = run_time_of(path)
        engine = block_number(run_time)

        effective = engine if index == 0 else engine + max(FREEZE_BLOCKS, 1)

        for _, row in frame.iterrows():

            if block_number(row["timestamp"]) >= effective:
                day[row["timestamp"]] = (
                    float(row[column]), f"{run_time:%H:%M}"
                )

    return day


def stitch_day(paths, column="forecast_mw"):
    """One day's runs -> {timestamp: value}, under the freeze horizon."""

    return {
        timestamp: value
        for timestamp, (value, _) in stitch_day_with_runs(paths, column).items()
    }


def score_day(day_values, meter):

    if not day_values:
        return None

    frame = pd.DataFrame({
        "timestamp": list(day_values),
        "scheduled_mw": list(day_values.values()),
    }).sort_values("timestamp")

    merged = frame.merge(meter, on="timestamp", how="inner").dropna(
        subset=["actual_mw"]
    )

    if merged.empty:
        return None

    merged["deviation_mw"] = merged["actual_mw"] - merged["scheduled_mw"]
    merged["penalty_rs"] = merged["deviation_mw"].map(dsm_penalty)

    deviation_pct = (
        merged["deviation_mw"].abs().mean() / CAPACITY_MW * 100
    )

    # "Accuracy" as the assessment document uses it: the share of blocks
    # inside the free 10% band, i.e. blocks that cost nothing.
    within_band = float(
        (merged["deviation_mw"].abs() / CAPACITY_MW * 100 <= 10).mean() * 100
    )

    return {
        "blocks": len(merged),
        "deviation_pct": float(deviation_pct),
        "accuracy_pct": within_band,
        "penalty_rs": float(merged["penalty_rs"].sum()),
        "scheduled_mwh": float(merged["scheduled_mw"].sum() * 0.25),
        "actual_mwh": float(merged["actual_mw"].sum() * 0.25),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Score stitched day schedules the mentor's way"
    )
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )

    args = parser.parse_args()

    folder = Path(args.folder)

    paths = sorted(folder.glob("*_schedule.csv"))

    if not paths:
        raise SystemExit(f"No schedules in {folder}")

    meter = load_meter_history()

    if meter is None or meter.empty:
        raise SystemExit("No meter history to score against.")

    meter = meter.copy()

    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        meter["timestamp"] = meter["timestamp"].dt.tz_convert(TIMEZONE)

    if "is_real_measurement" in meter.columns:
        meter = meter[meter["is_real_measurement"].fillna(False)]

    meter["actual_mw"] = meter["active_power_kw"] / 1000.0
    meter = meter[["timestamp", "actual_mw"]].dropna()

    by_day = {}

    for path in paths:

        run_time = run_time_of(path)

        if run_time is None:
            continue

        by_day.setdefault(run_time.date(), []).append(path)

    rows = []

    for day in sorted(by_day):

        model = score_day(stitch_day(by_day[day], "forecast_mw"), meter)
        baseline = score_day(stitch_day(by_day[day], "anchor_mw"), meter)

        if model is None:
            continue

        rows.append({
            "day": day,
            "runs": len(by_day[day]),
            "blocks": model["blocks"],
            "dev_pct": model["deviation_pct"],
            "accuracy_pct": model["accuracy_pct"],
            "penalty_rs": model["penalty_rs"],
            "base_dev_pct": baseline["deviation_pct"] if baseline else np.nan,
            "base_penalty_rs": baseline["penalty_rs"] if baseline else np.nan,
        })

    if not rows:
        raise SystemExit("Nothing could be scored.")

    results = pd.DataFrame(rows)

    print("=" * 92)
    print("DAY SCHEDULES, SCORED THE MENTOR'S WAY")
    print(f"  runs stitched under a {FREEZE_BLOCKS}-block freeze horizon; "
          "only blocks a run really owned are graded")
    print("  penalty = DSM slabs (free to 10%, then Rs 0.50 / 0.75 / 1.00 per kWh)")
    print("  base_* = damped persistence, stitched identically. No AI, no Windy.")
    print("=" * 92)

    print(results.to_string(index=False, float_format="%.2f"))

    print("-" * 92)
    print(f"days              : {len(results)}")
    print(f"mean deviation    : {results['dev_pct'].mean():.2f}%   "
          f"(baseline {results['base_dev_pct'].mean():.2f}%)")
    print(f"mean accuracy     : {results['accuracy_pct'].mean():.1f}% of blocks "
          "inside the free 10% band")
    print(f"TOTAL PENALTY     : Rs {results['penalty_rs'].sum():,.0f}   "
          f"(baseline Rs {results['base_penalty_rs'].sum():,.0f})")
    print(f"mean per day      : Rs {results['penalty_rs'].mean():,.0f}   "
          f"(baseline Rs {results['base_penalty_rs'].mean():,.0f})")

    cheaper = int((results["penalty_rs"] < results["base_penalty_rs"]).sum())

    print(f"\ncheaper than the baseline on {cheaper} of {len(results)} days")

    saved = results["base_penalty_rs"].sum() - results["penalty_rs"].sum()

    if saved > 0:
        print(f"VERDICT: the model saved Rs {saved:,.0f} over {len(results)} days.")
    else:
        print(f"VERDICT: the model COST Rs {abs(saved):,.0f} more than doing "
              "nothing clever.")

    output = Path("outputs/reports/day_schedule_scores.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
