"""
=========================================================
Solar Forecasting Project
Old pipeline vs new LLM pipeline, on identical actuals
=========================================================
The old pipeline's schedules already exist as workbooks in
the Reports folder, one per day, with the actual meter
reading beside each block. The new pipeline's schedules are
stitched out of outputs/llm_schedules the same way a live
day is stitched - oldest run first, each run writing only
from its effective block onward.

HOW THIS AVOIDS FLATTERING EITHER SIDE
--------------------------------------
  * Both are scored by the SAME dsm_penalty(), imported from
    score_day_schedules - not by whatever each workbook
    happened to compute.
  * Both are scored on the SAME blocks: the intersection of
    what both pipelines actually scheduled. A pipeline cannot
    win by declining to schedule its hard blocks.
  * Actuals are taken from the meter history and CROSS-CHECKED
    against the actual column in the old workbook. If the two
    disagree anywhere, that is reported and the day is flagged,
    because then the two are not being graded on the same day.
  * The old workbook's own summary rows are ignored and every
    figure is recomputed from its block columns. Those summary
    rows are not reliable - in
    Schedule_vs_Meter_Penalty_2026-08-10.xlsx they report the
    Enercast column's totals under AI-schedule labels.

Enercast is read only so it can be shown as a reference line.
It is not an input to either pipeline and is not scored as a
competitor.

Run:  python -m tests.compare_pipelines
      python -m tests.compare_pipelines --from 2026-08-01 --to 2026-08-05
=========================================================
"""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from tests.score_day_schedules import (
    CAPACITY_MW, TIMEZONE, dsm_penalty, run_time_of, stitch_day
)


REPORTS_DIR = Path(r"C:\Users\Acer\OneDrive\Desktop\Reports")

# The old pipeline's workbooks are named two different ways across the
# period, so both are tried before a day is called missing.
OLD_PATTERNS = [
    "Team2_Gemini_3.5_flash_Sirmour_{day}.xlsx",
    "Schedule_vs_Meter_Penalty_{day}.xlsx",
]

SHEET = "Schedule vs Meter + Penalty"


def old_day(day):
    """
    One day of the old pipeline, straight from its workbook: block,
    time, scheduled MW, the actual it was graded against, and Enercast
    for reference. Summary rows are deliberately not read.
    """

    for pattern in OLD_PATTERNS:

        path = REPORTS_DIR / pattern.format(day=f"{day:%Y-%m-%d}")

        if path.exists():
            break
    else:
        return None, None

    sheet = openpyxl.load_workbook(path, data_only=True)[SHEET]

    rows = []

    for index in range(5, sheet.max_row + 1):

        block = sheet.cell(index, 1).value

        # The block column turns non-numeric at the summary section.
        if not isinstance(block, (int, float)):
            break

        rows.append({
            "block": int(block),
            "window": sheet.cell(index, 2).value,
            "old_scheduled_mw": sheet.cell(index, 3).value,
            "workbook_actual_mw": sheet.cell(index, 4).value,
            "scheduled_at": sheet.cell(index, 8).value,
            "enercast_mw": sheet.cell(index, 9).value,
        })

    return pd.DataFrame(rows), path


def new_day(day, folder):
    """One day of the new pipeline, stitched from its 7 runs."""

    paths = [
        path for path in Path(folder).glob(f"*{day:%Y-%m-%d}*_schedule.csv")
        if run_time_of(path) is not None
    ]

    if not paths:
        return None, 0

    values = stitch_day(paths)

    if not values:
        return None, len(paths)

    frame = pd.DataFrame({
        "timestamp": list(values),
        "new_scheduled_mw": list(values.values()),
    }).sort_values("timestamp")

    # The anchor stitched the same way. Damped persistence off the meter
    # and a clock - no AI, no Windy, no weather. It is the "what if we
    # had not bothered" column, and without it a comparison between two
    # clever pipelines cannot say whether either is earning its keep.
    anchor = stitch_day(paths, column="anchor_mw")

    if anchor:
        frame["anchor_mw"] = frame["timestamp"].map(anchor)

    # Match the workbook on block number, which is how the old side is
    # indexed. Block 1 is the 00:00-00:15 window.
    frame["block"] = [
        stamp.hour * 4 + stamp.minute // 15 + 1 for stamp in frame["timestamp"]
    ]

    return frame, len(paths)


def meter_day(meter, day):
    """Actual generation for one day, in MW, indexed by block."""

    frame = meter.copy()

    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(TIMEZONE)

    frame = frame[frame["timestamp"].dt.date == day]

    if frame.empty:
        return pd.DataFrame(columns=["block", "actual_mw"])

    frame["actual_mw"] = frame["active_power_kw"] / 1000.0

    frame["block"] = [
        stamp.hour * 4 + stamp.minute // 15 + 1 for stamp in frame["timestamp"]
    ]

    return frame[["block", "actual_mw"]]


def score(frame, column):
    """Penalty and accuracy for one schedule column."""

    deviation = frame["actual_mw"] - frame[column]
    deviation_pct = deviation.abs() / CAPACITY_MW * 100

    penalty = deviation.map(dsm_penalty)

    return {
        "blocks": int(len(frame)),
        "penalty_rs": float(penalty.sum()),
        "deviation_pct": float(deviation_pct.mean()),
        "in_band_pct": float((deviation_pct <= 10).mean() * 100),
        "penalised": int((deviation_pct > 10).sum()),
        "worst_rs": float(penalty.max()) if len(penalty) else 0.0,
        "scheduled_mwh": float(frame[column].sum() * 0.25),
        "actual_mwh": float(frame["actual_mw"].sum() * 0.25),
    }


def compare_day(day, meter, folder):

    old, old_path = old_day(day)

    if old is None:
        return None, f"no old-pipeline workbook for {day}"

    new, run_count = new_day(day, folder)

    if new is None:
        return None, f"no new-pipeline schedule for {day} ({run_count} run file(s))"

    actual = meter_day(meter, day)

    if actual.empty:
        return None, f"no meter data for {day}"

    new_columns = ["block", "new_scheduled_mw"]

    if "anchor_mw" in new.columns:
        new_columns.append("anchor_mw")

    merged = (
        old.merge(new[new_columns], on="block", how="inner")
           .merge(actual, on="block", how="inner")
    )

    merged = merged.dropna(
        subset=["old_scheduled_mw", "new_scheduled_mw", "actual_mw"]
    )

    if merged.empty:
        return None, f"no overlapping scored blocks for {day}"

    # Do the two sources agree about what actually happened? If they do
    # not, the comparison is meaningless and must not be quietly shown.
    gap = (
        pd.to_numeric(merged["workbook_actual_mw"], errors="coerce")
        - merged["actual_mw"]
    ).abs()

    disagreement = float(gap.max()) if gap.notna().any() else float("nan")

    result = {
        "day": day,
        "blocks": int(len(merged)),
        "runs": run_count,
        "actual_disagreement_mw": disagreement,
        "old": score(merged, "old_scheduled_mw"),
        "new": score(merged, "new_scheduled_mw"),
        "workbook": old_path.name,
        "frame": merged,
    }

    if "anchor_mw" in merged.columns and merged["anchor_mw"].notna().any():
        anchored = merged.dropna(subset=["anchor_mw"])
        if not anchored.empty:
            result["anchor"] = score(anchored, "anchor_mw")

    if "enercast_mw" in merged.columns:
        reference = merged.dropna(subset=["enercast_mw"])
        if not reference.empty:
            result["enercast"] = score(reference, "enercast_mw")

    return result, None


def main():

    parser = argparse.ArgumentParser(
        description="Old pipeline vs new LLM pipeline on identical actuals"
    )
    parser.add_argument("--from", dest="start", default="2026-08-01")
    parser.add_argument("--to", dest="end", default="2026-08-05")
    parser.add_argument("--folder", default="outputs/llm_schedules")
    parser.add_argument("--csv", default="")

    args = parser.parse_args()

    meter = load_meter_history()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()

    print("=" * 78)
    print(f"OLD PIPELINE vs NEW PIPELINE   {start} to {end}")
    print(f"schedules from {args.folder}")
    print("=" * 78)

    results = []

    day = start

    while day <= end:

        result, problem = compare_day(day, meter, args.folder)

        if problem:
            print(f"  {day}  SKIPPED - {problem}")
        else:
            results.append(result)

        day += timedelta(days=1)

    if not results:
        raise SystemExit("Nothing to compare.")

    worst_gap = max(r["actual_disagreement_mw"] for r in results)

    print(f"\nActuals cross-check: the largest disagreement between the meter "
          f"history\nand the old workbook's actual column, over all days, is "
          f"{worst_gap:.4f} MW.")

    if worst_gap > 0.01:
        print("  WARNING: the two sides are NOT being graded on the same "
              "actuals. Investigate before quoting any figure below.")

    print("\n" + "-" * 78)
    print(f"{'day':<12} {'blk':>4} {'OLD Rs':>10} {'NEW Rs':>10} "
          f"{'ANCHOR Rs':>11} {'OLD dev%':>9} {'NEW dev%':>9} {'NEW band':>9}")
    print("-" * 78)

    for result in results:

        old, new = result["old"], result["new"]

        anchor = result.get("anchor")
        anchor_text = (
            f"{anchor['penalty_rs']:>11.2f}" if anchor else f"{'-':>11}"
        )

        print(f"{str(result['day']):<12} {result['blocks']:>4} "
              f"{old['penalty_rs']:>10.2f} {new['penalty_rs']:>10.2f} "
              f"{anchor_text} "
              f"{old['deviation_pct']:>8.2f}% {new['deviation_pct']:>8.2f}% "
              f"{new['in_band_pct']:>8.1f}%")

    print("-" * 78)

    old_total = sum(r["old"]["penalty_rs"] for r in results)
    new_total = sum(r["new"]["penalty_rs"] for r in results)
    blocks = sum(r["blocks"] for r in results)

    old_dev = float(np.mean([r["old"]["deviation_pct"] for r in results]))
    new_dev = float(np.mean([r["new"]["deviation_pct"] for r in results]))

    old_band = float(np.mean([r["old"]["in_band_pct"] for r in results]))
    new_band = float(np.mean([r["new"]["in_band_pct"] for r in results]))

    anchor_days = [r["anchor"] for r in results if "anchor" in r]
    anchor_total = sum(a["penalty_rs"] for a in anchor_days)

    anchor_text = (
        f"{anchor_total:>11.2f}"
        if len(anchor_days) == len(results) else f"{'-':>11}"
    )

    print(f"{'TOTAL':<12} {blocks:>4} {old_total:>10.2f} {new_total:>10.2f} "
          f"{anchor_text} "
          f"{old_dev:>8.2f}% {new_dev:>8.2f}% {new_band:>8.1f}%")
    print(f"{'(old band':<12} {'':>4} {old_band:>9.1f}%)")

    print("\n" + "=" * 78)

    change = new_total - old_total

    if abs(old_total) > 1e-9:
        percent = change / old_total * 100
        direction = "CHEAPER" if change < 0 else "MORE EXPENSIVE"
        print(f"Over {len(results)} day(s) and {blocks} scored block(s), the "
              f"new pipeline is\nRs {abs(change):,.2f} {direction} "
              f"({abs(percent):.1f}%).")

    wins = sum(
        1 for r in results if r["new"]["penalty_rs"] < r["old"]["penalty_rs"]
    )

    print(f"Days the new pipeline was cheaper: {wins} of {len(results)}.")

    if len(anchor_days) == len(results):

        print(f"\nAnchor only - damped persistence off the meter, no AI, no "
              f"Windy, no weather:\n  Rs {anchor_total:,.2f}. "
              f"The new pipeline is "
              f"Rs {abs(new_total - anchor_total):,.2f} "
              f"{'CHEAPER' if new_total < anchor_total else 'MORE EXPENSIVE'} "
              f"than doing nothing clever;\n  the old pipeline is "
              f"Rs {abs(old_total - anchor_total):,.2f} "
              f"{'CHEAPER' if old_total < anchor_total else 'MORE EXPENSIVE'}"
              f" than it.")

    if "enercast" in results[0]:
        reference = sum(
            r["enercast"]["penalty_rs"] for r in results if "enercast" in r
        )
        print(f"\nEnercast over the same days: Rs {reference:,.2f} "
              f"(reference only - not an input to either pipeline).")

    if args.csv:

        rows = []

        for result in results:
            frame = result["frame"].copy()
            frame.insert(0, "day", result["day"])
            rows.append(frame)

        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(out, index=False)

        print(f"\nBlock-level detail written to {out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
