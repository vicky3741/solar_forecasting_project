"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Schedule vs Meter + Penalty workbook
=========================================================
Writes the new pipeline's day schedule in the SAME layout as
the production pipeline's Schedule_vs_Meter_Penalty workbook,
so the two can be opened side by side and compared row for
row without reformatting either.

Same columns, same summary rows, same DSM slab table at the
bottom, same block numbering.

A NOTE ON THE PRODUCTION WORKBOOK'S SUMMARY

In Schedule_vs_Meter_Penalty_2026-08-10.xlsx the summary rows
labelled "Blocks that incurred a penalty", "Worst single-block
penalty" and "TOTAL DSM PENALTY FOR THE DAY" carry the
ENERCAST column's figures, not the AI schedule's:

    sum of the AI penalty column        Rs   227.76  (11 blocks)
    sum of the Enercast penalty column  Rs 2,306.09  (19 blocks)
    the summary reports                 Rs 2,306.09  (19 blocks)

That understates the production pipeline by a factor of ten.
The summary here is computed from this sheet's own schedule
column, and each summary row says which column it came from
so the same confusion cannot arise.

Run:  python -m tests.build_schedule_vs_meter_xlsx --day 2026-08-10
=========================================================
"""

import argparse
from datetime import date as date_type
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from modules.scheduling.effective_time import block_number
from tests.score_day_schedules import (
    FREEZE_BLOCKS, SLABS, dsm_penalty, run_time_of, stitch_day
)


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]
BLOCK_ENERGY_FACTOR = 250      # 0.25 h x 1000 kW/MW

DARK = "1B4332"
RED = "C00000"
RED_FILL = "FCE4E4"
BLUE_FILL = "E4EDFA"
HEAD_FILL = "1B4332"


def block_window(timestamp):
    """The block's clock window, e.g. 07:00-07:15."""

    end = timestamp + pd.Timedelta(minutes=15)

    return f"{timestamp:%H:%M}-{end:%H:%M}"


def build(day, folder, out_path):

    paths = [
        p for p in sorted(Path(folder).glob("*_schedule.csv"))
        if run_time_of(p) is not None and run_time_of(p).date() == day
    ]

    if not paths:
        raise SystemExit(f"No saved schedules for {day} in {folder}")

    stitched = stitch_day(paths, "forecast_mw")

    # Which run each block's number actually came from, under the freeze
    # horizon - the workbook's "Scheduled at" column. Recomputed the same
    # way stitch_day assigns them so the two can never disagree.
    owner = {}

    for index, path in enumerate(sorted(paths, key=lambda p: run_time_of(p))):

        run_time = run_time_of(path)
        engine = block_number(run_time)
        effective = engine if index == 0 else engine + max(FREEZE_BLOCKS, 1)

        frame = pd.read_csv(path, parse_dates=["timestamp"])

        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert(TIMEZONE)

        for stamp in frame["timestamp"]:
            if block_number(stamp) >= effective:
                owner[stamp] = f"{run_time:%H:%M}"

    frame = pd.DataFrame({
        "timestamp": list(stitched),
        "scheduled_mw": list(stitched.values()),
    }).sort_values("timestamp")

    meter = load_meter_history()

    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(TIMEZONE)

    if "is_real_measurement" in meter.columns:
        meter = meter[meter["is_real_measurement"].fillna(False)]

    meter = meter.assign(actual_mw=meter["active_power_kw"] / 1000.0)

    frame = frame.merge(
        meter[["timestamp", "actual_mw"]], on="timestamp", how="left"
    )

    frame["block"] = frame["timestamp"].map(block_number)
    frame["window"] = frame["timestamp"].map(block_window)
    frame["scheduled_at"] = frame["timestamp"].map(owner)

    # Signed, and in the workbook's direction: actual minus scheduled,
    # so a negative number means the plant produced LESS than promised.
    frame["deviation_mw"] = frame["actual_mw"] - frame["scheduled_mw"]
    frame["deviation_pct"] = frame["deviation_mw"] / CAPACITY_MW * 100
    frame["penalty_rs"] = frame["deviation_mw"].map(
        lambda d: dsm_penalty(d) if pd.notna(d) else None
    )

    # ---------------- workbook ----------------
    book = Workbook()
    sheet = book.active
    sheet.title = "Schedule vs Meter + Penalty"

    thin = Side(style="thin", color="C8CDD3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    sheet["A1"] = (
        f"{settings['plant']['name']} - Schedule vs Meter + DSM Penalty - {day}"
    )
    sheet["A1"].font = Font(name="Arial", size=14, bold=True, color=DARK)

    sheet["A2"] = (
        f"NEW LLM PIPELINE. Stitched from {len(paths)} run(s) under a "
        f"{FREEZE_BLOCKS}-block freeze horizon, so each run contributes only "
        f"the blocks it was actually allowed to change."
    )
    sheet["A2"].font = Font(name="Arial", size=9, italic=True, color="444444")

    headers = [
        "Block", "Time", "AI Schedule (MW)", "Actual (MW)", "Deviation (MW)",
        "Deviation % (Capacity)", "Penalty (Rs)", "Scheduled at",
    ]

    for column, title in enumerate(headers, start=1):
        cell = sheet.cell(4, column, title)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEAD_FILL)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = border

    row = 5

    for _, record in frame.iterrows():

        values = [
            int(record["block"]),
            record["window"],
            round(float(record["scheduled_mw"]), 4),
            None if pd.isna(record["actual_mw"]) else round(float(record["actual_mw"]), 4),
            None if pd.isna(record["deviation_mw"]) else round(float(record["deviation_mw"]), 4),
            None if pd.isna(record["deviation_pct"]) else round(float(record["deviation_pct"]), 3),
            None if record["penalty_rs"] is None else round(float(record["penalty_rs"]), 2),
            record["scheduled_at"],
        ]

        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row, column, value)
            cell.font = Font(name="Arial", size=10)
            cell.border = border
            cell.alignment = Alignment(horizontal="center")

        # Penalised blocks in red, free-band blocks in blue - the same
        # visual convention as the production workbook.
        penalty = record["penalty_rs"]

        if penalty is not None and penalty > 0:
            fill = PatternFill("solid", fgColor=RED_FILL)
            sheet.cell(row, 7).font = Font(name="Arial", size=10, bold=True,
                                           color=RED)
        elif pd.notna(record["actual_mw"]):
            fill = PatternFill("solid", fgColor=BLUE_FILL)
        else:
            fill = None

        if fill:
            for column in range(1, len(headers) + 1):
                if column != 7 or penalty in (None, 0):
                    sheet.cell(row, column).fill = fill

        row += 1

    scored = frame.dropna(subset=["actual_mw"])

    penalised = scored[scored["penalty_rs"] > 0]

    row += 1
    sheet.cell(row, 1, "DAY SUMMARY - computed from the 'AI Schedule (MW)' "
                       "column of THIS sheet").font = Font(
        name="Arial", size=11, bold=True, color=DARK)
    row += 1

    summary = [
        ("Blocks scheduled", len(frame), "0"),
        ("Blocks scored against real meter data", len(scored), "0"),
        ("Scheduled energy (MWh)", scored["scheduled_mw"].sum() * 0.25, "0.000"),
        ("Actual energy (MWh)", scored["actual_mw"].sum() * 0.25, "0.000"),
        ("Mean absolute deviation (MW)",
         scored["deviation_mw"].abs().mean(), "0.000"),
        ("Mean absolute deviation (% of capacity)",
         scored["deviation_mw"].abs().mean() / CAPACITY_MW * 100, "0.00"),
        ("Max absolute deviation (MW)",
         scored["deviation_mw"].abs().max(), "0.000"),
        ("Blocks inside the free 10% band",
         int((scored["deviation_mw"].abs() / CAPACITY_MW * 100 <= 10).sum()), "0"),
        ("Blocks that incurred a penalty", len(penalised), "0"),
        ("Worst single-block penalty (Rs)",
         float(penalised["penalty_rs"].max()) if len(penalised) else 0.0, "0.00"),
        ("TOTAL DSM PENALTY FOR THE DAY (Rs)",
         float(scored["penalty_rs"].sum()), "0.00"),
    ]

    for label, value, fmt in summary:

        sheet.cell(row, 1, label).font = Font(name="Arial", size=10)
        cell = sheet.cell(row, 3, value)
        cell.number_format = fmt
        cell.font = Font(
            name="Arial", size=10,
            bold="TOTAL" in label,
            color=RED if "TOTAL" in label else "000000",
        )
        row += 1

    row += 1
    sheet.cell(row, 1, "DSM SLAB PARAMETERS").font = Font(
        name="Arial", size=11, bold=True, color=DARK)
    row += 1

    sheet.cell(row, 1, "Installed capacity (MW)").font = Font(name="Arial", size=10)
    sheet.cell(row, 3, CAPACITY_MW)
    row += 1
    sheet.cell(row, 1, "Block energy factor (kWh per MW per block)").font = Font(
        name="Arial", size=10)
    sheet.cell(row, 3, BLOCK_ENERGY_FACTOR)
    row += 2

    for column, title in enumerate(
        ["Slab", "From %", "To %", "Rate (Rs/kWh)", "Upper edge (MW)"], start=1
    ):
        cell = sheet.cell(row, column, title)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEAD_FILL)
        cell.border = border

    row += 1

    for index, (low, high, rate) in enumerate(SLABS, start=1):

        edge = None if high is None else high / 100 * CAPACITY_MW

        for column, value in enumerate(
            [f"Slab {index}", low, "above" if high is None else high,
             rate, edge], start=1
        ):
            cell = sheet.cell(row, column, value)
            cell.font = Font(name="Arial", size=10)
            cell.border = border
            if column == 5 and edge is not None:
                cell.number_format = "0.000"

        row += 1

    widths = [8, 14, 17, 13, 14, 20, 13, 13]

    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width

    sheet.freeze_panes = "A5"

    book.save(out_path)

    return frame, scored, penalised


def main():

    parser = argparse.ArgumentParser(
        description="Schedule vs Meter + Penalty workbook, new pipeline"
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

    out = Path(
        args.out or
        Path.home() / "Downloads" /
        f"Schedule_vs_Meter_Penalty_{day}_NEW_PIPELINE.xlsx"
    )

    out.parent.mkdir(parents=True, exist_ok=True)

    frame, scored, penalised = build(day, args.folder, out)

    print("=" * 70)
    print(f"SCHEDULE vs METER + PENALTY — {day} — NEW PIPELINE")
    print("=" * 70)
    print(f"blocks scheduled  : {len(frame)}")
    print(f"blocks scored     : {len(scored)}")
    print(f"scheduled energy  : {scored['scheduled_mw'].sum() * 0.25:.3f} MWh")
    print(f"actual energy     : {scored['actual_mw'].sum() * 0.25:.3f} MWh")
    print(f"mean deviation    : "
          f"{scored['deviation_mw'].abs().mean() / CAPACITY_MW * 100:.2f}% "
          f"of capacity")
    print(f"in free band      : "
          f"{(scored['deviation_mw'].abs() / CAPACITY_MW * 100 <= 10).sum()}"
          f"/{len(scored)}")
    print(f"blocks penalised  : {len(penalised)}")
    print(f"TOTAL PENALTY     : Rs {scored['penalty_rs'].sum():,.2f}")
    print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
