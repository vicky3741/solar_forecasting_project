"""
=========================================================
Solar Forecasting Project
Simple Schedule vs Meter Comparison
=========================================================
The detailed workbook (one column per scheduling time) was
too dense to read against the meter file, so this emits the
plainest possible comparison, laid out exactly like the
meter CSV the mentor already reads:

    TimeStamp, Active Power (kW), ...

One row per 15-minute block, the SAME timestamps as the
meter file, and one power value per row. Nothing is
accumulated - each row is that block's own power, exactly
as the meter records it. Rows the meter covers but the
schedule does not (before the first scheduling run) are
left blank rather than dropped, so the two files line up
row for row and can be pasted side by side.

Run:  python -m tests.build_simple_schedule [YYYY-MM-DD]
=========================================================
"""

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from config.config import settings
from modules.preprocessing.preprocess import DataPreprocessor


DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-07-25"
SRC = Path("outputs/schedules")
OUT_DIR = Path("outputs/reports")

CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
PLANT = settings["plant"]["name"]

FONT = "Arial"
NAVY = "1F3864"
GREY = "F2F2F2"


def load_actual(day):
    """The day's meter readings, straight from the raw file."""

    path = Path(settings["paths"]["historical_data"]) / f"{day.replace('-', '_')}_SOLAR_INV.csv"

    frame = DataPreprocessor().preprocess(
        file_path=path,
        required_columns=["TimeStamp"],
        timestamp_column="TimeStamp",
    )

    columns = ["timestamp", "active_power_kw"]
    if "is_real_measurement" in frame.columns:
        columns.append("is_real_measurement")

    return frame[columns]


def main():

    schedule = pd.read_csv(SRC / f"day_schedule_{DAY}.csv", parse_dates=["timestamp"])
    actual = load_actual(DAY)

    # Every timestamp either side knows about, so the file lines up with
    # the meter CSV row for row.
    merged = actual.merge(
        schedule[["timestamp", "scheduled_mw"]], on="timestamp", how="outer"
    ).sort_values("timestamp").reset_index(drop=True)

    merged = merged[merged["timestamp"].dt.date == pd.Timestamp(DAY).date()]

    out = pd.DataFrame({
        "TimeStamp": merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Scheduled Power (kW)": (merged["scheduled_mw"] * 1000).round(2),
        "Actual Power (kW)": merged["active_power_kw"].round(2),
    })
    out["Error (kW)"] = (
        out["Scheduled Power (kW)"] - out["Actual Power (kW)"]
    ).round(2)

    csv_path = OUT_DIR / f"Schedule_vs_Meter_{DAY}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)

    # ---------------- the same thing as a workbook ----------------
    wb = Workbook()
    ws = wb.active
    ws.title = f"Schedule vs Meter"

    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center")
    f_head = Font(name=FONT, size=10, bold=True, color="FFFFFF")
    f_body = Font(name=FONT, size=10)
    f_bold = Font(name=FONT, size=10, bold=True)
    fill_head = PatternFill("solid", fgColor=NAVY)
    fill_grey = PatternFill("solid", fgColor=GREY)

    ws.cell(row=1, column=1,
            value=f"{PLANT} - Scheduled vs Actual - {DAY}  (15-minute blocks, kW)"
            ).font = Font(name=FONT, size=12, bold=True, color=NAVY)

    headers = list(out.columns)
    for i, h in enumerate(headers, start=1):
        cell = ws.cell(row=3, column=i, value=h)
        cell.font = f_head
        cell.fill = fill_head
        cell.alignment = center
        cell.border = box

    for r, (_, rec) in enumerate(out.iterrows(), start=4):
        for i, key in enumerate(headers, start=1):
            value = rec[key]
            if pd.isna(value):
                value = None
            cell = ws.cell(row=r, column=i, value=value)
            cell.font = f_body
            cell.border = box
            cell.alignment = center
            if i > 1:
                cell.number_format = "0.00"

    last = 3 + len(out)

    # totals over the blocks both sides cover
    both = out.dropna(subset=["Scheduled Power (kW)", "Actual Power (kW)"])

    trow = last + 1
    labels = ["TOTAL", f"{len(both)} blocks",
              round(both["Scheduled Power (kW)"].sum() / 1000, 3),
              round(both["Actual Power (kW)"].sum() / 1000, 3),
              round(both["Error (kW)"].sum() / 1000, 3)]
    for i, v in enumerate(labels[:1] + labels[2:], start=1):
        cell = ws.cell(row=trow, column=i, value=v)
        cell.font = f_head
        cell.fill = fill_head
        cell.alignment = center
        cell.border = box
        if i > 1:
            cell.number_format = "0.000"
    ws.cell(row=trow, column=1, value="TOTAL (MW)")

    note = trow + 2
    for i, (k, v) in enumerate([
        ("Rows", f"One row per 15-minute block, matching the meter file's timestamps "
                 f"exactly. Nothing is accumulated - each row is that block's own power."),
        ("Blank cells", "Blocks the meter covers but no scheduling run reached (before "
                        "06:45), or blocks with no meter reading (after 18:00)."),
        ("Error", "Scheduled minus actual. Negative = we scheduled below what the plant "
                  "produced; positive = above."),
        ("TOTAL row", f"Sum of the {len(both)} blocks both sides cover, shown in MW."),
    ]):
        ws.cell(row=note + i, column=1, value=k).font = f_bold
        c = ws.cell(row=note + i, column=2, value=v)
        c.font = Font(name=FONT, size=9, color="595959")
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[note + i].height = 24

    ws.column_dimensions["A"].width = 21
    for col in "BCD":
        ws.column_dimensions[col].width = 19
    ws.freeze_panes = ws.cell(row=4, column=1)

    xlsx_path = OUT_DIR / f"Schedule_vs_Meter_{DAY}.xlsx"
    target = xlsx_path
    for attempt in range(1, 20):
        try:
            wb.save(target)
            break
        except PermissionError:
            target = xlsx_path.with_name(
                f"{xlsx_path.stem}_{attempt}{xlsx_path.suffix}")

    print(f"rows: {len(out)}  (meter file day range)")
    print(f"blocks with both schedule and meter: {len(both)}")
    print()
    print(out.head(8).to_string(index=False))
    print("...")
    print(out.tail(4).to_string(index=False))
    print()
    print(f"saved CSV  : {csv_path}")
    print(f"saved Excel: {target}")


if __name__ == "__main__":
    main()
