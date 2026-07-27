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

    # Drop the pre-dawn blocks the meter covers but no scheduling run ever
    # reached - the first run is at 06:45, so nothing before 07:00 has a
    # schedule to compare against and those rows were only empty space.
    merged = merged[merged["scheduled_mw"].notna()].reset_index(drop=True)

    # The date is stated once in the title, so the rows carry time only.
    out = pd.DataFrame({
        "Time": merged["timestamp"].dt.strftime("%H:%M"),
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

    # ---------------- accuracy, in MW only ----------------
    # Blocks the meter actually measured; the trailing blocks after 18:00
    # are scheduled but have no reading yet, so they cannot be scored.
    both = out.dropna(subset=["Scheduled Power (kW)", "Actual Power (kW)"])

    scheduled_mw = both["Scheduled Power (kW)"] / 1000
    actual_mw = both["Actual Power (kW)"] / 1000
    error_mw = both["Error (kW)"] / 1000

    total_predicted = float(scheduled_mw.sum())
    total_actual = float(actual_mw.sum())
    total_error = total_predicted - total_actual
    total_abs_error = float(error_mw.abs().sum())
    mae = float(error_mw.abs().mean())
    rmse = float((error_mw ** 2).mean() ** 0.5)
    deviation = mae / (CAPACITY_KW / 1000) * 100

    mrow = last + 2
    ws.cell(row=mrow, column=1, value="ACCURACY vs ACTUAL METER DATA").font = Font(
        name=FONT, size=11, bold=True, color=NAVY)

    # Energy figures are deliberately absent: they can only be stated in
    # MWh, and this report is MW-only by request.
    metrics = [
        ("Blocks scored (real meter readings)", len(both), "0"),
        ("Average predicted (MW)", round(float(scheduled_mw.mean()), 3), "0.000"),
        ("Average actual (MW)", round(float(actual_mw.mean()), 3), "0.000"),
        ("Peak predicted (MW)", round(float(scheduled_mw.max()), 3), "0.000"),
        ("Peak actual (MW)", round(float(actual_mw.max()), 3), "0.000"),
        ("Total predicted (MW)", round(total_predicted, 3), "0.000"),
        ("Total actual (MW)", round(total_actual, 3), "0.000"),
        ("Total error - predicted minus actual (MW)", round(total_error, 3),
         "+0.000;-0.000"),
        ("Total absolute error (MW)", round(total_abs_error, 3), "0.000"),
        ("Average percentage deviation (%)", round(deviation, 2), "0.00"),
        ("Mean absolute error (MW)", round(mae, 4), "0.0000"),
        ("Root mean squared error (MW)", round(rmse, 4), "0.0000"),
    ]

    for j, (label, value, fmt) in enumerate(metrics, start=1):
        ws.cell(row=mrow + j, column=1, value=label).font = f_bold
        cell = ws.cell(row=mrow + j, column=3, value=value)
        cell.font = f_body
        cell.number_format = fmt
        cell.fill = fill_grey
        cell.border = box
        cell.alignment = center

    ws.column_dimensions["A"].width = 34
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

    print(f"rows: {len(out)}   scored blocks: {len(both)}")
    print()
    print(out.head(6).to_string(index=False))
    print("...")
    print(out.tail(4).to_string(index=False))
    print()
    print("ACCURACY (MW only)")
    for label, value, _fmt in metrics:
        print(f"  {label:<44} {value}")
    print()
    print(f"saved CSV  : {csv_path}")
    print(f"saved Excel: {target}")


if __name__ == "__main__":
    main()
