"""
=========================================================
Solar Forecasting Project
Single-Sheet Schedule Report
=========================================================
One sheet holding the entire workflow, as the evaluation
document describes it: every 15-minute block down the rows,
every scheduling time across the columns.

Each column shows what THAT scheduling run published for
that block. A run's column is blank for blocks that had
already passed when it ran - which is the document's rule
("the schedule values for the blocks that have already
passed will remain unchanged") made visible as a staircase.
The final published value, the actual meter reading and the
error close the table.

Run:  python -m tests.build_single_sheet_report [YYYY-MM-DD]
=========================================================
"""

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-07-25"
SRC = Path("outputs/schedules")
OUT = Path(f"outputs/reports/Schedule_{DAY}_single_sheet.xlsx")

CAPACITY_MW = 5.1
PLANT = "Sirmour Solar Plant"

FONT = "Arial"
NAVY = "1F3864"
BLUE = "DCE6F1"
GREY = "F2F2F2"
AMBER = "FCE4D6"

schedule = pd.read_csv(SRC / f"day_schedule_{DAY}.csv")
per_run = pd.read_csv(SRC / f"day_schedule_{DAY}_per_run.csv")
run_log = pd.read_csv(SRC / f"day_schedule_{DAY}_run_log.csv")

run_times = list(run_log["scheduling_time"])

# block -> {scheduling_time: scheduled_mw}
matrix = per_run.pivot_table(
    index="block", columns="scheduling_time", values="scheduled_mw", aggfunc="first"
)

base = schedule[["block", "block_time", "scheduled_mw", "actual_mw",
                 "error_mw", "actual_is_real", "scheduled_at"]].copy()
base = base.set_index("block")

thin = Side(style="thin", color="BFBFBF")
box = Border(left=thin, right=thin, top=thin, bottom=thin)

wb = Workbook()
ws = wb.active
ws.title = f"Schedule {DAY}"

n_run = len(run_times)
total_cols = 2 + n_run + 4        # block, time, runs..., final, actual, error, source

# ---------------- title ----------------
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
c = ws.cell(row=1, column=1,
            value=f"{PLANT} - Reconstructed Day Schedule - {DAY}  (all values in MW, "
                  f"installed capacity {CAPACITY_MW} MW)")
c.font = Font(name=FONT, size=13, bold=True, color=NAVY)
c.alignment = Alignment(horizontal="center")
ws.row_dimensions[1].height = 20

ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=total_cols)
c = ws.cell(row=2, column=1,
            value="Each scheduling column shows what that run scheduled for the block. "
                  "Blank = the block had already passed when that run executed, so its value "
                  "stayed frozen from an earlier run.")
c.font = Font(name=FONT, size=9, italic=True, color="595959")
c.alignment = Alignment(horizontal="center")

# ---------------- grouped header ----------------
HEAD1, HEAD2 = 4, 5

ws.merge_cells(start_row=HEAD1, start_column=1, end_row=HEAD2, end_column=1)
ws.cell(row=HEAD1, column=1, value="Block")
ws.merge_cells(start_row=HEAD1, start_column=2, end_row=HEAD2, end_column=2)
ws.cell(row=HEAD1, column=2, value="Time")

first_run_col = 3
last_run_col = 2 + n_run
ws.merge_cells(start_row=HEAD1, start_column=first_run_col,
               end_row=HEAD1, end_column=last_run_col)
ws.cell(row=HEAD1, column=first_run_col,
        value="SCHEDULE GENERATED AT EACH SCHEDULING TIME (MW)")

for i, rt in enumerate(run_times):
    ws.cell(row=HEAD2, column=first_run_col + i, value=rt)

col_final = last_run_col + 1
col_actual = col_final + 1
col_error = col_actual + 1
col_src = col_error + 1

ws.merge_cells(start_row=HEAD1, start_column=col_final, end_row=HEAD1, end_column=col_error)
ws.cell(row=HEAD1, column=col_final, value="FINAL SCHEDULE vs ACTUAL")
ws.cell(row=HEAD2, column=col_final, value="Final (MW)")
ws.cell(row=HEAD2, column=col_actual, value="Actual (MW)")
ws.cell(row=HEAD2, column=col_error, value="Error (MW)")

ws.merge_cells(start_row=HEAD1, start_column=col_src, end_row=HEAD2, end_column=col_src)
ws.cell(row=HEAD1, column=col_src, value="Set by run")

for row in (HEAD1, HEAD2):
    for col in range(1, total_cols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = Font(name=FONT, size=9.5, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = box
ws.row_dimensions[HEAD1].height = 26
ws.row_dimensions[HEAD2].height = 18

# ---------------- data rows ----------------
start = HEAD2 + 1
blocks = list(base.index)

for r, block in enumerate(blocks, start=start):

    rec = base.loc[block]

    ws.cell(row=r, column=1, value=int(block)).number_format = "0"
    ws.cell(row=r, column=2, value=rec["block_time"])

    for i, rt in enumerate(run_times):
        value = matrix.loc[block, rt] if rt in matrix.columns else None
        if pd.isna(value):
            value = None
        cell = ws.cell(row=r, column=first_run_col + i, value=value)
        if value is not None:
            cell.number_format = "0.0000"
            # highlight the run that ended up owning this block
            if rt == rec["scheduled_at"]:
                cell.fill = PatternFill("solid", fgColor=BLUE)
                cell.font = Font(name=FONT, size=9.5, bold=True)

    ws.cell(row=r, column=col_final, value=float(rec["scheduled_mw"])
            ).number_format = "0.0000"

    actual = None if pd.isna(rec["actual_mw"]) else float(rec["actual_mw"])
    error = None if pd.isna(rec["error_mw"]) else float(rec["error_mw"])

    ws.cell(row=r, column=col_actual, value=actual).number_format = "0.0000"
    ec = ws.cell(row=r, column=col_error, value=error)
    ec.number_format = "0.0000;-0.0000"
    if error is not None:
        ec.font = Font(name=FONT, size=9.5,
                       color="C00000" if error > 0 else "1F4E79")

    ws.cell(row=r, column=col_src, value=rec["scheduled_at"])

    for col in range(1, total_cols + 1):
        cell = ws.cell(row=r, column=col)
        cell.border = box
        cell.alignment = Alignment(horizontal="center")
        # Font was already set deliberately on the highlighted and
        # error cells; only fill in the plain ones.
        if not cell.font.bold and col != col_error:
            cell.font = Font(name=FONT, size=9.5)
        if not bool(rec["actual_is_real"]):
            cell.fill = PatternFill("solid", fgColor=AMBER)

last_row = start + len(blocks) - 1

# ---------------- metrics ----------------
mrow = last_row + 2
ws.cell(row=mrow, column=1, value="ACCURACY vs ACTUAL METER DATA").font = Font(
    name=FONT, size=11, bold=True, color=NAVY)
ws.merge_cells(start_row=mrow, start_column=1, end_row=mrow, end_column=4)

fl = f"{get_column_letter(col_actual)}{start}:{get_column_letter(col_actual)}{last_row}"
er = f"{get_column_letter(col_error)}{start}:{get_column_letter(col_error)}{last_row}"
fn = f"{get_column_letter(col_final)}{start}:{get_column_letter(col_final)}{last_row}"

metrics = [
    ("Blocks scheduled", f"=COUNT({fn})", "0"),
    ("Blocks scored (real readings only)", f"=COUNT({fl})", "0"),
    ("Average percentage deviation (%)",
     f"=SUMPRODUCT(ABS({er}))/COUNT({er})/{CAPACITY_MW}*100", "0.00"),
    ("Mean absolute error (MW)", f"=SUMPRODUCT(ABS({er}))/COUNT({er})", "0.0000"),
    ("Root mean squared error (MW)",
     f"=SQRT(SUMPRODUCT({er},{er})/COUNT({er}))", "0.0000"),
    ("Scheduled energy (MWh)", f"=SUMIF({fl},\">=0\",{fn})*0.25", "0.000"),
    ("Actual energy (MWh)", f"=SUM({fl})*0.25", "0.000"),
    ("Average scheduled (MW)", f"=SUMIF({fl},\">=0\",{fn})/COUNT({fl})", "0.000"),
    ("Average actual (MW)", f"=AVERAGE({fl})", "0.000"),
    ("Peak scheduled (MW)", f"=MAX({fn})", "0.000"),
    ("Peak actual (MW)", f"=MAX({fl})", "0.000"),
]

for j, (label, formula, fmt) in enumerate(metrics, start=1):
    ws.cell(row=mrow + j, column=1, value=label).font = Font(name=FONT, size=10, bold=True)
    ws.merge_cells(start_row=mrow + j, start_column=1, end_row=mrow + j, end_column=3)
    c = ws.cell(row=mrow + j, column=4, value=formula)
    c.font = Font(name=FONT, size=10)
    c.number_format = fmt
    c.fill = PatternFill("solid", fgColor=GREY)
    c.border = box

# ---------------- per-run detail ----------------
srow = mrow + len(metrics) + 3
ws.cell(row=srow, column=1, value="WINDY CLIP USED AT EACH SCHEDULING TIME").font = Font(
    name=FONT, size=11, bold=True, color=NAVY)
ws.merge_cells(start_row=srow, start_column=1, end_row=srow, end_column=4)

heads = ["Scheduling time", "Windy clip stored for that time", "Vision used",
         "Blocks scheduled", "Weather bias factor"]
for i, h in enumerate(heads, start=1):
    cell = ws.cell(row=srow + 1, column=i, value=h)
    cell.font = Font(name=FONT, size=9.5, bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(horizontal="center", wrap_text=True)
    cell.border = box

for i, (_, rec) in enumerate(run_log.iterrows(), start=srow + 2):
    values = [rec["scheduling_time"], rec["windy_video_used"], rec["vision_signal"],
              int(rec["blocks_written"]), float(rec["weather_bias_factor"])]
    for j, v in enumerate(values, start=1):
        cell = ws.cell(row=i, column=j, value=v)
        cell.font = Font(name=FONT, size=9.5)
        cell.border = box
        cell.alignment = Alignment(horizontal="center")
        if j == 5:
            cell.number_format = "0.000"

# ---------------- legend ----------------
lrow = srow + len(run_log) + 3
notes = [
    ("Method", "Per 'Schedule Generation Workflow for Model Evaluation': each scheduling "
               "time uses only the Windy clip stored at that time and meter data up to block "
               "T, then schedules to end of day. Past blocks stay frozen; only future blocks "
               "are rewritten."),
    ("Blue cells", "The scheduling run whose value was finally published for that block."),
    ("Blank cells", "That block had already passed when the run executed."),
    ("Orange rows", "No measured meter reading - excluded from all accuracy figures."),
    ("Deviation", f"Mean absolute error as a percentage of the {CAPACITY_MW} MW installed "
                  "capacity."),
    ("Weather bias factor", "The weather forecast is scaled by this, learned from how much it "
                            "over-forecast sunlight on preceding days (0.732 = discounted by "
                            "26.8%)."),
]
for i, (k, v) in enumerate(notes):
    ws.cell(row=lrow + i, column=1, value=k).font = Font(name=FONT, size=9, bold=True)
    c = ws.cell(row=lrow + i, column=2, value=v)
    c.font = Font(name=FONT, size=9, color="595959")
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=lrow + i, start_column=2, end_row=lrow + i, end_column=total_cols)
    ws.row_dimensions[lrow + i].height = 26

# ---------------- widths / freeze ----------------
ws.column_dimensions["A"].width = 8
ws.column_dimensions["B"].width = 9
for i in range(n_run):
    ws.column_dimensions[get_column_letter(first_run_col + i)].width = 10.5
ws.column_dimensions[get_column_letter(col_final)].width = 11
ws.column_dimensions[get_column_letter(col_actual)].width = 11
ws.column_dimensions[get_column_letter(col_error)].width = 11
ws.column_dimensions[get_column_letter(col_src)].width = 11

ws.freeze_panes = ws.cell(row=start, column=3)

OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print(f"written: {OUT}")
print(f"one sheet: '{ws.title}'  |  {len(blocks)} blocks x {n_run} scheduling times")
