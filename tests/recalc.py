"""
=========================================================
Solar Forecasting Project
Workbook Recalculation + Chart Export
=========================================================
openpyxl writes FORMULAS but cannot evaluate them, so a
freshly built report has no cached values anywhere. That is
invisible if you only ever open the file in Excel - Excel
recalculates on open and everything looks right - but it
breaks two things that matter:

  * the CHARTS. A chart series pointing at a formula cell
    with no cached value plots nothing, so the penalty bar
    chart and the moving penalty band render EMPTY in every
    viewer that does not recalculate (Excel's own preview,
    Google Sheets import, WhatsApp/Drive previews - i.e.
    exactly how the mentor sees it).
  * anything reading the file with data_only=True, which
    returns None for every formula cell.

This drives real Excel over COM to do the one thing openpyxl
cannot: evaluate the sheet and SAVE the cached values back.
It also exports each chart to PNG, so the graphs can be
looked at (and pasted into a message) without opening Excel
at all.

Run:  python -m tests.recalc <workbook.xlsx> [more.xlsx ...]
      python -m tests.recalc <workbook.xlsx> --no-png
=========================================================
"""

import argparse
import sys
from pathlib import Path

# Excel constants (xlNone / xlLineStyleNone) - both report -4142.
XL_NONE = -4142

# Excel's Chart.Export writes whatever size the chart is on the
# sheet; these scale it up first so the exported PNG is legible
# rather than a 400px thumbnail of a 24 cm chart.
EXPORT_SCALE = 2.0


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("workbooks", nargs="+", help="xlsx files to recalculate")
    parser.add_argument(
        "--no-png", dest="png", action="store_false",
        help="recalculate only, do not export the charts as PNG",
    )
    return parser.parse_args()


def prune_invisible_legend_entries(chart):
    """
    Drops legend entries belonging to series that draw nothing.

    A "fill between two lines" band is built by stacking an INVISIBLE
    anchor series under a visible one (see build_penalty_report), and
    that anchor has no business in the legend - it is scaffolding, not
    a quantity. openpyxl writes the standard request to hide it
    (<legendEntry><idx 0><delete 1>) and Excel simply ignores it on a
    combined chart, so it has to be deleted through Excel itself.

    An entry is scaffolding when its key has neither fill nor border:
    a real area series has a fill, a real line series has a border.
    Deleted back-to-front so the surviving indices do not shift.

    Returns how many entries were removed.
    """

    legend = chart.Legend
    removed = 0

    for index in range(legend.LegendEntries().Count, 0, -1):

        key = legend.LegendEntries(index).LegendKey

        try:
            invisible = (key.Interior.ColorIndex == XL_NONE
                         and key.Border.LineStyle == XL_NONE)
        except Exception:
            # Some key types (3-D, pie slices) expose neither property;
            # those are never band scaffolding, so leave them alone.
            continue

        if invisible:
            legend.LegendEntries(index).Delete()
            removed += 1

    return removed


def export_charts(worksheet, workbook_path):
    """
    Every chart on the sheet to <workbook stem>_chart<N>.png beside
    the workbook. Returns the paths written.
    """

    written = []

    for index in range(1, worksheet.ChartObjects().Count + 1):

        chart_object = worksheet.ChartObjects(index)

        width, height = chart_object.Width, chart_object.Height
        chart_object.Width = width * EXPORT_SCALE
        chart_object.Height = height * EXPORT_SCALE

        png_path = workbook_path.with_name(
            f"{workbook_path.stem}_chart{index}.png"
        )
        chart_object.Chart.Export(str(png_path), "PNG")

        # Put the chart back the size the report author chose, so
        # recalculating does not quietly restyle the workbook.
        chart_object.Width, chart_object.Height = width, height

        written.append(png_path)

    return written


def main():

    args = parse_args()

    try:
        import win32com.client as com
    except ImportError:
        print("pywin32 is not installed - pip install pywin32")
        return 1

    excel = com.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    try:
        for name in args.workbooks:

            path = Path(name).resolve()

            if not path.exists():
                print(f"  ! {path} not found")
                continue

            workbook = excel.Workbooks.Open(str(path))

            # CalculateFullRebuild rebuilds the dependency tree as well
            # as the values: a plain Calculate() can skip cells Excel
            # believes are already up to date, which is every cell in a
            # file it has just opened for the first time.
            excel.CalculateFullRebuild()

            sheet = workbook.Sheets(1)

            pruned = sum(
                prune_invisible_legend_entries(sheet.ChartObjects(i).Chart)
                for i in range(1, sheet.ChartObjects().Count + 1)
            )

            pngs = export_charts(sheet, path) if args.png else []

            workbook.Save()
            workbook.Close(SaveChanges=False)

            print(f"  + {path.name} recalculated"
                  + (f", {pruned} band-anchor legend entries removed" if pruned else ""))
            for png in pngs:
                print(f"      chart -> {png.name}")

    finally:
        excel.Quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
