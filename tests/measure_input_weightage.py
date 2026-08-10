"""
=========================================================
Solar Forecasting Project - NEW APPROACH
How much does each input actually move the schedule?
=========================================================
"Weightage" is easy to assert and hard to earn. Nothing in
this pipeline holds a weight for Windy or the meter or the
weather - there is no `windy_weight = 0.4` anywhere. The LLM
reads a table and returns a clear-sky index per block, and a
validator may pull that back toward the anchor. So the only
honest way to state a weightage is to MEASURE how far the
output moves with each input.

This measures it in three layers, all from files already on
disk - no new Gemini calls.

  1. AVAILABILITY. An input that is empty has weight zero
     however good it is. Counted per block, per day.

  2. WHAT DROVE THE LLM's DECISION. Each saved prompt carries
     the per-block table it was given; each saved schedule
     carries the `llm_kt` that came back. Regressing llm_kt on
     the inputs it was shown - anchor_kt, weather_kt, windy_kt
     - gives the LLM's own revealed weighting.

  3. WHAT DROVE THE FINAL NUMBER. The LLM does not have the
     last word: the blend and the validator sit after it.
     Non-negative least squares of forecast_mw on the LLM's
     own view and on the anchor splits the credit between
     them.

The regressions are standardised (each input divided by its
own spread) so a coefficient means "how far the output moves
for a typical move in this input", not "what units is this in".
Shares are reported as |coefficient| normalised to 100%.

CAVEAT WORTH READING
--------------------
This is a decomposition of what the pipeline DID, not proof of
what each input is WORTH. An input can dominate the output and
still be useless if it is wrong. Weight and skill are different
questions - the penalty comparison answers the second one.

Run:  python -m tests.measure_input_weightage
      python -m tests.measure_input_weightage --from 2026-08-01 --to 2026-08-05
=========================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tests.score_day_schedules import run_time_of


# The prompt renders its per-block table as pipe-separated rows under a
# header that starts with "block | time". Columns vary between runs -
# all-empty ones are dropped before the prompt is built - so the header
# is parsed rather than assumed.
_HEADER = re.compile(r"^\s*block\s*\|\s*time\s*\|")


def parse_prompt_table(path):
    """The per-block input table the model was actually shown."""

    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    header = None
    rows = []

    for line in text.splitlines():

        if header is None:
            if _HEADER.match(line):
                header = [cell.strip() for cell in line.split("|")]
            continue

        cells = [cell.strip() for cell in line.split("|")]

        if len(cells) != len(header):
            # The table has ended; everything after it is prose.
            if rows:
                break
            continue

        if not cells[0].isdigit():
            if rows:
                break
            continue

        rows.append(cells)

    if header is None or not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, columns=header)

    for name in frame.columns:
        if name != "time":
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    return frame


def load_runs(folder, start, end):
    """Every run in the window, prompt table joined to its schedule."""

    frames = []

    for path in sorted(Path(folder).glob("*_schedule.csv")):

        run_time = run_time_of(path)

        if run_time is None or not (start <= run_time.date() <= end):
            continue

        prompt_path = Path(str(path).replace("_schedule.csv", "_prompt.txt"))

        if not prompt_path.exists():
            continue

        schedule = pd.read_csv(path)

        table = parse_prompt_table(prompt_path)

        if table.empty or "block" not in schedule.columns:
            continue

        merged = schedule.merge(
            table, on="block", how="left", suffixes=("", "_prompt")
        )

        merged["run_time"] = run_time
        merged["day"] = run_time.date()

        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def availability(frame):
    """Share of blocks on which each input carried a value at all."""

    total = len(frame)

    checks = {
        "Windy scraped numbers": "windy_kt",
        "Weather forecast (ECMWF)": "weather_kt",
        "Meter anchor": "anchor_mw",
        "Clear-sky physics": "clearsky_power_mw",
    }

    rows = []

    for label, column in checks.items():

        if column not in frame.columns:
            rows.append((label, column, 0.0, "column absent entirely"))
            continue

        present = float(frame[column].notna().mean() * 100)

        note = "" if present > 0 else "present as a column, empty on every block"

        rows.append((label, column, present, note))

    return rows


def shares(frame, target, inputs):
    """
    Standardised least squares of `target` on `inputs`, reported as
    normalised absolute shares.

    Standardising matters: anchor_kt and weather_kt live on similar
    scales but clear-sky MW does not, and on raw values the widest
    column would look like the most influential one purely because it
    is the widest.
    """

    usable = [
        name for name in inputs
        if name in frame.columns and frame[name].notna().sum() > 10
        and float(frame[name].std(skipna=True) or 0) > 1e-9
    ]

    if not usable or target not in frame.columns:
        return None

    data = frame[[target] + usable].dropna()

    if len(data) < 20:
        return None

    y = data[target].to_numpy(dtype=float)
    y = (y - y.mean()) / (y.std() or 1.0)

    matrix = []

    for name in usable:
        column = data[name].to_numpy(dtype=float)
        matrix.append((column - column.mean()) / (column.std() or 1.0))

    design = np.column_stack(matrix)

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    predicted = design @ coefficients

    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum(y ** 2))

    r_squared = 1.0 - residual / total if total > 1e-12 else float("nan")

    magnitude = np.abs(coefficients)
    total_magnitude = magnitude.sum()

    return {
        "n": len(data),
        "r_squared": r_squared,
        "terms": [
            {
                "name": name,
                "coefficient": float(coefficient),
                "share_pct": float(
                    magnitude[index] / total_magnitude * 100
                ) if total_magnitude > 1e-12 else 0.0,
            }
            for index, (name, coefficient) in enumerate(
                zip(usable, coefficients)
            )
        ],
    }


def report_shares(title, result, note=""):

    print(f"\n{title}")

    if result is None:
        print("  not measurable - too few blocks or the inputs do not vary")
        return

    print(f"  {result['n']} block(s), R-squared {result['r_squared']:.3f}"
          f"{'  ' + note if note else ''}")

    for term in sorted(result["terms"], key=lambda t: -abs(t["coefficient"])):
        bar = "#" * int(round(term["share_pct"] / 3))
        print(f"    {term['name']:<22} {term['share_pct']:>5.1f}%  "
              f"(coef {term['coefficient']:+.3f})  {bar}")


def main():

    parser = argparse.ArgumentParser(
        description="Measure how much each input moves the schedule"
    )
    parser.add_argument("--from", dest="start", default="2026-08-01")
    parser.add_argument("--to", dest="end", default="2026-08-05")
    parser.add_argument("--folder", default="outputs/llm_schedules")

    args = parser.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()

    frame = load_runs(args.folder, start, end)

    if frame.empty:
        raise SystemExit(f"No runs with prompts found in {args.folder}")

    days = sorted(frame["day"].unique())

    print("=" * 74)
    print(f"INPUT WEIGHTAGE   {start} to {end}")
    print(f"{frame['run_time'].nunique()} run(s) across {len(days)} day(s), "
          f"{len(frame)} scheduled block(s)")
    print("=" * 74)

    print("\n[1] AVAILABILITY - an empty input has weight zero")
    print("-" * 74)

    for label, column, present, note in availability(frame):
        flag = "  <-- " + note if note else ""
        print(f"  {label:<26} {column:<20} {present:>6.1f}%{flag}")

    # The LLM is shown kt, not MW, so its decision is regressed on the kt
    # columns it actually saw.
    if "anchor_mw" in frame.columns and "clearsky_power_mw" in frame.columns:
        frame["anchor_kt_derived"] = (
            frame["anchor_mw"] / frame["clearsky_power_mw"].replace(0, np.nan)
        )

    llm_inputs = [
        name for name in
        ("anchor_kt", "anchor_kt_derived", "weather_kt", "windy_kt")
        if name in frame.columns
    ]

    # anchor_kt from the prompt is preferred; the derived one is a
    # fallback for runs whose prompt dropped that column.
    if "anchor_kt" in llm_inputs and "anchor_kt_derived" in llm_inputs:
        llm_inputs.remove("anchor_kt_derived")

    print("\n" + "-" * 74)
    print("[2] WHAT DROVE THE LLM's DECISION (llm_kt regressed on its inputs)")
    print("-" * 74)

    report_shares("  llm_kt ~ inputs shown in the prompt",
                  shares(frame, "llm_kt", llm_inputs))

    print("\n" + "-" * 74)
    print("[3] WHAT DROVE THE FINAL NUMBER (after blend and validator)")
    print("-" * 74)

    if {"llm_kt", "clearsky_power_mw"} <= set(frame.columns):
        frame["llm_view_mw"] = frame["llm_kt"] * frame["clearsky_power_mw"]

    report_shares(
        "  forecast_mw ~ the LLM's own view vs the meter anchor",
        shares(frame, "forecast_mw", ["llm_view_mw", "anchor_mw"]),
    )

    if "was_adjusted" in frame.columns:

        adjusted = frame["was_adjusted"].fillna(False)

        if adjusted.dtype == object:
            adjusted = adjusted.astype(str).str.lower().isin(["true", "1"])

        print(f"\n  Validator overrode the LLM on "
              f"{int(adjusted.sum())} of {len(frame)} block(s) "
              f"({adjusted.mean() * 100:.1f}%).")

        if "adjustment_note" in frame.columns:
            notes = frame.loc[adjusted, "adjustment_note"].value_counts()
            for note, count in notes.head(5).items():
                print(f"    {count:>5}  {note}")

    print("\n" + "=" * 74)
    print("Read this as INFLUENCE, not as VALUE. An input can dominate the")
    print("output and still be wrong - the penalty comparison is what says")
    print("whether the influence was earned.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
