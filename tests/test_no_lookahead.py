"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Proof that a backtested run cannot see its own answer
=========================================================
A backtest is worth nothing if the pipeline can read the
meter for the blocks it is about to forecast. Every accuracy
number in the comparison report rests on that not happening,
so it is asserted here rather than assumed.

Four ways the answer could leak in, all checked:

  1. THE FEATURE TABLE. `actual_power_mw` must be present for
     blocks at or before the run time and absent for every
     block after it. This is the direct one.

  2. THE PROMPT. The feature table is rendered to text before
     it reaches Gemini. A column can be non-empty in the frame
     and still be dropped from the prompt, or - the danger -
     survive into the prompt after being filled in downstream.
     So the rendered prompt is searched for future actuals as
     literal numbers.

  3. THE CASE STORE. Retrieved precedent carries `actual_kw`.
     The store is built once across the whole period, so it
     holds days after the run being tested. Only days strictly
     before the run day may come back.

  4. THE CORRECTION LAYERS. Level calibration and block bias
     learn from finished days; both must ignore the run day
     and everything after it.

Run:  python -m tests.test_no_lookahead
      python -m tests.test_no_lookahead --day 2026-08-01
=========================================================
"""

import argparse
import sys

import pandas as pd

from config.config import settings
from modules.forecasting.llm_block_bias import LLMBlockBias
from modules.fusion.case_retrieval import CaseRetriever
from modules.fusion.llm_scheduler import LLMScheduler
from modules.preprocessing.windy_features import (
    WindyFeatureBuilder, load_meter_history
)


PASS = "  PASS"
FAIL = "  FAIL"


def check_feature_table(frame, run_time):
    """No actual generation may be attached to a future block."""

    if "actual_power_mw" not in frame.columns:
        print(f"{PASS}  feature table carries no actuals column at all")
        return True

    future = frame[frame["timestamp"] > run_time]
    leaked = future[future["actual_power_mw"].notna()]

    past = frame[frame["timestamp"] <= run_time]
    known = int(past["actual_power_mw"].notna().sum())

    if leaked.empty:
        print(f"{PASS}  feature table: {known} past block(s) have actuals, "
              f"0 of {len(future)} future block(s) do")
        return True

    print(f"{FAIL}  feature table: {len(leaked)} FUTURE block(s) carry "
          f"actual_power_mw")
    print(leaked[["timestamp", "actual_power_mw"]].head(10).to_string(index=False))
    return False


def check_prompt(prompt, frame, meter, run_time):
    """
    The rendered prompt must not contain the actual generation of any
    block the run is forecasting.

    Matching on the number as text is deliberately blunt. A value like
    "2.13" could appear by coincidence, so a hit is reported with the
    surrounding line and the count of distinct future values found -
    one stray match is noise, a dozen is a leak.
    """

    if meter is None or meter.empty:
        print(f"{PASS}  prompt: no meter history to leak")
        return True

    meter = meter.copy()

    timezone = settings["plant"]["timezone"]

    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(timezone)
    else:
        meter["timestamp"] = meter["timestamp"].dt.tz_convert(timezone)

    day = pd.Timestamp(run_time).date()

    future = meter[
        (meter["timestamp"] > run_time)
        & (meter["timestamp"].dt.date == day)
    ]

    if future.empty:
        print(f"{PASS}  prompt: no future meter rows exist for this day")
        return True

    # The raw meter frame is in kW - `active_power_kw` - while the
    # prompt talks in MW. Matching kW against an MW prompt would find
    # nothing and report a false PASS, so convert before comparing.
    scale = 1.0

    for name, factor in (
        ("active_power_kw", 0.001),
        ("power_kw", 0.001),
        ("active_power_mw", 1.0),
        ("power_mw", 1.0),
    ):
        if name in future.columns:
            column, scale = name, factor
            break
    else:
        raise SystemExit(
            "No power column found in the meter frame - this check would "
            f"pass vacuously. Columns present: {list(future.columns)}"
        )

    # Searching the WHOLE prompt for each value is too blunt to be
    # useful: a prompt carrying 29 blocks of clear-sky MW and kt will
    # coincidentally contain "0.25" or "1.49" somewhere almost every
    # time, and those first hits were exactly that - "measured kt 0.25"
    # on a PAST block, and a clear-sky column on an unrelated one.
    #
    # A leak is a block being told ITS OWN answer. So each future
    # block's actual is looked for only on the prompt lines that
    # mention that block's own time.

    lines = prompt.splitlines()

    # One prompt column is allowed to equal a future actual: clear-sky
    # power. It is pvlib sun geometry for that date, time and plant tilt
    # and never touches the meter, so a match there is arithmetic
    # coincidence, not knowledge. It fired on 2026-08-03 block 71, where
    # clear-sky and actual were both 0.83 MW - and gave itself away by
    # being IDENTICAL at all three run times while the anchor column
    # beside it moved (1.01 -> 0.68 -> 0.80), which is what a constant
    # does and what a leak cannot.
    SAFE_COLUMNS = {"clearsky_power_mw"}

    safe_indexes = set()

    for line in lines:
        if "block" in line and "time" in line and "|" in line:
            header = [cell.strip() for cell in line.split("|")]
            safe_indexes = {
                i for i, name in enumerate(header) if name in SAFE_COLUMNS
            }
            break

    checked = 0
    hits = []
    excused = []

    for _, row in future.iterrows():

        actual = float(row[column]) * scale

        # Near-zero night blocks match everything and prove nothing.
        if not (actual > 0.20):
            continue

        checked += 1

        clock = pd.Timestamp(row["timestamp"]).strftime("%H:%M")

        target = f"{actual:.2f}"

        for line in lines:

            if clock not in line or target not in line:
                continue

            cells = [cell.strip() for cell in line.split("|")]

            # Where the match sits only in a meter-independent column,
            # record it as coincidence rather than treating it as proof
            # of a leak.
            guilty = [
                i for i, cell in enumerate(cells)
                if target in cell and i not in safe_indexes
            ]

            if guilty:
                hits.append((clock, round(actual, 2), line.strip()[:96]))
            else:
                excused.append((clock, round(actual, 2)))

            break

    if not hits:
        note = ""
        if excused:
            note = (f"; {len(excused)} coincidental match(es) in "
                    f"{'/'.join(sorted(SAFE_COLUMNS))} ignored")
        print(f"{PASS}  prompt: {checked} future block(s) checked, none is "
              f"shown its own actual (matched on {column}){note}")
        return True

    print(f"{FAIL}  prompt: {len(hits)} of {checked} future block(s) appear "
          f"alongside their own actual generation")

    for clock, value, line in hits[:5]:
        print(f"          {clock}  actual {value:.2f} MW  ->  {line}")

    return False


def check_cases(run_time):
    """Retrieved precedent may only come from days before the run day."""

    retriever = CaseRetriever()

    if not retriever.available:
        print(f"{PASS}  case store absent or disabled - nothing to leak")
        return True

    day = pd.Timestamp(run_time).date()

    store_max = retriever.store["date"].max().date()

    cases = retriever.retrieve(
        {
            "block_hour": 12.0,
            "kt_now": 0.6,
            "final_forecast_kw": 2500.0,
            "horizon_min": 180.0,
        },
        exclude_date=day,
    )

    if cases.empty:
        print(f"{PASS}  case retrieval returned nothing for {day}")
        return True

    latest = cases["date"].max().date()

    if latest < day:
        print(f"{PASS}  case retrieval: {len(cases)} case(s), newest "
              f"{latest}, run day {day} (store runs to {store_max})")
        return True

    print(f"{FAIL}  case retrieval returned {latest} for a run on {day}")
    return False


def check_corrections(run_time):
    """Level and block-bias corrections must learn only from past days."""

    day = pd.Timestamp(run_time).date()

    bias = LLMBlockBias()

    frames = bias.history(day)

    ok = True

    for frame in frames:
        if "timestamp" in frame.columns:
            stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
            latest = stamps.dropna().max()
            if pd.notna(latest) and latest.date() >= day:
                print(f"{FAIL}  correction history includes {latest.date()}")
                ok = False

    if ok:
        print(f"{PASS}  correction layers: {len(frames)} finished day(s) of "
              f"history, all before {day}")

    return ok


def main():

    parser = argparse.ArgumentParser(
        description="Assert the pipeline cannot see the blocks it forecasts"
    )
    parser.add_argument("--day", default="2026-08-01")
    parser.add_argument("--time", default="11:15")

    args = parser.parse_args()

    hour, minute = map(int, args.time.split(":"))

    run_time = pd.Timestamp(
        f"{args.day} {hour:02d}:{minute:02d}",
        tz=settings["plant"]["timezone"],
    )

    print("=" * 70)
    print(f"NO-LOOKAHEAD CHECK  -  run at {run_time:%Y-%m-%d %H:%M %Z}")
    print("=" * 70)

    meter = load_meter_history()

    builder = WindyFeatureBuilder()

    frame, info = builder.build(run_time=run_time, meter=meter)

    results = []

    print("\n[1] feature table")
    results.append(check_feature_table(frame, run_time))

    print("\n[2] rendered prompt")

    scheduler = LLMScheduler()

    # dry_run builds the prompt and returns without calling Gemini, so
    # this check costs no quota and can run as often as it likes.
    _, meta = scheduler.decide(
        frame, run_time, satellite=info.get("satellite"), dry_run=True
    )

    prompt = meta["prompt"]

    print(f"  prompt is {len(prompt):,} characters, "
          f"{meta['blocks_requested']} block(s) requested")

    results.append(check_prompt(prompt, frame, meter, run_time))

    print("\n[3] case retrieval")
    results.append(check_cases(run_time))

    print("\n[4] correction layers")
    results.append(check_corrections(run_time))

    print("\n" + "-" * 70)

    if all(results):
        print("ALL CHECKS PASSED - the run cannot see its own answer.")
        return 0

    print("LEAK DETECTED - backtest numbers from this configuration are void.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
