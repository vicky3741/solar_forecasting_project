"""
=========================================================
Solar Forecasting Project
Effective Time / Freeze Horizon Test
=========================================================
Checks modules/scheduling/effective_time.py against the
mentor's "Simple Effective Time Schedule Guide" - literally
the two tables printed in it, so a future change to the
freeze logic that disagrees with the guide fails here
rather than in a schedule that has already been submitted.

Also checks the two behaviours that are easy to get wrong:

  * a block is frozen only if the PREVIOUS schedule actually
    has a value for it (the guide's "first schedule of the
    day - there may be nothing to freeze"), otherwise the
    first run of the day leaves a hole between where it
    stops and where the next run starts;
  * freeze_blocks of 0 or 1 reproduces the original
    behaviour exactly - every block after the run time is
    rewritten. That is what Sirmour runs on, so this is the
    guard that says the new machinery cannot disturb it.

Run:  python -m tests.test_effective_time
=========================================================
"""

import sys

import pandas as pd

from modules.scheduling.effective_time import (
    apply_freeze,
    block_number,
    effective_start,
    freeze_window,
)


# The guide's quick-reference table, verbatim.
BLOCK_TABLE = {
    "06:45": 28,
    "08:15": 34,
    "09:45": 40,
    "11:15": 46,
    "12:45": 52,
    "14:15": 58,
    "15:45": 64,
    "17:15": 70,
}

# The guide's per-plant tables: run time -> (frozen from, frozen to,
# effective from). Sirmour freezes 6 blocks, Kasipet/Kothagudem 3.
SIRMOUR_TABLE = {
    "06:45": (28, 33, 34),
    "08:15": (34, 39, 40),
    "09:45": (40, 45, 46),
    "11:15": (46, 51, 52),
    "12:45": (52, 57, 58),
    "14:15": (58, 63, 64),
    "15:45": (64, 69, 70),
}

KASIPET_TABLE = {
    "06:45": (28, 30, 31),
    "08:15": (34, 36, 37),
    "09:45": (40, 42, 43),
    "11:15": (46, 48, 49),
    "12:45": (52, 54, 55),
    "14:15": (58, 60, 61),
    "15:45": (64, 66, 67),
    "17:15": (70, 72, 73),
}

DAY = "2026-08-06"

failures = []


def check(label, actual, expected):

    if actual == expected:
        print(f"  ok   {label}")
        return

    print(f"  FAIL {label}: got {actual}, expected {expected}")
    failures.append(label)


def run_time_for(run_str):

    hour, minute = map(int, run_str.split(":"))

    return pd.Timestamp(DAY) + pd.Timedelta(hours=hour, minutes=minute)


def main():

    print("Block numbers (guide's quick reference)")
    for run_str, expected in BLOCK_TABLE.items():
        check(f"block at {run_str}", block_number(run_time_for(run_str)), expected)

    for name, freeze, table in [
        ("Sirmour (6 blocks / 90 min)", 6, SIRMOUR_TABLE),
        ("Kasipet + Bhupalpally (3 blocks / 45 min)", 3, KASIPET_TABLE),
    ]:
        print(f"\n{name}")
        for run_str, expected in table.items():
            got = freeze_window(run_time_for(run_str), freeze)
            check(f"{run_str} -> frozen {expected[0]}-{expected[1]}, "
                  f"effective {expected[2]}", got, expected)

    print("\nNo freeze horizon (Sirmour's current setting) is a no-op")
    for freeze in (0, 1):
        run_time = run_time_for("11:15")
        check(
            f"freeze_blocks={freeze} takes effect one block later",
            effective_start(run_time, freeze),
            run_time + pd.Timedelta(minutes=15),
        )
        check(
            f"freeze_blocks={freeze} freezes nothing",
            freeze_window(run_time, freeze)[:2],
            (None, None),
        )

    print("\nA block is frozen only when the previous schedule has one")

    run_time = run_time_for("11:15")
    # This run's raw output for the next four blocks.
    new_values = {
        run_time + pd.Timedelta(minutes=15 * step): 1000.0 * step
        for step in range(1, 5)
    }
    # The standing schedule covers blocks 47 and 48 but not 49/50.
    previous = {
        run_time + pd.Timedelta(minutes=15): 111.0,
        run_time + pd.Timedelta(minutes=30): 222.0,
    }

    published, frozen = apply_freeze(new_values, previous, run_time, freeze_blocks=3)

    check("2 blocks held", len(frozen), 2)
    check("block 47 keeps the declared value",
          published[run_time + pd.Timedelta(minutes=15)], 111.0)
    check("block 48 keeps the declared value",
          published[run_time + pd.Timedelta(minutes=30)], 222.0)
    check("block 49 takes the new value",
          published[run_time + pd.Timedelta(minutes=45)], 3000.0)

    published, frozen = apply_freeze(new_values, {}, run_time, freeze_blocks=3)

    check("nothing to freeze on the first schedule of the day", len(frozen), 0)
    check("every block gets this run's value",
          published[run_time + pd.Timedelta(minutes=15)], 1000.0)

    print()

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1

    print("All effective-time checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
