"""
=========================================================
Solar Forecasting Project
Day-Schedule Backfill
=========================================================
Reconstructs the mentor's day schedule for EVERY day a plant
has meter data for, oldest first, by calling
tests/generate_schedule_for_day.py once per day.

Why a plant needs this before it is really running
--------------------------------------------------
The block-bias corrector (mentor guidance 2026-08-06) learns
its time-of-day shape from the reconstructed day schedules of
the last few finished days, and refuses to correct at all
when fewer than `min_days` exist or the newest is stale. A
newly onboarded plant therefore has that correction silently
switched off until its history has been reconstructed once.
This is that one-off pass - and it is also how the new plants
get an evaluation baseline to show the mentor.

ORDER MATTERS. Days are built oldest first, and each day's
BlockBiasCorrector loads only schedules strictly BEFORE it,
so no day is ever built with knowledge of its own outcome or
a later day's. Building them in any other order would leak
the future into the past.

The meter history is preprocessed ONCE and reused for every
day, rather than re-read 37 times.

Run:
    SOLAR_PLANT=kasipet python -m tests.backfill_day_schedules
    SOLAR_PLANT=kasipet python -m tests.backfill_day_schedules --from 2026-07-20
    SOLAR_PLANT=kasipet python -m tests.backfill_day_schedules --redo
=========================================================
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import pandas as pd

from config.config import settings
from tests import generate_schedule_for_day as generator


def parse_args():

    parser = argparse.ArgumentParser(
        description="Reconstruct every available day's schedule for this plant."
    )
    parser.add_argument("--from", dest="start", default=None,
                        help="first day to build (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end", default=None,
                        help="last day to build (YYYY-MM-DD)")
    parser.add_argument("--redo", action="store_true",
                        help="rebuild days whose schedule file already exists")
    parser.add_argument("--quiet", action="store_true",
                        help="hide each day's full printed report")
    return parser.parse_args()


def main():

    args = parse_args()

    plant = settings["plant"]
    out_dir = Path(settings["outputs"]["schedules"])

    print("=" * 78)
    print(f"BACKFILL DAY SCHEDULES - {plant['name']} ({plant['capacity_mw']} MW)")
    print("=" * 78)
    print("Loading and preprocessing the meter history once...")

    data = generator.load_data()

    days = sorted(data["timestamp"].dt.date.unique())

    if args.start:
        days = [d for d in days if d >= pd.Timestamp(args.start).date()]
    if args.end:
        days = [d for d in days if d <= pd.Timestamp(args.end).date()]

    print(f"{len(data)} blocks across {len(days)} days "
          f"({days[0]} .. {days[-1]})")
    print(f"freeze horizon: {generator.FREEZE_BLOCKS} blocks")
    print(f"writing to    : {out_dir}")
    print()

    built, skipped, failed = 0, 0, []

    for day in days:

        target = out_dir / f"day_schedule_{day}.csv"

        if target.exists() and not args.redo:
            print(f"  = {day}  already built")
            skipped += 1
            continue

        started = time.time()

        try:
            # Each day's own report is verbose; the backfill's value is
            # the one-line-per-day progress, so it is captured unless
            # the caller asked to see it.
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    generator.main(day_str=str(day), data=data)
            else:
                generator.main(day_str=str(day), data=data)

        except Exception as error:
            print(f"  ! {day}  FAILED: {error}")
            failed.append((day, str(error)))
            continue

        summary = out_dir / f"day_schedule_{day}_summary.csv"
        deviation = "-"

        if summary.exists():
            frame = pd.read_csv(summary)
            match = frame[frame["metric"] == "Average percentage deviation (%)"]
            if not match.empty:
                deviation = f"{float(match['value'].iloc[0]):.2f}%"

        print(f"  + {day}  deviation {deviation:>7s}   "
              f"({time.time() - started:.0f}s)")
        built += 1

    print()
    print("-" * 78)
    print(f"built   : {built}")
    print(f"skipped : {skipped}")
    print(f"failed  : {len(failed)}")

    for day, error in failed:
        print(f"   ! {day}: {error}")

    if built:
        print()
        print("The block-bias corrector can now learn from these days. Next:")
        print(f"    SOLAR_PLANT={plant['key']} python -m tests.test_preprocessing")
        print(f"    SOLAR_PLANT={plant['key']} python -m tests.test_backtest")
        print(f"    SOLAR_PLANT={plant['key']} python -m tests.test_case_based_experiment")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
