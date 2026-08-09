"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Daily actuals feedback loop
=========================================================
Adopted from daily_feedback.py / process_daily_actuals.py in
github.com/Kushal70-51/Windy-Project-3.

Closes the loop this project has been missing. Three jobs,
run once a day (or after any run):

  1. ATTACH   put real meter generation onto every saved LLM
              schedule whose blocks have since happened.
  2. LEARN    turn those scored blocks into cases, so what
              happened today becomes precedent tomorrow.
  3. MEASURE  report the recent bias, which is what decides
              how much rope the validator gives the model
              (modules/fusion/validator.py).

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
Two things in this codebase are already broken for want of
exactly this:

  * The existing case store was built ONCE, on 2026-07-24,
    and then quietly nudged live forecasts with Jul 6-22
    analogues for a fortnight. Nobody noticed. settings.yaml
    now carries a staleness warning about it - a warning is
    a workaround for not having this file.
  * The validator's adaptive deviation bound reads
    `actual_mw` and `forecast_mw` off saved schedules. Those
    columns do not exist until something writes them. Until
    then the bound silently stays at its default forever,
    which looks identical to working.

A SEPARATE CASE STORE, ON PURPOSE
---------------------------------
New cases go to models/llm_case_store.csv, NOT to
models/case_store.csv.

The existing store feeds modules/forecasting/case_based_
correction.py, a correction validated at +0.47 pts leave-one-
day-out. Its cases all come from the main hybrid pipeline.
Pouring LLM-pipeline outcomes into it would change what that
validated correction does, without re-running the experiment
that validated it. Two stores, one purpose each.

NO LOOKAHEAD, EVER
------------------
A block is only scored once its timestamp has passed AND the
meter carries a real measurement for it. Interpolated meter
rows are skipped: scoring a forecast against a number that
was itself interpolated measures the interpolator.
=========================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from utils.logger import get_logger


TIMEZONE = settings["plant"]["timezone"]
CAPACITY_MW = settings["plant"]["capacity_mw"]

LLM_CASE_STORE = Path("models/llm_case_store.csv")

# "SIRMOUR_2026-07-09_14-15_schedule.csv" -> date and time parts.
# Matched explicitly rather than by string surgery: the obvious
# stem.replace("-", ":", 2) rewrites the hyphens in the DATE, not the
# time, and silently yields "2026:07:09 14-15" - which parses as
# nothing, so every case was dropped and the store came out empty.
_RUN_TIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})")


def parse_run_time(filename):
    """The run time encoded in a schedule filename, or None."""

    match = _RUN_TIME_PATTERN.search(filename)

    if not match:
        return None

    date, hour, minute = match.groups()

    try:
        stamp = pd.Timestamp(f"{date} {hour}:{minute}")
    except ValueError:
        return None

    return stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp


class ActualsFeedback:

    def __init__(self, schedule_dir=None):

        self.logger = get_logger()

        self.schedule_dir = Path(
            schedule_dir or settings.get("fusion", {}).get(
                "output_dir", "outputs/llm_schedules"
            )
        )

        self.case_store = LLM_CASE_STORE

    # --------------------------------------------------

    def meter(self):
        """
        Meter history in local time with a real-measurement mask.
        Returns None when there is nothing to score against.
        """

        frame = load_meter_history()

        if frame is None or frame.empty:
            return None

        frame = frame.copy()

        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert(TIMEZONE)

        # Only REAL readings. An interpolated meter row would let a
        # forecast be graded against a guess.
        if "is_real_measurement" in frame.columns:
            frame = frame[frame["is_real_measurement"].fillna(False)]

        frame["actual_mw"] = frame["active_power_kw"] / 1000.0

        return frame[["timestamp", "actual_mw"]].dropna()

    # --------------------------------------------------

    def attach(self, meter):
        """
        Writes `actual_mw` and `deviation_mw` onto every saved schedule
        that can now be scored. Returns a per-file summary.
        """

        rows = []

        for path in sorted(self.schedule_dir.glob("*_schedule.csv")):

            try:
                schedule = pd.read_csv(path, parse_dates=["timestamp"])
            except Exception as error:
                self.logger.warning(f"Could not read {path.name}: {error}")
                continue

            if schedule.empty or "forecast_mw" not in schedule.columns:
                continue

            if schedule["timestamp"].dt.tz is None:
                schedule["timestamp"] = schedule["timestamp"].dt.tz_localize(
                    TIMEZONE
                )
            else:
                schedule["timestamp"] = schedule["timestamp"].dt.tz_convert(
                    TIMEZONE
                )

            # Drop any previous attempt's columns before re-merging, so
            # running this twice does not produce actual_mw_x/_y.
            schedule = schedule.drop(
                columns=["actual_mw", "deviation_mw"], errors="ignore"
            )

            merged = schedule.merge(meter, on="timestamp", how="left")

            scored = merged["actual_mw"].notna()

            if not scored.any():
                continue

            merged["deviation_mw"] = merged["actual_mw"] - merged["forecast_mw"]

            merged.to_csv(path, index=False)

            deviation_pct = float(
                merged.loc[scored, "deviation_mw"].abs().mean()
                / CAPACITY_MW * 100
            )

            rows.append({
                "run": path.name.replace("_schedule.csv", ""),
                "blocks": int(scored.sum()),
                "of": len(merged),
                "deviation_pct": deviation_pct,
                "bias_mw": float(merged.loc[scored, "deviation_mw"].sum()),
            })

            self.logger.info(
                f"Scored {path.name}: {int(scored.sum())}/{len(merged)} blocks, "
                f"{deviation_pct:.2f}% deviation"
            )

        return pd.DataFrame(rows)

    # --------------------------------------------------

    def build_cases(self):
        """
        Scored schedule blocks -> case rows, in the same schema the
        existing case store uses, so one retriever can read either.

        Written to models/llm_case_store.csv - see the module docstring
        on why this does not go into the validated store.
        """

        cases = []

        for path in sorted(self.schedule_dir.glob("*_schedule.csv")):

            try:
                schedule = pd.read_csv(path, parse_dates=["timestamp"])
            except Exception:
                continue

            if "actual_mw" not in schedule.columns:
                continue

            scored = schedule.dropna(subset=["actual_mw"])

            if scored.empty:
                continue

            # The run time is encoded in the filename, and the horizon
            # is measured from it - reading it off the first block
            # instead would make every run look like a 15-minute one.
            run_time = parse_run_time(path.name)

            if run_time is None:
                self.logger.warning(
                    f"Could not read a run time from {path.name} - skipped"
                )
                continue

            for _, row in scored.iterrows():

                horizon = (row["timestamp"] - run_time).total_seconds() / 60

                if horizon <= 0:
                    continue

                cases.append({
                    "block_hour": row["timestamp"].hour
                    + row["timestamp"].minute / 60,
                    "horizon_min": round(horizon),
                    "kt_now": row.get("windy_kt", np.nan),
                    "final_forecast_kw": row["forecast_mw"] * 1000,
                    "residual_kw": (row["actual_mw"] - row["forecast_mw"]) * 1000,
                    "date": row["timestamp"].date(),
                })

        if not cases:
            return pd.DataFrame()

        frame = pd.DataFrame(cases)

        # Rebuilt from the schedules every time rather than appended.
        # Appending would double every case on a second run, and the
        # schedules are the source of truth anyway.
        self.case_store.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.case_store, index=False)

        self.logger.info(
            f"LLM case store rebuilt: {len(frame)} case(s) -> {self.case_store}"
        )

        return frame

    # --------------------------------------------------

    def recent_bias(self, summary, days=5):
        """
        Daily (actual - scheduled) totals for the most recent finished
        days, oldest first - the sequence the validator reads to decide
        how far the model may stray from the anchor.
        """

        if summary.empty:
            return []

        summary = summary.copy()

        summary["day"] = summary["run"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]

        daily = summary.groupby("day")["bias_mw"].sum().sort_index()

        return list(daily.tail(days).to_numpy())

    # --------------------------------------------------

    def run(self):

        meter = self.meter()

        if meter is None:
            raise SystemExit(
                "No meter history in data/historical - nothing to feed back."
            )

        if not self.schedule_dir.exists():
            raise SystemExit(f"No schedules at {self.schedule_dir}")

        summary = self.attach(meter)

        if summary.empty:
            print(
                "No saved schedule has scoreable blocks yet.\n"
                "Blocks are scored once their time has passed AND the meter "
                "carries a real measurement for them."
            )
            return summary

        cases = self.build_cases()

        print("=" * 74)
        print("DAILY ACTUALS FEEDBACK")
        print("=" * 74)
        print(summary.to_string(index=False, float_format="%.2f"))
        print("-" * 74)
        print(f"runs scored : {len(summary)}")
        print(f"mean dev    : {summary['deviation_pct'].mean():.2f}% of capacity")
        print(f"cases built : {len(cases)}")

        bias = self.recent_bias(summary)

        if bias:

            from modules.fusion.validator import ScheduleValidator

            allowed = ScheduleValidator().suggested_max_deviation_fraction(bias)

            print(f"\nrecent daily bias (MW-blocks, oldest first): "
                  f"{[round(v, 2) for v in bias]}")
            print(f"validator will allow deviation up to {allowed:.0%} of anchor")

            if allowed > 0.40:
                print("  (widened - recent days were biased consistently one way)")

        return summary


# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Attach actual generation to saved schedules and "
                    "rebuild the LLM case store"
    )
    parser.add_argument("--schedule-dir", default=None)

    args = parser.parse_args()

    ActualsFeedback(args.schedule_dir).run()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
