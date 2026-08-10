"""
=========================================================
Solar Forecasting Project - NEW APPROACH
How well has each input actually been predicting?
=========================================================
Scores every input SEPARATELY against real generation over
recent finished days, and hands the model those scores so it
can weigh the inputs on evidence instead of on impression.

WHY ASKING THE MODEL TO SELF-WEIGHT DOES NOT WORK
-------------------------------------------------
The obvious version of "let it decide which input to trust"
is to ask it: "which did you rely on, and how much?" That
produces a confident answer and no information. A model has
no way to know that Windy was 22% wrong last Tuesday - it
cannot see last Tuesday. It would be reporting a feeling.

This measures it instead. Each input is turned into a
standalone forecast, priced against what the plant really
did, and the resulting table goes into the prompt:

    over the last 5 finished days, at this time of day:
      meter persistence   8.9% average error   best on 4 of 5 days
      Windy forecast     14.2%                 best on 1 of 5 days
      weather forecast   12.6%                 best on 0 of 5 days

Now "rely more on the meter" is a fact the model can read,
not a preference someone hard-coded.

WHY THIS PARTICULAR FIX, NOW
----------------------------
The 2026-08-09 scoring found the model was overriding the
one input that was working. Anchor alone cost Rs 17,411 over
12 days; the model cost Rs 27,663, and the blend sweep was
monotonic - every gram of weight given to the model cost
money. It was not short of inputs. It was short of any way
to know which one deserved belief.

NO LOOKAHEAD
------------
Only days strictly BEFORE the run day are scored, and only
blocks with a real measurement. A score that included today
would be telling the model the answer.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from utils.logger import get_logger


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]

# Column in the saved schedules -> how to describe it to the model.
# Each is a standalone forecast in MW for the same blocks, so they are
# directly comparable to one another and to actual generation.
INPUTS = {
    "anchor_mw": "meter persistence (last measured cloudiness, damped)",
    "windy_mw": "Windy's own solar forecast",
    "weather_mw": "ECMWF weather forecast (bias-corrected)",
    "llm_raw_mw": "the model's own unblended call",
}


class InputSkill:

    def __init__(self, schedule_dir=None, lookback_days=5):

        self.logger = get_logger()

        self.schedule_dir = Path(
            schedule_dir or settings.get("fusion", {}).get(
                "output_dir", "outputs/llm_schedules"
            )
        )

        self.lookback_days = lookback_days

    # --------------------------------------------------

    def gather(self, before_date):
        """
        Every scored block from finished days before `before_date`, with
        each input's standalone forecast beside the actual.
        """

        from tests.score_day_schedules import run_time_of

        frames = []

        for path in sorted(self.schedule_dir.glob("*_schedule.csv")):

            run_time = run_time_of(path)

            if run_time is None or run_time.date() >= before_date:
                continue

            try:
                frame = pd.read_csv(path, parse_dates=["timestamp"])
            except Exception:
                continue

            if "actual_mw" not in frame.columns:
                continue      # not yet scored by actuals_feedback

            frame = frame.dropna(subset=["actual_mw"])

            if frame.empty:
                continue

            # Windy and weather are stored as clear-sky indices; turn
            # them into MW so every input is on the same scale as the
            # actual generation it is being judged against.
            clearsky = frame.get("clearsky_power_mw")

            if clearsky is not None:
                for source, target in (
                    ("windy_kt", "windy_mw"), ("weather_kt", "weather_mw")
                ):
                    if source in frame.columns:
                        frame[target] = np.clip(
                            frame[source] * clearsky, 0, CAPACITY_MW
                        )

            frame["day"] = run_time.date()

            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # --------------------------------------------------

    def score(self, before_date):
        """
        {column: {error_pct, days_best, days}} over the lookback window.
        Inputs with no usable values are simply absent.
        """

        blocks = self.gather(before_date)

        if blocks.empty:
            return {}

        days = sorted(blocks["day"].unique())[-self.lookback_days:]
        blocks = blocks[blocks["day"].isin(days)]

        available = [
            column for column in INPUTS
            if column in blocks.columns and blocks[column].notna().any()
        ]

        if not available:
            return {}

        result = {}
        per_day = {}

        for column in available:

            usable = blocks.dropna(subset=[column])

            if usable.empty:
                continue

            error = (
                (usable[column] - usable["actual_mw"]).abs()
                / CAPACITY_MW * 100
            )

            result[column] = {
                "error_pct": float(error.mean()),
                "blocks": int(len(usable)),
                "days": len(days),
            }

            per_day[column] = (
                usable.assign(err=error).groupby("day")["err"].mean()
            )

        # Which input was closest on each day - a mean can be dragged by
        # one bad afternoon, and "best on 4 of 5 days" is the more
        # honest summary of whether an input is dependable.
        if per_day:

            table = pd.DataFrame(per_day)
            winners = table.idxmin(axis=1).value_counts()

            for column in result:
                result[column]["days_best"] = int(winners.get(column, 0))

        return result

    # --------------------------------------------------

    def prompt_section(self, run_time):
        """
        The track record, as prompt text. Returns None when there is not
        enough scored history yet - in which case the prompt should say
        nothing rather than imply a record that does not exist.
        """

        before = pd.Timestamp(run_time).date()

        scores = self.score(before)

        if not scores:
            return None

        ranked = sorted(scores.items(), key=lambda item: item[1]["error_pct"])

        days = ranked[0][1]["days"]

        lines = [
            f"How each input has actually performed over the last {days} "
            "finished day(s), measured against real generation:"
        ]

        for column, score in ranked:

            lines.append(
                f"  {INPUTS[column]:52s} "
                f"{score['error_pct']:5.1f}% average error, "
                f"closest on {score.get('days_best', 0)} of {days} day(s)"
            )

        best = INPUTS[ranked[0][0]]

        lines.append(
            f"\nOn this record {best} has been the most reliable. That is a "
            "measurement, not a rule - weigh it against what today actually "
            "looks like, and if you depart from the best-performing input, "
            "say which evidence made you."
        )

        return "\n".join(lines)


# --------------------------------------------------

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="How well has each input been predicting lately?"
    )
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--folder", default=None)

    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()

    skill = InputSkill(args.folder, args.lookback)

    section = skill.prompt_section(as_of)

    if section is None:
        print(
            "Not enough scored history yet.\n"
            "Run:  python -m modules.evaluation.actuals_feedback\n"
            "which attaches actual generation to the saved schedules."
        )
        return 1

    print("=" * 78)
    print(f"INPUT TRACK RECORD as of {as_of:%Y-%m-%d}")
    print("=" * 78)
    print(section)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
