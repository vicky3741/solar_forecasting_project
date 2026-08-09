"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Schedule Validator
=========================================================
Deterministic safety checks on whatever the LLM returns,
adopted from validator.py in
github.com/Kushal70-51/Windy-Project-3.

This is the single most valuable thing in their repo that we
did not have. Our LLM output was completely unbounded, and
we have measured evidence that matters: on the first live run
(2026-08-09 16:45) Gemini took Windy's forecast clear-sky
index of 0.88 and replaced it with 0.20 - a 4x cut - on the
strength of a rain figure. That call might be right. It might
also be an afternoon of under-forecasting and DSM penalty.
Nothing in the pipeline could tell the difference, and
nothing stopped it.

THREE CHECKS
------------
1. RANGE      every block within [0, capacity].
2. DEVIATION  a block may not move more than
              `max_deviation_fraction` away from the physics
              anchor. Theirs is 0.40. The model keeps its
              judgement; it just cannot quadruple or quarter
              a block on a hunch.
3. SMOOTHNESS consecutive blocks may not differ by more than
              `max_step_change_mw`. Solar generation does not
              jump the width of the plant in 15 minutes, so a
              step that large is a model artefact, not weather.

EARNED TRUST
------------
`suggested_max_deviation_fraction` widens the deviation cap
when the recent record justifies it: if several finished days
in a row were biased the SAME direction, the anchor itself is
off, and holding the model tightly to a wrong anchor just
copies the anchor's error. Their thresholds: 3+ consistent
days -> 0.80, 2 days -> 0.60, otherwise the default.

Every adjustment is recorded rather than applied silently.
A validator that quietly rewrites forecasts is indistinguish-
able from a bug, and the reason a block moved has to survive
into the report.
=========================================================
"""

import numpy as np
import pandas as pd

from config.config import settings
from utils.logger import get_logger


class ScheduleValidator:

    def __init__(self):

        self.logger = get_logger()

        self.capacity_mw = settings["plant"]["capacity_mw"]

        rules = settings.get("fusion", {}).get("validator", {})

        self.enabled = rules.get("enabled", True)

        self.max_deviation_fraction = rules.get("max_deviation_fraction", 0.40)

        # Their MAX_STEP_CHANGE_MW is capacity * 0.35. Expressed as a
        # fraction here so it carries across the three plants (5.1, 15
        # and 10 MW) without a per-plant constant.
        self.max_step_fraction = rules.get("max_step_fraction", 0.35)

        self.consistent_bias_days = rules.get("consistent_bias_days", 3)
        self.widened_deviation = rules.get("widened_deviation", 0.80)
        self.partly_widened_deviation = rules.get("partly_widened_deviation", 0.60)

    # --------------------------------------------------

    @property
    def max_step_change_mw(self):

        return self.capacity_mw * self.max_step_fraction

    # --------------------------------------------------

    def suggested_max_deviation_fraction(self, recent_bias):
        """
        How far the model may stray from the anchor, given how the
        anchor has recently behaved.

        `recent_bias` is an ordered sequence of (actual - scheduled)
        signs or values for recent finished days, most recent last.
        Consistent one-way bias means the anchor is systematically
        wrong, so the model is allowed more room to correct it.
        """

        if not recent_bias:
            return self.max_deviation_fraction

        signs = [np.sign(value) for value in recent_bias if value]

        if not signs:
            return self.max_deviation_fraction

        # How many days at the END of the run share one direction.
        streak = 1

        for previous, current in zip(reversed(signs), list(reversed(signs))[1:]):
            if current == previous:
                streak += 1
            else:
                break

        if streak >= self.consistent_bias_days:
            return self.widened_deviation

        if streak >= 2:
            return self.partly_widened_deviation

        return self.max_deviation_fraction

    # --------------------------------------------------

    def validate(self, schedule, anchor_column="anchor_mw",
                 value_column="forecast_mw", max_deviation_fraction=None):
        """
        Applies the three checks in order and returns
        (schedule, notes).

        The schedule gains two columns - `was_adjusted` and
        `adjustment_note` - so a report can say exactly which blocks
        the validator touched and why.
        """

        schedule = schedule.copy()

        schedule["was_adjusted"] = False
        schedule["adjustment_note"] = ""

        if not self.enabled:
            return schedule, []

        if value_column not in schedule.columns:
            raise KeyError(f"schedule has no '{value_column}' column")

        limit = (
            self.max_deviation_fraction if max_deviation_fraction is None
            else max_deviation_fraction
        )

        values = schedule[value_column].to_numpy(dtype=float)
        notes = []

        # --- 1. range ---
        out_of_range = (values < 0) | (values > self.capacity_mw)

        if out_of_range.any():
            for index in np.flatnonzero(out_of_range):
                notes.append(
                    f"block {int(schedule['block'].iloc[index])}: "
                    f"{values[index]:.3f} MW outside [0, {self.capacity_mw}]"
                )
            values = np.clip(values, 0.0, self.capacity_mw)
            schedule.loc[out_of_range, "was_adjusted"] = True
            schedule.loc[out_of_range, "adjustment_note"] = "clipped to capacity"

        # --- 2. deviation from the physics anchor ---
        if anchor_column in schedule.columns:

            anchor = schedule[anchor_column].to_numpy(dtype=float)

            # Only where the anchor is meaningfully non-zero. Near
            # sunrise and sunset the anchor is ~0, and a fractional
            # bound on ~0 would pin every dawn block to zero.
            live = anchor > 0.05 * self.capacity_mw

            lower = anchor * (1 - limit)
            upper = anchor * (1 + limit)

            too_far = live & ((values < lower) | (values > upper))

            if too_far.any():

                for index in np.flatnonzero(too_far):
                    notes.append(
                        f"block {int(schedule['block'].iloc[index])}: "
                        f"{values[index]:.3f} MW is more than {limit:.0%} from "
                        f"anchor {anchor[index]:.3f} MW - pulled back"
                    )

                values = np.where(
                    too_far, np.clip(values, lower, upper), values
                )

                schedule.loc[too_far, "was_adjusted"] = True
                schedule.loc[too_far, "adjustment_note"] = (
                    f"pulled back to {limit:.0%} of anchor"
                )

        # --- 3. smoothness between consecutive blocks ---
        step_limit = self.max_step_change_mw

        for index in range(1, len(values)):

            change = values[index] - values[index - 1]

            if abs(change) <= step_limit:
                continue

            capped = values[index - 1] + np.sign(change) * step_limit

            notes.append(
                f"block {int(schedule['block'].iloc[index])}: jump of "
                f"{change:+.3f} MW exceeds {step_limit:.3f} MW - smoothed to "
                f"{capped:.3f} MW"
            )

            values[index] = capped

            schedule.iloc[
                index, schedule.columns.get_loc("was_adjusted")
            ] = True
            schedule.iloc[
                index, schedule.columns.get_loc("adjustment_note")
            ] = "smoothed"

        schedule[value_column] = values

        if notes:
            self.logger.info(
                f"Validator adjusted {int(schedule['was_adjusted'].sum())} of "
                f"{len(schedule)} block(s)"
            )
            for note in notes[:8]:
                self.logger.info(f"  {note}")

        return schedule, notes


# --------------------------------------------------

def recent_daily_bias(schedule_dir, lookback_days=5, as_of=None):
    """
    (actual - scheduled) totals for recent finished days, oldest first,
    read from the reconstructed day schedules the existing pipeline
    already writes.

    Used to decide whether the deviation cap should be widened. Returns
    [] when there is not enough history, which leaves the cap at its
    default - the safe direction.
    """

    from pathlib import Path

    folder = Path(schedule_dir)

    if not folder.exists():
        return []

    as_of = pd.Timestamp(as_of).date() if as_of else pd.Timestamp.now().date()

    biases = []

    for path in sorted(folder.glob("*_schedule.csv")):

        try:
            frame = pd.read_csv(path, parse_dates=["timestamp"])
        except Exception:
            continue

        if "actual_mw" not in frame.columns or "forecast_mw" not in frame.columns:
            continue

        day = frame["timestamp"].dt.date.iloc[0]

        if day >= as_of:
            continue      # today is not finished; it cannot vote

        matched = frame.dropna(subset=["actual_mw", "forecast_mw"])

        if matched.empty:
            continue

        biases.append((
            day,
            float((matched["actual_mw"] - matched["forecast_mw"]).sum()),
        ))

    biases.sort()

    return [value for _, value in biases[-lookback_days:]]
