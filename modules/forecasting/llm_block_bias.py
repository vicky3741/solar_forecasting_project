"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Block bias correction, learned from this pipeline's own days
=========================================================
Shifts each 15-minute block by `strength` x the median
(actual - scheduled) that block showed over recent finished
days.

Mentor guidance 2026-08-06: "analyze the pattern of the
results for the last 4-5 days ... identify the pattern among
the blocks causing higher penalties & convey the same to the
model."

WHY A SEPARATE MODULE FROM THE PRODUCTION ONE
---------------------------------------------
modules/forecasting/block_bias_correction.py learns from
outputs/schedules/day_schedule_<date>.csv, which the OLD
pipeline writes. This pipeline's history lives in its own
saved run schedules, and its errors are its own - a profile
learned from the production blend would be correcting a
different model's mistakes.

STRENGTH 0.25, NOT THE PRODUCTION 0.5
-------------------------------------
Measured on this pipeline's own 12 days with
tests/whatif_correction.py, walk-forward:

    strength   0.00    0.10    0.20   0.25    0.35    0.50    1.00
    penalty  27,663  27,457  27,376 27,371  27,426  27,709  30,363

A clean minimum at 0.25 - saving Rs 291 over 12 days - and
monotonic harm beyond it. Production uses 0.5 because the
production blend's error has a steadier time-of-day shape;
this model's error is more erratic, so less of it is
recoverable and over-correcting costs money quickly.

Keep it in proportion: Rs 291 against a total of Rs 27,663 is
about 1%, while the model as a whole is Rs 10,252 worse than
publishing the anchor alone. This is a real improvement to a
component that is not yet the problem.

SMOOTHING IS NOT COSMETIC
-------------------------
A raw per-block median off five days is five noisy samples.
Unsmoothed it made the production penalty WORSE. The
recoverable pattern is a broad time-of-day shape, not a
per-block offset.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.scheduling.effective_time import block_number
from utils.logger import get_logger


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]


class LLMBlockBias:

    def __init__(self, schedule_dir=None):

        self.logger = get_logger()

        fusion = settings.get("fusion", {})
        cfg = settings.get("block_bias_correction", {})

        self.enabled = fusion.get("block_bias", True)

        # Measured on this pipeline, not inherited from production.
        self.strength = float(fusion.get("block_bias_strength", 0.25))

        self.lookback_days = cfg.get("lookback_days", 5)
        self.smooth_blocks = cfg.get("smooth_blocks", 6)
        self.min_days = cfg.get("min_days", 4)
        self.min_block_samples = cfg.get("min_block_samples", 2)
        self.max_shift_mw = cfg.get("max_shift_mw", 1.0)

        self.schedule_dir = Path(
            schedule_dir or fusion.get("output_dir", "outputs/llm_schedules")
        )

        self.profile = {}
        self.days_used = 0

    # --------------------------------------------------

    def history(self, before_date):
        """
        Scored blocks from finished days strictly BEFORE before_date,
        newest `lookback_days` only.

        Reads the saved schedules directly - actuals_feedback has
        already written actual_mw into them - so this needs no separate
        day-schedule reconstruction.
        """

        from tests.score_day_schedules import run_time_of

        by_day = {}

        for path in sorted(self.schedule_dir.glob("*_schedule.csv")):

            run_time = run_time_of(path)

            if run_time is None or run_time.date() >= before_date:
                continue

            try:
                frame = pd.read_csv(path, parse_dates=["timestamp"])
            except Exception:
                continue

            if not {"forecast_mw", "actual_mw"} <= set(frame.columns):
                continue

            frame = frame.dropna(subset=["actual_mw", "forecast_mw"])

            if frame.empty:
                continue

            if frame["timestamp"].dt.tz is None:
                frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)

            frame = frame.assign(block=frame["timestamp"].map(block_number))

            by_day.setdefault(run_time.date(), []).append(
                frame[["block", "forecast_mw", "actual_mw"]]
            )

        days = sorted(by_day)[-self.lookback_days:]

        return [pd.concat(by_day[day], ignore_index=True) for day in days]

    # --------------------------------------------------

    def load(self, as_of):
        """
        Learns the profile for a run on `as_of`. Leaves it empty - and
        therefore inert - when there is not enough history.
        """

        self.profile = {}
        self.days_used = 0

        if not self.enabled:
            return

        frames = self.history(pd.Timestamp(as_of).date())

        if len(frames) < self.min_days:
            self.logger.info(
                f"Block bias: only {len(frames)} finished day(s) of history, "
                f"need {self.min_days} - not correcting"
            )
            return

        combined = pd.concat(frames, ignore_index=True)
        combined["error_mw"] = combined["actual_mw"] - combined["forecast_mw"]

        median = combined.groupby("block")["error_mw"].median()
        counts = combined.groupby("block")["error_mw"].size()

        # A block seen on fewer days than this is guesswork; drop it
        # BEFORE smoothing so it cannot bleed into its neighbours.
        median = median[counts >= self.min_block_samples].sort_index()

        if median.empty:
            return

        if self.smooth_blocks:
            median = median.rolling(
                2 * self.smooth_blocks + 1, center=True, min_periods=1
            ).mean()

        self.profile = median.clip(
            -self.max_shift_mw, self.max_shift_mw
        ).to_dict()

        self.days_used = len(frames)

        self.logger.info(
            f"Block bias profile learned from {self.days_used} day(s), "
            f"{len(self.profile)} block(s), strength {self.strength}"
        )

    # --------------------------------------------------

    @property
    def available(self):

        return bool(self.profile)

    # --------------------------------------------------

    def apply(self, schedule, column="forecast_mw"):
        """
        Applies the learned shift. Returns the schedule unchanged when
        no profile was learned, so a caller never has to check first.
        """

        if not self.available:
            return schedule

        schedule = schedule.copy()

        blocks = (
            schedule["block"] if "block" in schedule.columns
            else schedule["timestamp"].map(block_number)
        )

        shift = blocks.map(self.profile).fillna(0.0).to_numpy()

        schedule["block_bias_shift_mw"] = self.strength * shift

        schedule[column] = np.clip(
            schedule[column].to_numpy() + self.strength * shift,
            0.0, CAPACITY_MW,
        )

        return schedule
