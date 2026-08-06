"""
=========================================================
Solar Forecasting Project
Block Bias Correction (time-of-day shape)
=========================================================
MENTOR GUIDANCE 2026-08-06: "analyze the pattern of the
results for the last 4-5 days ... identify the pattern
among the blocks causing higher penalties and convey the
same to the model."

This is the "convey to the model" half. The analysis half
is tests/test_block_penalty_pattern.py, and what it found
on the last 5 finished days was:

  * every rupee of DSM penalty lands in blocks 40-66
    (09:45-16:30). Outside that window the plant is small
    enough that any miss stays inside the free +/-10%
    dead band, so those blocks cannot be penalised.
  * inside it the schedule carries a repeating TIME-OF-DAY
    SHAPE, not a flat level error: the late morning is
    over-forecast and the mid-afternoon under-forecast on
    most days.

So the correction learns one number per 15-min block -
the median (actual - scheduled) over the last N finished
days - and shifts that block's forecast by half of it.

Three deliberate design choices, each one forced by a
negative result in tests/test_block_bias_experiment.py:

  1. SMOOTHED over +/-6 blocks (+/-90 min). A raw per-block
     median off 5 days is 5 samples of a noisy quantity;
     using it unsmoothed made the penalty WORSE at every
     strength above 0.5. The recoverable pattern is a
     broad shape, not a per-block offset.
  2. STRENGTH 0.5 - apply half. Same lesson as the
     case-based corrector: the history hints at a bias, it
     does not dictate a value.
  3. A FLAT whole-day shift (the obvious simpler version)
     was tested as a control and HURT. The information is
     in the shape across blocks, which is exactly what the
     mentor pointed at.

The profile is rebuilt at run time from the most recent
finished day schedules rather than saved to a file on a
good day and then trusted forever - the case store went
stale that way (built 2026-07-24, still serving Jul 6-22
analogues weeks later). It refuses to run when the newest
day available is older than `max_profile_age_days`.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings


BLOCKS = range(1, 97)


def block_number(timestamp):
    """Indian scheduling block number - block 1 is 00:00-00:15."""

    return timestamp.dt.hour * 4 + timestamp.dt.minute // 15 + 1


class BlockBiasCorrector:

    def __init__(self, lookback_days=None, strength=None, smooth_blocks=None):

        cfg = settings.get("block_bias_correction", {})

        self.enabled = cfg.get("enabled", False)

        self.lookback_days = lookback_days or cfg.get("lookback_days", 5)
        self.smooth_blocks = (
            smooth_blocks if smooth_blocks is not None
            else cfg.get("smooth_blocks", 6)
        )
        self.strength = (
            strength if strength is not None else cfg.get("strength", 0.5)
        )

        self.min_days = cfg.get("min_days", 4)
        self.min_block_samples = cfg.get("min_block_samples", 2)
        self.max_profile_age_days = cfg.get("max_profile_age_days", 3)
        self.max_shift_mw = cfg.get("max_shift_mw", 1.0)

        self.schedule_dir = Path(cfg.get("schedule_dir", "outputs/schedules"))

        self.capacity_kw = settings["plant"]["capacity_mw"] * 1000

        self._profile_kw = None      # Series indexed by block 1..96
        self._days_used = []

    # --------------------------------------------------

    def find_history(self, as_of=None):
        """
        The most recent finished reconstructed day schedules
        strictly BEFORE as_of - the same files the daily report
        pipeline already writes. Returns [(date_string, frame)],
        oldest first.
        """

        found = []

        for path in sorted(self.schedule_dir.glob("day_schedule_*.csv")):

            # day_schedule_<date>.csv only - not the _per_run /
            # _summary / _run_log siblings.
            stem = path.stem.replace("day_schedule_", "")
            if "_" in stem:
                continue

            try:
                day = pd.Timestamp(stem).date()
            except ValueError:
                continue

            if as_of is not None and day >= pd.Timestamp(as_of).date():
                continue

            found.append((day, path))

        found.sort()

        recent = found[-self.lookback_days:]

        frames = []
        for day, path in recent:

            frame = pd.read_csv(path)

            if not {"block", "scheduled_mw", "actual_mw"} <= set(frame.columns):
                continue

            if "actual_is_real" in frame.columns:
                frame = frame[frame["actual_is_real"].astype(bool)]

            frame = frame.dropna(subset=["actual_mw", "scheduled_mw"])

            if frame.empty:
                continue

            frames.append((day, frame))

        return frames

    # --------------------------------------------------

    def learn_profile(self, frames):
        """
        One bias number per block: the median of
        (actual - scheduled) across the history, smoothed over
        +/-smooth_blocks, in kW.

        Blocks seen on fewer than `min_block_samples` days are
        zeroed AFTER smoothing - otherwise the smoother bleeds a
        dawn block's bias into night blocks that have no data of
        their own and the forecast stops going to zero at night.
        """

        history = pd.concat(
            [frame.assign(_day=day) for day, frame in frames],
            ignore_index=True,
        )

        history["bias_mw"] = history["actual_mw"] - history["scheduled_mw"]

        median = (
            history.groupby("block")["bias_mw"].median().reindex(BLOCKS)
        )
        counts = (
            history.groupby("block")["bias_mw"].size().reindex(BLOCKS).fillna(0)
        )

        if self.smooth_blocks:
            median = median.rolling(
                2 * self.smooth_blocks + 1, center=True, min_periods=1
            ).mean()

        median = median.where(counts >= self.min_block_samples, 0.0).fillna(0.0)

        median = median.clip(-self.max_shift_mw, self.max_shift_mw)

        return median * 1000.0

    # --------------------------------------------------

    def load(self, as_of=None):
        """
        Builds the profile from the days available before as_of.
        Leaves the profile as None when there is not enough
        history - `available` then reports False.
        """

        frames = self.find_history(as_of)

        if len(frames) < self.min_days:
            self._profile_kw = None
            self._days_used = []
            return self

        newest = frames[-1][0]

        reference = pd.Timestamp(as_of).date() if as_of else pd.Timestamp.now().date()

        if (reference - newest).days > self.max_profile_age_days:
            # Stale history teaches a stale shape - refuse rather
            # than quietly correct today with last month's pattern.
            self._profile_kw = None
            self._days_used = []
            return self

        self._profile_kw = self.learn_profile(frames)
        self._days_used = [day for day, _ in frames]

        return self

    # --------------------------------------------------

    @property
    def available(self):

        return self.enabled and self._profile_kw is not None

    # --------------------------------------------------

    @property
    def days_used(self):

        return list(self._days_used)

    # --------------------------------------------------

    def apply(self, forecast):
        """
        Returns the forecast with final_forecast_kw shifted by
        strength x the learned per-block bias. The uncorrected
        value is preserved for transparency, same convention as
        the other two correctors.
        """

        if self._profile_kw is None:
            return forecast

        blocks = block_number(forecast["timestamp"])
        shift = blocks.map(self._profile_kw).fillna(0.0).to_numpy()

        forecast = forecast.copy()

        if "final_forecast_uncorrected_kw" not in forecast.columns:
            forecast["final_forecast_uncorrected_kw"] = forecast["final_forecast_kw"]

        forecast["block_bias_shift_kw"] = self.strength * shift

        forecast["final_forecast_kw"] = np.clip(
            forecast["final_forecast_kw"] + self.strength * shift,
            0,
            self.capacity_kw,
        )

        return forecast
