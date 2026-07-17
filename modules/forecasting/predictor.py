"""
=========================================================
Solar Forecasting Project
Hybrid Predictor
=========================================================
Combines multiple forecasting signals into one final
generation forecast for every 15-minute block from the
latest completed reading up to the forecast end time.

Everything works in clear-sky index (kt) space, not raw
power - kt = actual GHI / clear-sky GHI, a value with no
daily shape of its own (it does not know about sunrise or
sunset). The exact sunrise/sunset/seasonal shape always
comes from the clear-sky curve, which is exact physics.
The forecasting signals only ever have to answer "is cloud
cover trending better or worse", which is what they are
actually good at:

  1. Chronos      - a pretrained zero-shot AI model that
                    reads the plant's kt history as a plain
                    sequence and projects it forward.
  2. Persistence  - starts from today's most recent kt and
                    extrapolates its recent slope forward
                    with damping ("smart persistence": if
                    clouds have been clearing for the last
                    hour, assume that continues briefly
                    rather than freezing conditions flat).
  3. Vision       - an optional per-block adjustment profile
                    from the Windy cloud-video analysis,
                    phased in at the predicted cloud
                    arrival/clearing time.

Blending in kt-space (rather than blending raw power
forecasts) guarantees the final forecast always goes to
zero at night regardless of what any signal predicts, since
the clear-sky multiplier is applied after the blend.

`compute_signals` runs the expensive part (Chronos) once.
`blend_signals` is pure numpy and safe to call many times
with different chronos_weight / performance_ratio candidates
- this is what makes parameter tuning fast, since it does
not require re-running Chronos for every candidate.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.chronos_model import ChronosModel
from modules.forecasting.clearsky import ClearSkyModel


class HybridPredictor:

    def __init__(self):

        self.chronos = ChronosModel()

        self.clearsky = ClearSkyModel()

        self.capacity_kw = settings["plant"]["capacity_mw"] * 1000

        self.interval_minutes = settings["forecast"]["interval_minutes"]
        self.forecast_end_time = settings["forecast"]["forecast_end_time"]

        blend_settings = settings["hybrid_blend"]

        self.chronos_weight = blend_settings["chronos_weight"]
        self.min_context_points = blend_settings["min_context_points"]
        self.trend_halflife_min = blend_settings.get("trend_halflife_min", 90)

        self.performance_ratio = settings["clearsky"]["performance_ratio"]

        # Clear-sky index rarely exceeds ~1.2 even under cloud-enhancement
        # effects; clip any forecasted kt to a sane physical range.
        self.max_clear_sky_index = 1.2

    # --------------------------------------------------

    def build_forecast_timestamps(self, run_time):
        """
        Builds the list of 15-minute-block timestamps from
        just after run_time up to the forecast end time,
        on the same calendar day as run_time.
        """

        end_hour, end_minute = map(
            int,
            self.forecast_end_time.split(":")
        )

        end_time = run_time.replace(
            hour=end_hour,
            minute=end_minute,
            second=0,
            microsecond=0
        )

        step = timedelta(minutes=self.interval_minutes)

        timestamp = run_time.replace(second=0, microsecond=0) + step

        timestamps = []

        while timestamp <= end_time:
            timestamps.append(timestamp)
            timestamp += step

        return pd.DatetimeIndex(timestamps)

    # --------------------------------------------------

    def get_clear_sky_index_history(self, dataframe, run_time):
        """
        Clear-sky index (kt) for every available reading up
        to run_time, across all historical days. This is the
        context series Chronos forecasts forward - a bounded,
        mostly-stationary weather signal, unlike raw power
        which has a strong sunrise/sunset shape Chronos has
        no way to learn from a handful of irregular days.
        """

        history = dataframe[dataframe["timestamp"] <= run_time]

        with_kt = self.clearsky.compute_clear_sky_index(
            history,
            ghi_column="ghi_w_m2",
            timestamp_column="timestamp"
        )

        return with_kt["clear_sky_index"].dropna().to_numpy()

    # --------------------------------------------------

    def get_today_kt_series(self, dataframe, run_time):
        """
        Today's valid daylight clear-sky index readings up to
        run_time, with their timestamps.
        """

        today = dataframe[
            dataframe["timestamp"].dt.date == run_time.date()
        ]

        today = today[today["timestamp"] <= run_time]

        with_kt = self.clearsky.compute_clear_sky_index(
            today,
            ghi_column="ghi_w_m2",
            timestamp_column="timestamp"
        )

        return with_kt[["timestamp", "clear_sky_index"]].dropna()

    # --------------------------------------------------

    def get_current_clear_sky_index(self, kt_series):
        """
        Today's current cloud attenuation: the mean of the
        most recent valid daylight kt readings.

        Defaults to 1.0 (clear-sky assumption) when no valid
        daylight reading exists yet, e.g. before sunrise on
        the first run of the day.
        """

        recent_kt = kt_series["clear_sky_index"].tail(4)

        if recent_kt.empty:
            return 1.0

        return float(recent_kt.mean())

    # --------------------------------------------------

    def get_kt_slope_per_minute(self, kt_series):
        """
        Per-minute slope of today's recent kt readings - the
        "is the sky currently clearing or clouding over"
        signal that smart persistence extrapolates forward.
        Clamped to a physically-sane range so one bad sensor
        reading cannot swing the whole afternoon.
        """

        recent = kt_series.tail(6)

        if len(recent) < 3:
            return 0.0

        minutes = (
            recent["timestamp"] - recent["timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy() / 60

        slope = np.polyfit(
            minutes,
            recent["clear_sky_index"].to_numpy(),
            1
        )[0]

        return float(np.clip(slope, -0.01, 0.01))

    # --------------------------------------------------

    def get_chronos_kt_forecast(self, dataframe, run_time, prediction_length):
        """
        Forecasts the clear-sky index forward using Chronos.
        """

        kt_context = self.get_clear_sky_index_history(
            dataframe,
            run_time
        )

        if len(kt_context) == 0:
            # No prior kt readings at all (e.g. the very first run of the
            # very first day in the dataset) - nothing for Chronos to
            # extrapolate from, so fall back to a clear-sky assumption.
            return np.full(prediction_length, 1.0)

        result = self.chronos.forecast(
            context_values=kt_context,
            prediction_length=prediction_length
        )

        return np.clip(result["mean"], 0, self.max_clear_sky_index)

    # --------------------------------------------------

    def get_context_ratio(self, dataframe, run_time):
        """
        How much of today has been observed so far, as a
        fraction of `min_context_points`, capped at 1.0.
        Used to lean on persistence early in the day and on
        Chronos once enough of today's data has come in.
        """

        points_today = (
            (dataframe["timestamp"].dt.date == run_time.date())
            & (dataframe["timestamp"] <= run_time)
        ).sum()

        return min(1.0, points_today / self.min_context_points)

    # --------------------------------------------------

    def compute_signals(self, dataframe, run_time):
        """
        Runs the expensive, parameter-independent part of the
        forecast once: Chronos's kt projection, today's
        persisted kt and its recent slope, the context ratio,
        the forecast horizon in minutes per block, and the
        raw POA clear-sky curve (before capacity/performance-
        ratio scaling, so different performance ratios can be
        applied later without recomputing solar position).
        """

        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

        forecast_timestamps = self.build_forecast_timestamps(run_time)

        prediction_length = len(forecast_timestamps)

        horizon_minutes = (
            (forecast_timestamps - run_time).total_seconds().to_numpy() / 60
        )

        poa_curve = self.clearsky.get_poa_irradiance(
            forecast_timestamps
        )["poa_global"].to_numpy()

        chronos_kt_forecast = self.get_chronos_kt_forecast(
            dataframe,
            run_time,
            prediction_length
        )

        kt_series = self.get_today_kt_series(dataframe, run_time)

        kt_now = self.get_current_clear_sky_index(kt_series)

        kt_slope_per_min = self.get_kt_slope_per_minute(kt_series)

        context_ratio = self.get_context_ratio(dataframe, run_time)

        return {
            "forecast_timestamps": forecast_timestamps,
            "horizon_minutes": horizon_minutes,
            "chronos_kt_forecast": chronos_kt_forecast,
            "kt_now": kt_now,
            "kt_slope_per_min": kt_slope_per_min,
            "poa_curve": poa_curve,
            "context_ratio": context_ratio
        }

    # --------------------------------------------------

    def blend_signals(self,
                      signals,
                      chronos_weight=None,
                      performance_ratio=None,
                      trend_halflife_min=None,
                      vision_adjustment=0.0):
        """
        Cheap, pure-numpy blend of already-computed signals.
        Safe to call repeatedly with different chronos_weight
        / performance_ratio / trend_halflife candidates for
        tuning, since it never touches Chronos or pvlib.

        `vision_adjustment` may be a scalar or a per-block
        array matching the forecast horizon.
        """

        if chronos_weight is None:
            chronos_weight = self.chronos_weight

        if performance_ratio is None:
            performance_ratio = self.performance_ratio

        if trend_halflife_min is None:
            trend_halflife_min = self.trend_halflife_min

        alpha = chronos_weight * signals["context_ratio"]

        horizon = signals["horizon_minutes"]

        # Smart persistence: extrapolate the recent kt slope
        # forward, saturating at slope * halflife so a brief
        # trend never runs away over a long horizon. A halflife
        # of 0 disables the trend (plain flat persistence).
        if trend_halflife_min > 0:
            trend = (
                signals["kt_slope_per_min"]
                * trend_halflife_min
                * (1 - np.exp(-horizon / trend_halflife_min))
            )
        else:
            trend = 0.0

        kt_persistence = np.clip(
            (signals["kt_now"] + trend) * (1 + vision_adjustment),
            0,
            self.max_clear_sky_index
        )

        final_kt = (
            alpha * signals["chronos_kt_forecast"]
            + (1 - alpha) * kt_persistence
        )

        clearsky_curve = (
            signals["poa_curve"] / 1000
            * self.capacity_kw
            * performance_ratio
        )

        chronos_forecast_kw = np.clip(
            signals["chronos_kt_forecast"] * clearsky_curve, 0, self.capacity_kw
        )

        physics_forecast_kw = np.clip(
            kt_persistence * clearsky_curve, 0, self.capacity_kw
        )

        final_forecast_kw = np.clip(
            final_kt * clearsky_curve, 0, self.capacity_kw
        )

        result = pd.DataFrame({
            "timestamp": signals["forecast_timestamps"],
            "chronos_forecast_kw": chronos_forecast_kw,
            "physics_forecast_kw": physics_forecast_kw,
            "final_forecast_kw": final_forecast_kw
        })

        result.attrs["chronos_weight_used"] = alpha
        result.attrs["clear_sky_index_now"] = signals["kt_now"]

        return result

    # --------------------------------------------------

    def predict(self, dataframe, run_time, vision_features=None):
        """
        Convenience wrapper: computes signals and blends them
        with the configured defaults. Pass `vision_features`
        (a Gemini weather_features dict) to apply the Windy
        cloud-video adjustment profile to the physics baseline.
        """

        signals = self.compute_signals(dataframe, run_time)

        vision_adjustment = 0.0

        if vision_features is not None:
            from modules.fusion.fusion import FeatureFusion
            vision_adjustment = FeatureFusion().trend_adjustment_profile(
                vision_features,
                signals["horizon_minutes"]
            )

        return self.blend_signals(
            signals,
            vision_adjustment=vision_adjustment
        )
