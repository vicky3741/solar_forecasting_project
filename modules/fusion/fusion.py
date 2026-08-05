"""
=========================================
Feature Fusion Module
=========================================
Combines numerical weather data and
vision-based weather features into one
dataset for forecasting.
=========================================
"""

import numpy as np
import pandas as pd


class FeatureFusion:

    def __init__(self):
        pass


    def prepare_vision_features(self, vision_result):

        """
        Extracts only the weather features returned by
            the Vision module.
        """

        return vision_result["weather_features"]
    def vision_to_dataframe(self, vision_features):

        """
        Converts vision features dictionary
        into a single-row DataFrame.
        """

        return pd.DataFrame([vision_features])
    
    def fuse(self, numerical_df, vision_df):

        """
        Combine numerical data with vision data.
        """

        for column in vision_df.columns:

            numerical_df[column] = vision_df.iloc[0][column]

        return numerical_df

    def trend_adjustment_factor(self, vision_features, max_adjustment=0.15):

        """
        Converts Gemini's free-text trend fields into a bounded
        nudge on the physics persistence forecast: positive
        when clouds are trending away, negative when they are
        moving in, scaled down when the vision model itself
        reports low confidence. Neutral text (or missing
        fields) gives no adjustment.
        """

        trend_text = " ".join([
            str(vision_features.get("irradiance_trend", "")),
            str(vision_features.get("expected_generation_trend", ""))
        ]).lower()

        if "increas" in trend_text:
            direction = 1
        elif "decreas" in trend_text:
            direction = -1
        else:
            direction = 0

        try:
            confidence = float(vision_features.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        confidence = min(max(confidence, 0.0), 1.0)

        return direction * max_adjustment * confidence

    # --------------------------------------------------

    def _trend_direction(self, text):

        text = str(text).lower()

        if "increas" in text or "clear" in text:
            return 1

        if "decreas" in text or "cloud" in text:
            return -1

        return 0

    # --------------------------------------------------

    # cloud_field_structure -> extra multiplier on the vision adjustment.
    # Untested at scale (2026-08-05: 2 real days only) - NOT wired into any
    # live caller. See tests/test_cloud_structure_amplifier.py before this
    # is ever passed from predictor.py/pipeline.py.
    STRUCTURE_MULTIPLIER = {"solid": 1.0, "patchy": 1.5, "broken": 2.0}

    def trend_adjustment_profile(self,
                                 vision_features,
                                 horizon_minutes,
                                 max_adjustment=0.15,
                                 structure_multiplier=None):

        """
        Per-block vision adjustment, replacing the old flat
        scalar nudge. A cloud front arriving in 90 minutes
        should not change the next 15-minute block - so the
        adjustment:

          - ramps in around the predicted arrival/clearing
            time (`minutes_until_change`),
          - uses the next-2h trend for near blocks and the
            2-4h trend for far blocks,
          - decays with horizon (a single video says little
            about 5+ hours ahead),
          - scales with the vision model's own confidence.

        `structure_multiplier` optionally scales the whole result by
        STRUCTURE_MULTIPLIER[cloud_field_structure] - patchy/broken skies
        measured 4.2x worse DSM penalty than calm ones (volatility proxy,
        2026-08-04), so a patchy/broken read leans harder on vision's
        directional call instead of smoothing through the spike like
        persistence does. None (default) = old behavior, unchanged.

        Falls back to the old flat behavior when the features
        came from the v1 prompt (no timing fields), and to no
        adjustment when features are missing entirely.
        """

        horizon_minutes = np.asarray(horizon_minutes, dtype=float)

        if not vision_features:
            return np.zeros_like(horizon_minutes)

        # v1-format cache: no timing fields - flat fallback
        if "trend_next_2h" not in vision_features:
            return np.full_like(
                horizon_minutes,
                self.trend_adjustment_factor(vision_features, max_adjustment)
            )

        try:
            confidence = float(vision_features.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        confidence = min(max(confidence, 0.0), 1.0)

        near_direction = self._trend_direction(
            vision_features.get("trend_next_2h", "")
        )
        far_direction = self._trend_direction(
            vision_features.get("trend_2h_to_4h", "")
        )

        direction = np.where(
            horizon_minutes <= 120,
            near_direction,
            far_direction
        ).astype(float)

        try:
            minutes_until = float(vision_features.get("minutes_until_change"))
        except (TypeError, ValueError):
            minutes_until = 0.0

        # 0 before the change arrives, ramping to full effect
        # across the ~30 minutes around the predicted arrival
        ramp = np.clip(
            (horizon_minutes - minutes_until) / 30.0 + 1.0,
            0.0,
            1.0
        )

        # One video's information goes stale hours out
        staleness = np.exp(-horizon_minutes / 240.0)

        result = direction * max_adjustment * confidence * ramp * staleness

        if structure_multiplier:
            structure = vision_features.get("cloud_field_structure")
            result = result * self.STRUCTURE_MULTIPLIER.get(structure, 1.0)

        return result