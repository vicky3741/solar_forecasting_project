"""
=========================================================
Solar Forecasting Project
Residual Correction (LightGBM)
=========================================================
Learns the hybrid forecast's systematic mistakes from the
backtest record and corrects live forecasts with them.

Validated honestly before shipping (2026-07-19):
leave-one-day-out across 13 days improved unseen-day
deviation by +1.14 percentage points, helping 10/13 days
(see tests/test_residual_experiment.py, which also trains
and saves the production model).

A walk-forward test (tests/test_walkforward_experiment.py)
additionally showed corrections trained on fewer than ~8
days make forecasts WORSE - so the corrector refuses to
run until its trained model has at least
`min_training_days` of history behind it (recorded in a
metadata file saved next to the model).

Features are strictly things known at prediction time:
block hour, forecast horizon, kt at run time, and our own
forecast value. No future information.
=========================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.config import settings


FEATURES = ["block_hour", "horizon_min", "kt_now", "final_forecast_kw"]

TRAINING_PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_data_in_leaf": 40,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
    "verbose": -1
}

NUM_BOOST_ROUND = 300


class ResidualCorrector:

    def __init__(self):

        correction_settings = settings.get("residual_correction", {})

        self.enabled = correction_settings.get("enabled", False)

        self.model_path = Path(
            correction_settings.get("model_path", "models/residual_lgbm.txt")
        )

        self.min_training_days = correction_settings.get("min_training_days", 8)

        self.capacity_kw = settings["plant"]["capacity_mw"] * 1000

        self._model = None

    # --------------------------------------------------

    @property
    def meta_path(self):

        return self.model_path.with_suffix(".meta.json")

    # --------------------------------------------------

    def get_training_days(self):
        """
        How many days of history the saved model was trained
        on, from the metadata written at training time.
        Returns 0 when no metadata exists (old/unknown model).
        """

        if not self.meta_path.exists():
            return 0

        with open(self.meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)

        return int(meta.get("training_days", 0))

    # --------------------------------------------------

    @property
    def available(self):
        """
        The corrector only runs when enabled, a model exists,
        AND that model was trained on enough days - the
        walk-forward experiment showed under-trained
        corrections actively hurt.
        """

        return (
            self.enabled
            and self.model_path.exists()
            and self.get_training_days() >= self.min_training_days
        )

    # --------------------------------------------------

    def load(self):

        if self._model is None:
            self._model = lgb.Booster(model_file=str(self.model_path))

        return self._model

    # --------------------------------------------------

    @staticmethod
    def build_features(forecast, run_time, kt_now):
        """
        forecast : DataFrame with timestamp, final_forecast_kw
        """

        return pd.DataFrame({
            "block_hour": (
                forecast["timestamp"].dt.hour
                + forecast["timestamp"].dt.minute / 60
            ),
            "horizon_min": (
                (forecast["timestamp"] - run_time).dt.total_seconds() / 60
            ),
            "kt_now": kt_now,
            "final_forecast_kw": forecast["final_forecast_kw"]
        })

    # --------------------------------------------------

    def apply(self, forecast, run_time, kt_now):
        """
        Returns the forecast with final_forecast_kw corrected.
        The uncorrected value is kept in
        final_forecast_uncorrected_kw for transparency.
        """

        model = self.load()

        features = self.build_features(forecast, run_time, kt_now)

        predicted_residual = model.predict(features[FEATURES])

        forecast = forecast.copy()

        forecast["final_forecast_uncorrected_kw"] = forecast["final_forecast_kw"]

        forecast["final_forecast_kw"] = np.clip(
            forecast["final_forecast_kw"] + predicted_residual,
            0,
            self.capacity_kw
        )

        return forecast

    # --------------------------------------------------

    def train_and_save(self, feature_frame, residuals, training_days):
        """
        Trains on the full available history and saves the
        production model plus its metadata (how many days it
        was trained on - the `available` gate reads this).
        Called by the experiment script after (and only
        after) the leave-one-day-out validation verdict is
        positive.
        """

        train_set = lgb.Dataset(feature_frame[FEATURES], label=residuals)

        model = lgb.train(TRAINING_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model.save_model(str(self.model_path))

        with open(self.meta_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "training_days": int(training_days),
                    "trained_at": pd.Timestamp.now().isoformat()
                },
                file,
                indent=4
            )

        return self.model_path
