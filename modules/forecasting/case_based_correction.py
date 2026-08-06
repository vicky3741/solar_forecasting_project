"""
=========================================================
Solar Forecasting Project
Case-Based Correction (k-Nearest-Neighbour retrieval)
=========================================================
Instead of a trained model (the LightGBM residual
corrector), this looks up the most similar PAST forecast
situations that already have real measured outcomes, and
nudges the current forecast by how those analogous cases
actually turned out.

"Similar" = nearest in a small, normalised, weighted feature
space, using only information known at prediction time:
block hour, forecast horizon, kt (cloudiness) at run time,
and our own forecast value. No future information, and only
REAL measured blocks ever enter the case store (Part 2:
reconstructed blocks are never learned from).

Like every other signal we have added (weather, optical
flow, the LightGBM corrector), this is validated
leave-one-day-out against actual generation BEFORE it is
allowed to touch a live forecast - see
tests/test_case_based_experiment.py. It ships only if it
helps on unseen days.
=========================================================
"""

import numpy as np
import pandas as pd

from config.config import settings


# Same 4 prediction-time features as the LightGBM corrector, so the
# two approaches are directly comparable.
FEATURES = ["block_hour", "horizon_min", "kt_now", "final_forecast_kw"]

# Feature emphasis, in the spirit of Kushal's weighting (time-of-day
# and cloudiness matter most for how a day turns out).
DEFAULT_WEIGHTS = {
    "block_hour": 2.0,
    "kt_now": 2.0,
    "final_forecast_kw": 1.5,
    "horizon_min": 1.0,
}


class CaseBasedCorrector:

    def __init__(self, k=None, weights=None, strength=None):

        cfg = settings.get("case_based_correction", {})

        self.enabled = cfg.get("enabled", False)
        self.k = k or cfg.get("k", 40)
        self.weights = weights or cfg.get("weights", DEFAULT_WEIGHTS)

        # Damping: apply only this fraction of the retrieved
        # correction. Validated leave-one-day-out - full strength
        # (1.0) OVERFITS and hurts (-0.5 pts); a gentle 0.5 helps
        # (+0.47 pts, 12/17 days). The analogues give a hint about
        # bias, not a value to trust outright.
        self.strength = (
            strength if strength is not None else cfg.get("strength", 0.5)
        )

        self.min_case_days = cfg.get("min_case_days", 8)
        self.capacity_kw = settings["plant"]["capacity_mw"] * 1000

        from pathlib import Path
        self.case_store_path = Path(
            cfg.get("case_store_path", "models/case_store.csv")
        )

        self._cases = None       # normalised + weighted feature rows
        self._residuals = None   # actual - forecast for each case
        self._mean = None
        self._std = None
        self._weight_vector = None
        self._loaded = False

    # --------------------------------------------------

    @staticmethod
    def build_features(forecast, run_time, kt_now):
        """
        forecast : DataFrame with timestamp, final_forecast_kw.
        Identical feature construction to the LightGBM corrector.
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
            "final_forecast_kw": forecast["final_forecast_kw"],
        })

    # --------------------------------------------------

    def fit(self, feature_frame, residuals):
        """
        Builds the case store: normalise each feature (z-score),
        then scale by the square root of its weight so weighted
        Euclidean distance emphasises the important features.
        """

        matrix = feature_frame[FEATURES].to_numpy(dtype=float)

        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0) + 1e-9

        self._weight_vector = np.sqrt(
            np.array([self.weights[f] for f in FEATURES], dtype=float)
        )

        normalised = (matrix - self._mean) / self._std

        self._cases = normalised * self._weight_vector
        self._residuals = np.asarray(residuals, dtype=float)

        return self

    # --------------------------------------------------

    def predict_residual(self, feature_frame):
        """
        For each query block, retrieve the k nearest cases and
        return their inverse-distance-weighted mean residual
        (closer analogues count for more).
        """

        if self._cases is None:
            raise RuntimeError("CaseBasedCorrector.fit must be called first")

        query = feature_frame[FEATURES].to_numpy(dtype=float)
        query = ((query - self._mean) / self._std) * self._weight_vector

        k = min(self.k, len(self._cases))

        predictions = np.empty(len(query))

        for i in range(len(query)):

            distances_sq = ((self._cases - query[i]) ** 2).sum(axis=1)

            nearest = np.argpartition(distances_sq, k - 1)[:k]

            # Inverse-distance weights - the closer the analogue,
            # the more it counts. The epsilon avoids divide-by-zero
            # when a query lands exactly on a stored case.
            weights = 1.0 / (np.sqrt(distances_sq[nearest]) + 1e-6)

            predictions[i] = np.average(
                self._residuals[nearest], weights=weights
            )

        return predictions

    # --------------------------------------------------

    def apply(self, forecast, run_time, kt_now):
        """
        Returns the forecast with final_forecast_kw nudged by the
        retrieved analogues' average outcome. The uncorrected
        value is preserved for transparency.
        """

        features = self.build_features(forecast, run_time, kt_now)

        # Gentle nudge only - full strength overfits (see __init__).
        predicted_residual = self.predict_residual(features) * self.strength

        forecast = forecast.copy()
        forecast["final_forecast_uncorrected_kw"] = forecast["final_forecast_kw"]

        forecast["final_forecast_kw"] = np.clip(
            forecast["final_forecast_kw"] + predicted_residual,
            0,
            self.capacity_kw,
        )

        return forecast

    # --------------------------------------------------

    def save_case_store(self, feature_frame, residuals, dates):
        """
        Persists the raw case store (features + residual + day) so a
        live run can rebuild it without re-running the backtest.
        Called by the experiment only after the out-of-sample
        verdict is positive.
        """

        store = feature_frame[FEATURES].copy()
        store["residual_kw"] = np.asarray(residuals, dtype=float)
        store["date"] = list(dates)

        self.case_store_path.parent.mkdir(parents=True, exist_ok=True)
        store.to_csv(self.case_store_path, index=False)

        return self.case_store_path

    # --------------------------------------------------

    @property
    def available(self):
        """
        Runs live only when enabled, a saved case store exists, and
        it covers enough distinct days (too few days = unreliable
        analogues, same lesson as the LightGBM corrector).
        """

        if not (self.enabled and self.case_store_path.exists()):
            return False

        store = pd.read_csv(self.case_store_path)
        return store["date"].nunique() >= self.min_case_days

    # --------------------------------------------------

    def load(self):
        """Fits the corrector from the saved case store (once)."""

        if not self._loaded:
            store = pd.read_csv(self.case_store_path)
            self.fit(store, store["residual_kw"])
            self._loaded = True

        return self
