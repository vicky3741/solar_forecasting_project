"""
=========================================================
Solar Forecasting Project
Chronos Model
=========================================================
Loads Amazon's pretrained Chronos time-series model and
produces zero-shot forecasts. No training on our own data
is required - the model has already learned general
time-series patterns, so it works even with very limited
plant history.
=========================================================
"""

import torch
from chronos import BaseChronosPipeline

from config.config import settings


class ChronosModel:

    def __init__(self):

        forecast_settings = settings["forecast_model"]

        self.model_name = forecast_settings["model"]
        self.device = forecast_settings.get("device", "cpu")

        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float32
        )

    # --------------------------------------------------

    def forecast(self,
                 context_values,
                 prediction_length,
                 quantile_levels=(0.1, 0.5, 0.9)):
        """
        Produces a zero-shot forecast from a 1D series of
        historical values.

        Returns quantiles (shape: [len(quantile_levels), prediction_length])
        and the mean forecast (shape: [prediction_length]).
        """

        context = torch.tensor(
            context_values,
            dtype=torch.float32
        )

        quantiles, mean = self.pipeline.predict_quantiles(
            inputs=context,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels)
        )

        return {
            "quantiles": quantiles[0].numpy(),
            "mean": mean[0].numpy(),
            "quantile_levels": list(quantile_levels)
        }
