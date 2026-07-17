"""
=========================================================
Solar Forecasting Project
Evaluator
=========================================================
Scores our forecast against actual generation, and against
Enercast as a side-by-side comparison reference only -
Enercast is never used as a training input, per the mentor
brief.
=========================================================
"""

from pathlib import Path

import pandas as pd

from config.config import settings
from modules.evaluation import metrics


class Evaluator:

    def __init__(self):

        plant = settings["plant"]

        self.capacity_kw = plant["capacity_mw"] * 1000

        self.enercast_folder = Path(
            settings["paths"]["enercast_data"]
        )

    # --------------------------------------------------

    def load_enercast_day(self, date):
        """
        Loads the Enercast schedule for one calendar day.
        Returns None if no Enercast file exists for that date.
        """

        month_name = date.strftime("%B").lower()

        filename = f"Sirmour_{date.day}{month_name}_enercast.csv"

        file_path = self.enercast_folder / filename

        if not file_path.exists():
            return None

        enercast = pd.read_csv(file_path)

        timestamps = pd.to_datetime(
            date.strftime("%Y-%m-%d") + " " + enercast["Time"]
        )

        return pd.DataFrame({
            "timestamp": timestamps,
            "enercast_kw": enercast["Scheduled MW"] * 1000
        })

    # --------------------------------------------------

    def score(self, forecast_column, comparison):

        return {
            "mae_kw": metrics.mean_absolute_error(
                comparison[forecast_column],
                comparison["active_power_kw"]
            ),
            "rmse_kw": metrics.root_mean_squared_error(
                comparison[forecast_column],
                comparison["active_power_kw"]
            ),
            "avg_deviation_pct": metrics.average_percentage_deviation(
                comparison[forecast_column],
                comparison["active_power_kw"],
                self.capacity_kw
            ),
            "scheduling_penalty": metrics.scheduling_penalty(
                comparison[forecast_column],
                comparison["active_power_kw"],
                self.capacity_kw
            )
        }

    # --------------------------------------------------

    def evaluate(self, forecast, actual):
        """
        forecast : DataFrame with timestamp, final_forecast_kw
        actual   : DataFrame with timestamp, active_power_kw

        Returns the merged comparison table plus a metrics
        report for our forecast and, where available, Enercast.
        """

        comparison = forecast.merge(
            actual[["timestamp", "active_power_kw"]],
            on="timestamp",
            how="inner"
        )

        enercast_day = self.load_enercast_day(
            comparison["timestamp"].iloc[0].date()
        )

        if enercast_day is not None:
            comparison = comparison.merge(
                enercast_day,
                on="timestamp",
                how="left"
            )

        report = {
            "our_forecast": self.score("final_forecast_kw", comparison)
        }

        has_enercast = (
            "enercast_kw" in comparison.columns
            and comparison["enercast_kw"].notna().any()
        )

        if has_enercast:
            report["enercast"] = self.score("enercast_kw", comparison)

        return comparison, report
