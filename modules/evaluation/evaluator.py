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

        # Enercast is a Sirmour-only reference feed. The Telangana
        # plants have none, so their overlay turns this off and every
        # Enercast column simply does not appear in their reports -
        # rather than the reports carrying an empty column that looks
        # like a data outage.
        enercast = settings.get("enercast", {})
        self.enercast_enabled = enercast.get("enabled", True)
        self.enercast_prefix = enercast.get("filename_prefix", "Sirmour")

    # --------------------------------------------------

    def load_enercast_day(self, date):
        """
        Loads the Enercast schedule for one calendar day.
        Returns None if this plant has no Enercast feed, or if no
        Enercast file exists for that date.
        """

        if not self.enercast_enabled:
            return None

        month_name = date.strftime("%B").lower()

        filename = f"{self.enercast_prefix}_{date.day}{month_name}_enercast.csv"

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
                   (optionally is_real_measurement)

        Returns the merged comparison table plus a metrics
        report for our forecast and, where available, Enercast.

        When `actual` carries an is_real_measurement flag, blocks
        the meter never actually recorded (reconstructed by
        gap-filling) are dropped before scoring - so accuracy is
        only ever measured against genuine readings, never against
        interpolated data. The forecast itself is unaffected; this
        only governs what we grade against.
        """

        actual_columns = ["timestamp", "active_power_kw"]

        if "is_real_measurement" in actual.columns:
            actual_columns.append("is_real_measurement")

        comparison = forecast.merge(
            actual[actual_columns],
            on="timestamp",
            how="inner"
        )

        if "is_real_measurement" in comparison.columns:
            comparison = comparison[
                comparison["is_real_measurement"].fillna(False)
            ].reset_index(drop=True)
            comparison = comparison.drop(columns=["is_real_measurement"])

        # No overlap between forecast and actual - happens when the
        # day's meter data has not arrived yet, or when every block
        # in range was reconstructed (no real reading to grade
        # against). Return an empty result rather than indexing row 0
        # of an empty frame, which used to raise "single positional
        # indexer is out-of-bounds" and abort the end-of-day run.
        if comparison.empty:
            return comparison, {
                "our_forecast": None,
                "note": (
                    "no real (measured) generation overlapping this "
                    "forecast - cannot score the run yet"
                )
            }

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
