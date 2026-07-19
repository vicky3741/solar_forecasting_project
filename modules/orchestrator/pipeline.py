"""
=========================================================
Solar Forecasting Project
Orchestrator
=========================================================
Runs one complete forecast cycle end to end: load the
latest available data, read the latest same-day Windy
video if one exists, forecast forward with the hybrid
predictor, and produce the output files the mentor brief
asks for:

  - An individual forecast file for this run
  - An append-only archive of every run's forecast
  - The Current Final Schedule - actual generation for
    every block that has already happened today, and the
    latest forecast for every block still ahead
  - An end-of-day validation file, once the last official
    run time of the day has passed

NOTE: "latest available data" here means whatever is on
disk in data/historical and data/windy/videos. There is no
live meter/video feed wired up yet - that is a separate,
not-yet-built integration. Running this against yesterday's
static files does not simulate a live plant.
=========================================================
"""

from pathlib import Path

import pandas as pd

from config.config import settings
from modules.preprocessing.preprocess import DataPreprocessor
from modules.forecasting.predictor import HybridPredictor
from modules.forecasting.residual_correction import ResidualCorrector
from modules.evaluation.evaluator import Evaluator
from modules.vision.vision_module import VisionModule
from modules.fusion.fusion import FeatureFusion
from utils import file_manager
from utils.logger import get_logger


class Orchestrator:

    def __init__(self):

        self.logger = get_logger()

        self.preprocessor = DataPreprocessor()
        self.predictor = HybridPredictor()
        self.corrector = ResidualCorrector()
        self.evaluator = Evaluator()
        self.vision = VisionModule()
        self.fusion = FeatureFusion()

        self.historical_folder = Path(settings["paths"]["historical_data"])
        self.windy_folder = Path(settings["paths"]["windy_data"]) / "videos"
        self.vision_output_folder = "outputs/extracted_frames"

        self.forecasts_folder = Path(settings["outputs"]["forecasts"])
        self.schedules_folder = Path(settings["outputs"]["schedules"])
        self.reports_folder = Path(settings["outputs"]["reports"])

        self.official_run_times = settings["forecast"]["run_times"]

    # --------------------------------------------------

    def load_data(self):
        """
        Preprocesses every daily raw CSV independently (each
        day has its own gaps/coverage, so resampling per day
        avoids interpolating across the overnight gap between
        days), then concatenates the results.
        """

        csv_files = sorted(self.historical_folder.glob("*.csv"))

        processed_days = [
            self.preprocessor.preprocess(
                file_path=csv_file,
                required_columns=["TimeStamp"],
                timestamp_column="TimeStamp"
            )
            for csv_file in csv_files
        ]

        dataframe = pd.concat(processed_days, ignore_index=True)

        return dataframe.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------

    def get_vision_features(self, run_time):
        """
        Most recent same-day Windy video's Gemini features, or
        None. A Gemini failure degrades the run to no-vision
        rather than blocking the forecast.
        """

        video_path = self.vision.find_latest_video(
            self.windy_folder,
            run_time
        )

        if video_path is None:
            return None

        try:
            result = self.vision.analyze_video(
                video_path,
                self.vision_output_folder
            )

        except Exception as error:
            self.logger.warning(
                f"Vision analysis failed for {video_path.name}: {error} "
                "- continuing without the vision signal"
            )
            return None

        return self.fusion.prepare_vision_features(result)

    # --------------------------------------------------

    def build_current_schedule(self, dataframe, forecast, run_time):
        """
        Actual generation for every block already completed
        today, plus this run's forecast for every block still
        ahead - this is the mentor brief's "Current Final
        Schedule", overwritten after every run.
        """

        today = dataframe[dataframe["timestamp"].dt.date == run_time.date()]
        today = today[today["timestamp"] <= run_time]

        past = today[["timestamp", "active_power_kw"]].rename(
            columns={"active_power_kw": "value_kw"}
        )
        past["source"] = "actual"

        future = forecast[["timestamp", "final_forecast_kw"]].rename(
            columns={"final_forecast_kw": "value_kw"}
        )
        future["source"] = "forecast"

        schedule = pd.concat([past, future], ignore_index=True)
        schedule["last_updated_run_time"] = run_time

        return schedule

    # --------------------------------------------------

    def is_last_official_run(self, run_time):

        run_time_str = run_time.strftime("%H:%M")

        return run_time_str == self.official_run_times[-1]

    # --------------------------------------------------

    def save_end_of_day_validation(self, dataframe, run_time):
        """
        Compares the Current Final Schedule's forecasted
        blocks against actual generation for the full day,
        once it is available.
        """

        schedule_path = self.schedules_folder / "current_final_schedule.csv"

        schedule = file_manager.load_dataframe(
            schedule_path,
            parse_dates=["timestamp"]
        )

        if schedule is None:
            return None

        forecast_blocks = schedule[schedule["source"] == "forecast"].rename(
            columns={"value_kw": "final_forecast_kw"}
        )

        actual = dataframe[["timestamp", "active_power_kw"]]

        _, report = self.evaluator.evaluate(forecast_blocks, actual)

        report_path = (
            self.reports_folder
            / f"{run_time.date()}_end_of_day_validation.json"
        )

        file_manager.save_json(report, report_path)

        self.logger.info(f"End-of-day validation saved: {report_path}")

        return report

    # --------------------------------------------------

    def run(self, run_time=None):

        if run_time is None:
            run_time = pd.Timestamp.now().floor("min")

        self.logger.info(f"Starting forecast run for {run_time}")

        dataframe = self.load_data()

        vision_features = self.get_vision_features(run_time)

        signals = self.predictor.compute_signals(dataframe, run_time)

        vision_adjustment = self.fusion.trend_adjustment_profile(
            vision_features,
            signals["horizon_minutes"]
        )

        forecast = self.predictor.blend_signals(
            signals,
            vision_adjustment=vision_adjustment
        )

        if self.corrector.available:
            forecast = self.corrector.apply(
                forecast,
                run_time,
                signals["kt_now"]
            )
            self.logger.info("Residual correction applied")

        run_label = run_time.strftime("%Y-%m-%d_%H-%M")

        forecast_path = self.forecasts_folder / f"{run_label}.csv"
        file_manager.save_dataframe(forecast, forecast_path)

        archive_run = forecast.copy()
        archive_run["run_time"] = run_time
        file_manager.append_dataframe(
            archive_run,
            self.forecasts_folder / "archive.csv"
        )

        schedule = self.build_current_schedule(dataframe, forecast, run_time)
        file_manager.save_dataframe(
            schedule,
            self.schedules_folder / "current_final_schedule.csv"
        )

        if self.is_last_official_run(run_time):
            self.save_end_of_day_validation(dataframe, run_time)

        self.logger.info(f"Forecast run complete for {run_time}")

        return forecast
