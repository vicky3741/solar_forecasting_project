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

Live data feed: when storage.auto_pull is on, the run first
pulls the latest meter data from the team S3 bucket into
data/historical; when storage.auto_push is on, it pushes the
forecast and Current Final Schedule back to the bucket after.
Both degrade gracefully - if the cloud is unreachable the run
continues on whatever local data exists, so a network hiccup
never blocks a forecast.
=========================================================
"""

from pathlib import Path

import pandas as pd

from config.config import settings
from modules.preprocessing.preprocess import DataPreprocessor
from modules.forecasting.predictor import HybridPredictor
from modules.forecasting.residual_correction import ResidualCorrector
from modules.forecasting.case_based_correction import CaseBasedCorrector
from modules.evaluation.evaluator import Evaluator
from modules.vision.vision_module import VisionModule
from modules.fusion.fusion import FeatureFusion
from modules.storage.s3_client import S3Storage
from utils import file_manager
from utils.logger import get_logger


class Orchestrator:

    def __init__(self):

        self.logger = get_logger()

        self.preprocessor = DataPreprocessor()
        self.predictor = HybridPredictor()
        self.corrector = ResidualCorrector()
        self.case_corrector = CaseBasedCorrector()
        self.evaluator = Evaluator()
        self.vision = VisionModule()
        self.fusion = FeatureFusion()
        self.storage = S3Storage()

        store = settings.get("storage", {})
        self.auto_pull = store.get("enabled", False) and store.get("auto_pull", False)
        self.auto_push = store.get("enabled", False) and store.get("auto_push", False)

        self.historical_folder = Path(settings["paths"]["historical_data"])

        # Local video fallback, searched newest-folder-first when S3
        # has nothing. Auto-captured clips now land in their own
        # folder (windy_capture.video_dir), while the older
        # hand-recorded July 9 clips stay in data/windy/videos - both
        # are still valid sources, so both are searched.
        capture_folder = Path(
            settings.get("windy_capture", {}).get(
                "video_dir", "data/windy/new_videos"
            )
        )
        legacy_folder = Path(settings["paths"]["windy_data"]) / "videos"

        self.windy_folders = [capture_folder, legacy_folder]
        self.vision_output_folder = "outputs/extracted_frames"

        self.forecasts_folder = Path(settings["outputs"]["forecasts"])
        self.schedules_folder = Path(settings["outputs"]["schedules"])
        self.reports_folder = Path(settings["outputs"]["reports"])

        self.official_run_times = settings["forecast"]["run_times"]

    # --------------------------------------------------

    def pull_latest_from_cloud(self):
        """
        Pulls the latest meter data from the team S3 bucket
        into data/historical before forecasting. Never raises
        - a cloud/network failure just leaves the local data
        as-is and logs a warning, so the forecast still runs.
        """

        if not self.auto_pull:
            return

        try:
            synced = self.storage.sync_meter_data(self.historical_folder)

            if synced:
                self.logger.info(
                    f"Pulled {len(synced)} new/updated meter file(s) from S3"
                )
            else:
                self.logger.info("S3 meter data already up to date")

        except Exception as error:
            self.logger.warning(
                f"S3 pull skipped ({error}) - continuing on local data"
            )

    # --------------------------------------------------

    def push_outputs_to_cloud(self, run_label, forecast_path, schedule_path):
        """
        Pushes this run's forecast and the Current Final
        Schedule back to the team S3 bucket. Never raises - a
        push failure is logged but does not fail the run,
        since the outputs are already saved locally.
        """

        if not self.auto_push:
            return

        try:
            self.storage.push_output(forecast_path, "forecasts", f"{run_label}.csv")
            self.storage.push_output(schedule_path, "current_final_schedule.csv")
            self.logger.info("Pushed forecast + Current Final Schedule to S3")

        except Exception as error:
            self.logger.warning(
                f"S3 push skipped ({error}) - outputs are still saved locally"
            )

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

    def get_latest_video(self, run_time):
        """
        Path to the latest same-day Windy video at/before
        run_time. Prefers the live S3 feed (Team 3's bucket,
        also viewable at 13.206.205.164) when storage is on,
        falling back to any local videos. Returns None if
        neither has one.
        """

        if self.auto_pull:
            try:
                video_path = self.storage.fetch_latest_video(run_time)
                if video_path is not None:
                    return video_path

            except Exception as error:
                self.logger.warning(
                    f"S3 video fetch skipped ({error}) - trying local videos"
                )

        for folder in self.windy_folders:

            if not folder.exists():
                continue

            video_path = self.vision.find_latest_video(folder, run_time)

            if video_path is not None:
                return video_path

        return None

    # --------------------------------------------------

    def get_vision_features(self, run_time):
        """
        Most recent same-day Windy video's Gemini features, or
        None. A Gemini failure degrades the run to no-vision
        rather than blocking the forecast.
        """

        video_path = self.get_latest_video(run_time)

        if video_path is None:
            return None

        try:
            result = self.vision.analyze_video(
                video_path,
                self.vision_output_folder
            )

        except Exception as error:
            self.logger.warning(
                f"Vision analysis failed for {Path(video_path).name}: {error} "
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

        actual_columns = ["timestamp", "active_power_kw"]
        if "is_real_measurement" in dataframe.columns:
            actual_columns.append("is_real_measurement")

        actual = dataframe[actual_columns]

        comparison, report = self.evaluator.evaluate(forecast_blocks, actual)

        if comparison.empty:
            self.logger.warning(
                f"End-of-day validation skipped for {run_time.date()} - "
                "no actual meter data for this day yet. The forecast is "
                "saved; it just cannot be scored until the meter feed "
                "catches up."
            )
            return None

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
            # Snap to the 15-minute grid, not just the minute. The
            # scheduler fires at 06:45 but the capture takes ~1 min,
            # so the orchestrator actually starts at ~06:46 - and
            # flooring to the minute put every forecast block at
            # :01/:16/:31/:46. Those timestamps never match the meter
            # data's :00/:15/:30/:45 grid, so the 2026-07-26 server
            # run produced a forecast with ZERO gradeable blocks.
            # Flooring to 15 min anchors the run at its scheduled
            # block boundary and keeps every block on the grid.
            run_time = pd.Timestamp.now().floor("15min")

        self.logger.info(f"Starting forecast run for {run_time}")

        self.pull_latest_from_cloud()

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

        # Case-based correction (Kushal's idea): nudge by how similar
        # past days actually turned out. Validated +0.47 pts on unseen
        # days; only runs when enabled and the case store has enough
        # history (see modules/forecasting/case_based_correction.py).
        if self.case_corrector.available:
            self.case_corrector.load()
            forecast = self.case_corrector.apply(
                forecast,
                run_time,
                signals["kt_now"]
            )
            self.logger.info("Case-based correction applied")

        run_label = run_time.strftime("%Y-%m-%d_%H-%M")

        forecast_path = self.forecasts_folder / f"{run_label}.csv"
        file_manager.save_dataframe(forecast, forecast_path)

        archive_run = forecast.copy()
        archive_run["run_time"] = run_time
        file_manager.append_dataframe(
            archive_run,
            self.forecasts_folder / "archive.csv"
        )

        schedule_path = self.schedules_folder / "current_final_schedule.csv"
        schedule = self.build_current_schedule(dataframe, forecast, run_time)
        file_manager.save_dataframe(schedule, schedule_path)

        if self.is_last_official_run(run_time):
            self.save_end_of_day_validation(dataframe, run_time)

        self.push_outputs_to_cloud(run_label, forecast_path, schedule_path)

        self.logger.info(f"Forecast run complete for {run_time}")

        return forecast
