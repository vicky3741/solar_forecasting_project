"""
=========================================================
Solar Forecasting Project
Backtester
=========================================================
Simulates every one of the mentor brief's 7 official daily
forecast runs, on every available historical day, using
only data that would genuinely have been available at that
point in time (no lookahead). Scores each run against
actual generation and against Enercast.

Only 2026-07-09 has real Windy cloud videos, so that is the
only day where the vision cloud-trend signal is actually
used - every other day forecasts from Chronos + physics
alone. This is a real data limitation, not a bug: the
vision integration is wired for any day a video exists for,
but we only have video coverage for one day so far.
=========================================================
"""

from pathlib import Path

import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.evaluation.evaluator import Evaluator
from modules.evaluation import metrics
from modules.vision.vision_module import VisionModule
from modules.fusion.fusion import FeatureFusion


class Backtester:

    def __init__(self):

        self.predictor = HybridPredictor()
        self.evaluator = Evaluator()
        self.vision = VisionModule()
        self.fusion = FeatureFusion()

        self.official_run_times = settings["forecast"]["run_times"]

        self.windy_folder = Path(settings["paths"]["windy_data"]) / "videos"
        self.vision_output_folder = "outputs/extracted_frames"

    # --------------------------------------------------

    def get_vision_features(self, run_time):
        """
        Looks up the most recent same-day Windy video at/before
        run_time and returns its Gemini weather features.
        Returns (None, False) when no video exists yet at that
        point in time, or when the Gemini analysis fails - a
        vision outage should degrade the forecast to
        no-vision, never crash the run.
        """

        video_path = self.vision.find_latest_video(
            self.windy_folder,
            run_time
        )

        if video_path is None:
            return None, False

        try:
            result = self.vision.analyze_video(
                video_path,
                self.vision_output_folder
            )

        except Exception as error:
            print(f"Vision analysis failed for {video_path.name}: {error}")
            return None, False

        return self.fusion.prepare_vision_features(result), True

    # --------------------------------------------------

    def run(self, dataframe):
        """
        Runs every official run-time on every historical day.
        Returns a summary DataFrame (one row per day/run-time),
        a cache of the raw, parameter-independent signals for
        each run (so parameters can be re-tuned afterwards
        without re-running Chronos), and a per-block detail
        DataFrame (our forecast vs Enercast vs actual, in kW,
        for every 15-min block of every run - used by the web
        dashboard's comparison view).
        """

        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

        dates = sorted(dataframe["timestamp"].dt.date.unique())

        records = []
        raw_signals_cache = {}
        detail_frames = []

        for date in dates:

            for run_time_str in self.official_run_times:

                hour, minute = map(int, run_time_str.split(":"))

                run_time = pd.Timestamp(date) + pd.Timedelta(
                    hours=hour,
                    minutes=minute
                )

                vision_features, has_vision = self.get_vision_features(
                    run_time
                )

                signals = self.predictor.compute_signals(dataframe, run_time)

                vision_adjustment = self.fusion.trend_adjustment_profile(
                    vision_features,
                    signals["horizon_minutes"]
                )

                forecast = self.predictor.blend_signals(
                    signals,
                    vision_adjustment=vision_adjustment
                )

                comparison, report = self.evaluator.evaluate(
                    forecast,
                    dataframe
                )

                if comparison.empty:
                    continue

                record = {
                    "date": date,
                    "run_time": run_time_str,
                    "n_blocks_scored": len(comparison),
                    "has_vision": has_vision,
                    "our_mae_kw": report["our_forecast"]["mae_kw"],
                    "our_avg_deviation_pct": report["our_forecast"]["avg_deviation_pct"],
                    "our_scheduling_penalty": report["our_forecast"]["scheduling_penalty"]
                }

                if "enercast" in report:
                    record["enercast_mae_kw"] = report["enercast"]["mae_kw"]
                    record["enercast_avg_deviation_pct"] = report["enercast"]["avg_deviation_pct"]
                    record["enercast_scheduling_penalty"] = report["enercast"]["scheduling_penalty"]

                records.append(record)

                raw_signals_cache[(date, run_time_str)] = {
                    "signals": signals,
                    "vision_adjustment": vision_adjustment,
                    "actual": comparison[["timestamp", "active_power_kw"]].copy()
                }

                detail_columns = ["timestamp", "final_forecast_kw", "active_power_kw"]

                if "enercast_kw" in comparison.columns:
                    detail_columns.append("enercast_kw")

                detail = comparison[detail_columns].copy()
                detail.insert(0, "date", date)
                detail.insert(1, "run_time", run_time_str)

                detail_frames.append(detail)

        results = pd.DataFrame(records)

        details = pd.concat(detail_frames, ignore_index=True)

        return results, raw_signals_cache, details

    # --------------------------------------------------

    def tune(self,
             raw_signals_cache,
             chronos_weight_candidates,
             performance_ratio_candidates,
             trend_halflife_candidates=(0,)):
        """
        Coarse grid search over chronos_weight,
        performance_ratio and the smart-persistence trend
        halflife, re-blending the already-cached raw signals
        from run() - cheap, pure-numpy, no Chronos re-runs.
        Picks the combo with the lowest average percentage
        deviation across every cached run.
        """

        grid_results = []

        for chronos_weight in chronos_weight_candidates:

            for performance_ratio in performance_ratio_candidates:

                for trend_halflife in trend_halflife_candidates:

                    deviations = []

                    for cached in raw_signals_cache.values():

                        forecast = self.predictor.blend_signals(
                            cached["signals"],
                            chronos_weight=chronos_weight,
                            performance_ratio=performance_ratio,
                            trend_halflife_min=trend_halflife,
                            vision_adjustment=cached["vision_adjustment"]
                        )

                        comparison = forecast.merge(
                            cached["actual"],
                            on="timestamp",
                            how="inner"
                        )

                        if comparison.empty:
                            continue

                        deviation = metrics.average_percentage_deviation(
                            comparison["final_forecast_kw"],
                            comparison["active_power_kw"],
                            self.predictor.capacity_kw
                        )

                        deviations.append(deviation)

                    avg_deviation = sum(deviations) / len(deviations)

                    grid_results.append({
                        "chronos_weight": chronos_weight,
                        "performance_ratio": performance_ratio,
                        "trend_halflife_min": trend_halflife,
                        "avg_deviation_pct": avg_deviation
                    })

        grid = pd.DataFrame(grid_results)

        best = grid.loc[grid["avg_deviation_pct"].idxmin()].to_dict()

        return grid, best
