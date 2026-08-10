"""
=========================================================
Solar Forecasting Project - NEW APPROACH
One complete live run, end to end
=========================================================
The entry point for the new pipeline, the counterpart of
modules/orchestrator/pipeline.py for the production blend.
One scheduling time, one process:

    meter inbox  ->  scrape Windy  ->  feature table
                 ->  LLM decides   ->  block bias
                 ->  validator     ->  freeze horizon
                 ->  publish       ->  push to S3

Run it for the current block:

    python -m modules.orchestrator.llm_pipeline

or for a specific time:

    python -m modules.orchestrator.llm_pipeline --run-time "2026-08-10 09:45"

WHICH PLANT
-----------
Whichever SOLAR_PLANT names, exactly like the production
orchestrator. Nothing here names a plant: coordinates,
capacity, run times, freeze horizon and S3 prefixes all
arrive through `settings`.

WHAT IT PUBLISHES
-----------------
outputs/llm_schedules/<PLANT>_<date>_<time>_schedule.csv
    the forecast blocks, with anchor, the model's raw call,
    its per-block weight, and any validator adjustments

outputs/llm_schedules/<PLANT>_<date>_<time>_meta.json
    regime, reasoning, blocks adjusted, model used

outputs/schedules/llm_current_final_schedule.csv
    actual generation for every block already finished today,
    plus this run's forecast for everything ahead - the
    mentor brief's Current Final Schedule. Kept under its own
    name so it can never be mistaken for the production one.

EVERY STAGE DEGRADES RATHER THAN FAILS
--------------------------------------
A missing Windy scrape, an unreachable S3, no satellite clip -
each of these narrows what the model can see and is logged as
such, but none of them stops a schedule being published. The
one thing that CAN stop the run is the LLM itself, and that is
deliberate: publishing a physics-only schedule under the name
of the LLM pipeline would misreport what produced it.
=========================================================
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.fusion.llm_scheduler import LLMScheduler
from modules.preprocessing.windy_features import (
    WindyFeatureBuilder, load_meter_history
)
from modules.storage.s3_client import S3Storage
from utils import file_manager
from utils.logger import get_logger


class LLMOrchestrator:

    def __init__(self):

        self.logger = get_logger()

        self.builder = WindyFeatureBuilder()
        self.scheduler = LLMScheduler()
        self.storage = S3Storage()

        plant = settings["plant"]

        self.plant_tag = plant.get("code", "SIRMOUR")
        self.plant_name = plant.get("name", "plant")
        self.timezone = plant["timezone"]

        self.interval = settings["forecast"]["interval_minutes"]

        store = settings.get("storage", {})
        self.auto_pull = store.get("enabled", False) and store.get("auto_pull", False)
        self.auto_push = store.get("enabled", False) and store.get("auto_push", False)

        self.schedules_dir = Path(settings["outputs"]["schedules"])

    # --------------------------------------------------

    def sync_meter_inbox(self):
        """
        Copies any new meter exports from the team's drop folder into
        data/historical. Never raises - a missing inbox just means the
        run uses whatever is already ingested.
        """

        try:
            from modules.preprocessing.meter_inbox import MeterInbox

            result = MeterInbox().sync()

            if result["copied"]:
                self.logger.info(
                    f"Meter inbox: imported {len(result['copied'])} new file(s)"
                )

        except Exception as error:
            self.logger.warning(f"Meter inbox skipped ({error})")

    # --------------------------------------------------

    def scrape_windy(self, run_time):
        """
        Fetches this run's Windy values. Never raises: without them the
        feature table's Windy columns stay empty and the model is told
        so by windy_is_measured, which is a narrower run rather than a
        failed one.
        """

        if not settings.get("windy_scrape", {}).get("enabled", False):
            return None

        try:
            from modules.capture.windy_scraper import WindyScraper

            _, path = WindyScraper().scrape(run_time=run_time)

            return path

        except Exception as error:
            self.logger.warning(
                f"Windy scrape failed ({error}) - continuing without today's "
                "Windy values"
            )
            return None

    # --------------------------------------------------

    def previous_schedule(self):
        """
        The standing Current Final Schedule, whose near-term blocks the
        freeze horizon protects. {timestamp -> MW}, empty on the first
        run of the day.
        """

        previous = file_manager.load_dataframe(
            self.schedules_dir / "llm_current_final_schedule.csv",
            parse_dates=["timestamp"],
        )

        if previous is None or previous.empty:
            return {}

        if "source" in previous.columns:
            previous = previous[previous["source"] == "forecast"]

        if previous["timestamp"].dt.tz is None:
            previous["timestamp"] = previous["timestamp"].dt.tz_localize(
                self.timezone
            )

        return dict(zip(previous["timestamp"], previous["value_mw"]))

    # --------------------------------------------------

    def build_current_final_schedule(self, features, schedule, run_time):
        """
        Actual generation for every block already finished today, plus
        this run's forecast for everything ahead - the mentor brief's
        Current Final Schedule.
        """

        past = features[
            (features["is_past"] == 1) & features["actual_power_mw"].notna()
        ][["timestamp", "actual_power_mw"]].rename(
            columns={"actual_power_mw": "value_mw"}
        )
        past["source"] = "actual"

        future = schedule[["timestamp", "forecast_mw"]].rename(
            columns={"forecast_mw": "value_mw"}
        )
        future["source"] = "forecast"

        combined = pd.concat([past, future], ignore_index=True)
        combined["last_updated_run_time"] = run_time

        return combined.sort_values("timestamp")

    # --------------------------------------------------

    def push(self, run_label, paths):

        if not self.auto_push:
            return

        for name, path in paths.items():

            if path is None or not Path(path).exists():
                continue

            try:
                self.storage.push_output(
                    Path(path), "llm_schedules", f"{run_label}_{name}.csv"
                )
            except Exception as error:
                self.logger.warning(f"S3 push skipped for {name} ({error})")

    # --------------------------------------------------

    def run(self, run_time=None):

        run_time = pd.Timestamp(run_time) if run_time else pd.Timestamp.now()

        if run_time.tz is None:
            run_time = run_time.tz_localize(self.timezone)

        # Floor to the block grid. The scheduler fires at 09:45 but the
        # scrape takes minutes, so an unfloored run time puts every
        # forecast block off the meter's :00/:15/:30/:45 grid and the
        # whole run becomes ungradeable.
        run_time = run_time.floor(f"{self.interval}min")

        self.logger.info(
            f"LLM pipeline starting for {self.plant_name} at {run_time}"
        )

        self.sync_meter_inbox()

        self.scrape_windy(run_time)

        meter = load_meter_history()

        features, info = self.builder.build(run_time=run_time, meter=meter)

        schedule, meta = self.scheduler.decide(
            features,
            run_time,
            previous=self.previous_schedule(),
            satellite=info["satellite"],
        )

        paths = self.scheduler.save(schedule, meta, run_time)

        self.schedules_dir.mkdir(parents=True, exist_ok=True)

        current = self.build_current_final_schedule(features, schedule, run_time)

        current_path = self.schedules_dir / "llm_current_final_schedule.csv"
        file_manager.save_dataframe(current, current_path)

        run_label = run_time.strftime("%Y-%m-%d_%H-%M")

        self.push(run_label, {
            "schedule": paths.get("schedule"),
            "current_final": current_path,
        })

        self.logger.info(
            f"LLM pipeline complete: {len(schedule)} block(s), "
            f"regime {meta.get('regime')}, "
            f"{meta.get('adjusted_blocks', 0)} adjusted by the validator"
        )

        return schedule, meta


# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="One complete run of the LLM scheduling pipeline"
    )
    parser.add_argument("--run-time", default=None)

    args = parser.parse_args()

    orchestrator = LLMOrchestrator()

    try:
        schedule, meta = orchestrator.run(args.run_time)

    except Exception as error:
        orchestrator.logger.error(f"LLM pipeline failed: {error}")
        return 1

    print(f"\nregime    : {meta.get('regime')}")
    print(f"reasoning : {meta.get('reasoning')}")
    print(f"adjusted  : {meta.get('adjusted_blocks', 0)} block(s)")
    print(f"energy    : {schedule['forecast_mw'].sum() * 0.25:.3f} MWh\n")

    columns = [
        c for c in
        ("block", "time", "anchor_mw", "llm_raw_mw", "llm_weight", "forecast_mw")
        if c in schedule.columns
    ]

    print(schedule[columns].to_string(index=False, float_format="%.3f"))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
