"""
=========================================================
Solar Forecasting Project
Scheduler
=========================================================
Triggers the orchestrator at the mentor brief's 7 official
daily run times.

NOTE: this operates on whatever data is currently on disk
(data/historical, data/windy/videos) - there is no live
meter/video feed wired up yet, so running this continuously
against static historical files will just re-forecast the
same data until new files are actually dropped in.
=========================================================
"""

import time

import schedule as schedule_lib

from config.config import settings
from modules.orchestrator.pipeline import Orchestrator
from utils.logger import get_logger


class Scheduler:

    def __init__(self):

        self.orchestrator = Orchestrator()

        self.logger = get_logger()

        self.official_run_times = settings["forecast"]["run_times"]

    # --------------------------------------------------

    def run_now(self):

        self.logger.info("Scheduled trigger fired")

        try:
            self.orchestrator.run()

        except Exception as error:
            self.logger.error(f"Scheduled run failed: {error}")

    # --------------------------------------------------

    def register_jobs(self):

        for run_time in self.official_run_times:

            schedule_lib.every().day.at(run_time).do(self.run_now)

            self.logger.info(f"Registered daily run at {run_time}")

    # --------------------------------------------------

    def start(self):

        self.register_jobs()

        self.logger.info("Scheduler started - waiting for the next run time")

        while True:

            schedule_lib.run_pending()

            time.sleep(30)
