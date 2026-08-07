"""
=========================================================
Solar Forecasting Project
Scheduler
=========================================================
Triggers the full automation at the mentor brief's 7
official daily run times:

  1. Capture a Windy satellite clip and upload it to S3
  2. Run the orchestrator, which pulls the latest same-day
     video + meter data from S3, forecasts, and pushes the
     forecast and Current Final Schedule back to S3

The capture runs as a SEPARATE short-lived subprocess
rather than inside this long-running process. Playwright
previously failed when driven from the long-lived
Task Scheduler process while working fine standalone, so
each capture now gets a clean process and a hard timeout -
a hung or broken capture can never stall or kill the
scheduler, and the forecast still runs either way.

The Orchestrator is likewise built PER RUN, not once at
startup. Constructing it loads torch and the Chronos
weights (~420 MB), and holding that resident between runs
left too little headroom on the ~900 MB EC2 box for
Chromium to record a clip - the capture was OOM-killed
mid-recording (2026-07-26). Building it per run keeps this
process near-idle for the 23 hours a day it is waiting, and
means capture and forecast never hold their peak memory at
the same time.

ONE SCHEDULER PER PLANT (2026-08-08)
------------------------------------
Since Kasipet and Bhupalpally joined Sirmour, THREE of these
run side by side - one process per plant, each started with
its own SOLAR_PLANT (see automation/). They are genuinely
independent:

  * each holds its OWN single-instance lock port, so they
    never mistake each other for a duplicate of themselves and
    refuse to start (49732 / 49733 / 49734);
  * each writes its own log file;
  * each fires at its own plant's run times - Telangana has an
    extra 17:15 run that Sirmour does not;
  * each passes SOLAR_PLANT explicitly to the capture and
    forecast subprocesses it spawns, rather than trusting the
    inherited environment, so a launcher that sets the variable
    in an unexpected way cannot make one plant's scheduler
    forecast another plant.

A crash, a hung capture or an exhausted API quota on one plant
therefore stops that plant only.
=========================================================
"""

import gc
import os
import socket
import subprocess
import sys
import time

import schedule as schedule_lib

from config.config import settings
from utils.logger import get_logger

# NOTE: modules.orchestrator.pipeline is deliberately NOT imported here.
# Importing it pulls in torch and chronos, which costs ~450 MB of resident
# memory the moment this module loads - even though nothing runs until a
# scheduled time fires. On the ~900 MB EC2 box that idle cost is what
# starves Chromium during a capture. The import happens inside run_now().


class AlreadyRunning(Exception):
    """Raised when another scheduler instance already holds the lock."""


def acquire_single_instance_lock(port):
    """
    Single-instance guard via a bound localhost socket.

    The launcher (run_scheduler.bat) starts Python detached and
    exits immediately, so Windows Task Scheduler never sees the
    task as "running" and its IgnoreNew guard is useless - every
    logon + 6:30 trigger, plus any manual start, stacked ANOTHER
    scheduler. On 2026-07-24 three ran at once and collided over
    the same video filenames ("Access is denied", "file being
    used by another process").

    Binding a fixed port is an atomic OS-level lock: the second
    process to try it fails, so only one scheduler can ever run.
    Unlike a PID lock file it self-heals - the OS frees the port
    the instant the holding process dies, so there is no stale
    lock to clean up after a crash. The socket handle is returned
    and must be kept alive for the whole process lifetime.
    """

    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        lock_socket.bind(("127.0.0.1", port))
    except OSError:
        lock_socket.close()
        raise AlreadyRunning(
            f"another scheduler already holds port {port}"
        )

    return lock_socket


class Scheduler:

    def __init__(self):

        self.logger = get_logger()

        self.plant_key = settings["plant"]["key"]
        self.plant_name = settings["plant"].get("name", self.plant_key)

        # Handed to every subprocess so the child resolves the SAME
        # plant this scheduler resolved, whatever the parent's
        # environment happens to look like.
        self.child_env = {**os.environ, "SOLAR_PLANT": self.plant_key}

        self.official_run_times = settings["forecast"]["run_times"]

        capture_settings = settings.get("windy_capture", {})
        self.capture_enabled = capture_settings.get("enabled", False)
        self.capture_timeout = capture_settings.get(
            "subprocess_timeout_seconds", 180
        )

        # A forecast run takes ~40s (model load, S3 pull, vision, blend,
        # push); this is a hard kill for one that hangs, generous enough
        # to survive a slow network day.
        self.forecast_timeout = settings.get("scheduler", {}).get(
            "forecast_timeout_seconds", 600
        )

        # Fixed localhost port used purely as a single-instance lock
        # (see acquire_single_instance_lock). Not a network service.
        self.lock_port = settings.get("scheduler", {}).get(
            "single_instance_port", 49732
        )
        self._lock_socket = None

        # Seconds after the official run time that THIS plant actually
        # fires. See register_jobs.
        self.run_offset_seconds = int(
            settings.get("scheduler", {}).get("run_offset_seconds", 0)
        )

    # --------------------------------------------------

    def capture_video(self):
        """
        Runs one Windy capture in its own process. Never
        raises - the forecast must go ahead even if the
        capture fails, since S3 usually has a recent clip
        from Team 3's feed to fall back on.
        """

        if not self.capture_enabled:
            return

        try:
            result = subprocess.run(
                [sys.executable, "-m", "modules.capture.windy_capture"],
                capture_output=True,
                text=True,
                timeout=self.capture_timeout,
                env=self.child_env
            )

            if result.returncode == 0:
                self.logger.info("Windy capture completed")
            else:
                self.logger.error(
                    f"Windy capture exited {result.returncode}: "
                    f"{result.stderr.strip()[-500:]}"
                )

        except subprocess.TimeoutExpired:
            self.logger.error(
                f"Windy capture timed out after {self.capture_timeout}s "
                "- continuing to the forecast"
            )

        except Exception as error:
            self.logger.error(f"Windy capture could not start: {error}")

    # --------------------------------------------------

    def run_forecast(self):
        """
        Runs one forecast in its own process, for the same reason the
        capture gets one: torch and the Chronos weights cost ~450 MB,
        and CPython never returns freed arenas to the OS. Running
        in-process left this scheduler holding 577 MB after only three
        runs (2026-07-27), leaving ~283 MB free - less than Chromium
        needs for the next capture, so a 7-run day would eventually
        have starved itself. A separate process gives every byte back
        on exit.
        """

        try:
            result = subprocess.run(
                [sys.executable, "-m", "modules.orchestrator.pipeline"],
                capture_output=True,
                text=True,
                timeout=self.forecast_timeout,
                env=self.child_env
            )

            if result.returncode == 0:
                self.logger.info("Forecast run completed")
            else:
                self.logger.error(
                    f"Forecast exited {result.returncode}: "
                    f"{result.stderr.strip()[-500:]}"
                )

        except subprocess.TimeoutExpired:
            self.logger.error(
                f"Forecast timed out after {self.forecast_timeout}s"
            )

        except Exception as error:
            self.logger.error(f"Forecast could not start: {error}")

    # --------------------------------------------------

    def run_now(self):

        self.logger.info(f"Scheduled trigger fired for {self.plant_name}")

        # Capture first, in its own process, so Chromium has the
        # machine to itself before the forecast loads Chronos.
        self.capture_video()

        self.run_forecast()

        gc.collect()

    # --------------------------------------------------

    def trigger_time(self, run_time):
        """
        The wall-clock time this plant's job fires: its official run
        time plus this plant's stagger offset, as HH:MM:SS.

        The stagger exists because all three plants share the same
        official run times and one forecast peaks near 450 MB on a
        ~900 MB box - three at the same instant is what would OOM it.
        It does NOT move the schedule: the orchestrator floors its run
        time to the 15-minute block, so a run fired at 06:50 still
        produces the 06:45 schedule, and the Windy clip it records
        still falls inside the 20-minute match tolerance for that slot.
        Offsets stay well under the 90 minutes between run times.
        """

        hour, minute = map(int, run_time.split(":"))

        seconds = hour * 3600 + minute * 60 + self.run_offset_seconds

        return time.strftime("%H:%M:%S", time.gmtime(seconds % 86400))

    # --------------------------------------------------

    def register_jobs(self):

        for run_time in self.official_run_times:

            fires_at = self.trigger_time(run_time)

            schedule_lib.every().day.at(fires_at).do(self.run_now)

            if self.run_offset_seconds:
                self.logger.info(
                    f"Registered daily run at {run_time} for "
                    f"{self.plant_name} (fires {fires_at}, staggered "
                    f"{self.run_offset_seconds}s)"
                )
            else:
                self.logger.info(
                    f"Registered daily run at {run_time} for {self.plant_name}"
                )

    # --------------------------------------------------

    def start(self):

        # Refuse to start if another scheduler is already running,
        # so duplicate launches (task retriggers, manual starts)
        # exit cleanly instead of stacking up and colliding.
        try:
            self._lock_socket = acquire_single_instance_lock(self.lock_port)
        except AlreadyRunning as error:
            self.logger.warning(
                f"Not starting - {error}. Another scheduler is already "
                "running; this instance will exit."
            )
            return

        self.register_jobs()

        self.logger.info(
            f"Scheduler started for {self.plant_name} "
            f"(lock port {self.lock_port}) - waiting for the next run time"
        )

        while True:

            schedule_lib.run_pending()

            time.sleep(30)


if __name__ == "__main__":
    Scheduler().start()
