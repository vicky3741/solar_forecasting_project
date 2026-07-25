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
=========================================================
"""

import gc
import socket
import subprocess
import sys
import time

import schedule as schedule_lib

from config.config import settings
from modules.orchestrator.pipeline import Orchestrator
from utils.logger import get_logger


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

        self.official_run_times = settings["forecast"]["run_times"]

        capture_settings = settings.get("windy_capture", {})
        self.capture_enabled = capture_settings.get("enabled", False)
        self.capture_timeout = capture_settings.get(
            "subprocess_timeout_seconds", 180
        )

        # Fixed localhost port used purely as a single-instance lock
        # (see acquire_single_instance_lock). Not a network service.
        self.lock_port = settings.get("scheduler", {}).get(
            "single_instance_port", 49732
        )
        self._lock_socket = None

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
                timeout=self.capture_timeout
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

    def run_now(self):

        self.logger.info("Scheduled trigger fired")

        # Capture first, in its own process, so Chromium has the
        # machine to itself before the forecast loads Chronos.
        self.capture_video()

        try:
            # Built here rather than in __init__ (see module docstring):
            # the ~420 MB of torch/Chronos weights are released again
            # once the run returns, instead of sitting resident all day.
            Orchestrator().run()

        except Exception as error:
            self.logger.error(f"Scheduled run failed: {error}")

        finally:
            gc.collect()

    # --------------------------------------------------

    def register_jobs(self):

        for run_time in self.official_run_times:

            schedule_lib.every().day.at(run_time).do(self.run_now)

            self.logger.info(f"Registered daily run at {run_time}")

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

        self.logger.info("Scheduler started - waiting for the next run time")

        while True:

            schedule_lib.run_pending()

            time.sleep(30)


if __name__ == "__main__":
    Scheduler().start()
