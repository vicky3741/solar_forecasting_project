"""
=========================================================
Solar Forecasting Project
Logger
=========================================================
Shared loguru logger, writing to the configured log
directory in addition to the console.
=========================================================
"""

from pathlib import Path

from loguru import logger

from config.config import settings

_configured = False


def get_logger():

    global _configured

    if not _configured:

        log_settings = settings["logging"]

        log_dir = Path(log_settings["log_directory"])
        log_dir.mkdir(parents=True, exist_ok=True)

        # One file per plant. The three automations run as three
        # separate services on the same box, and letting them share
        # solar_forecasting.log would interleave their lines - making
        # "why did the 12:45 run fail" unanswerable without first
        # working out which plant each line belonged to. Sirmour keeps
        # its existing filename so its log history stays continuous.
        log_file = log_settings.get("log_file", "solar_forecasting.log")

        # enqueue=True routes writes through a queue so the
        # scheduler and the capture subprocesses it spawns can
        # share one log file. Without it, whichever process
        # triggers the daily rotation hits
        # "PermissionError: file is being used by another
        # process" because the other still holds the handle.
        logger.add(
            log_dir / log_file,
            level=log_settings["level"],
            rotation="1 day",
            retention="30 days",
            enqueue=True
        )

        _configured = True

    return logger
