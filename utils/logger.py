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

        logger.add(
            log_dir / "solar_forecasting.log",
            level=log_settings["level"],
            rotation="1 day",
            retention="30 days"
        )

        _configured = True

    return logger
