"""
=========================================================
Solar Forecasting Project
Global Constants
=========================================================
This file contains constant values used throughout
the project.
=========================================================
"""

# --------------------------------------------------------
# Time Constants
# --------------------------------------------------------

TIMEZONE = "Asia/Kolkata"

DATE_FORMAT = "%Y-%m-%d"

TIME_FORMAT = "%H:%M:%S"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------
# Forecast Constants
# --------------------------------------------------------

FORECAST_INTERVAL = 15

FORECAST_END_TIME = "19:00"


# --------------------------------------------------------
# File Formats
# --------------------------------------------------------

CSV_EXTENSION = ".csv"

JSON_EXTENSION = ".json"

IMAGE_EXTENSION = ".png"

VIDEO_EXTENSION = ".mp4"


# --------------------------------------------------------
# Supported Image Formats
# --------------------------------------------------------

SUPPORTED_IMAGE_FORMATS = [
    ".png",
    ".jpg",
    ".jpeg"
]


# --------------------------------------------------------
# Supported Video Formats
# --------------------------------------------------------

SUPPORTED_VIDEO_FORMATS = [
    ".mp4",
    ".avi",
    ".mov"
]


# --------------------------------------------------------
# Logging
# --------------------------------------------------------

LOG_FILE = "solar_forecasting.log"


# --------------------------------------------------------
# Project Information
# --------------------------------------------------------

PROJECT_NAME = "Solar Forecasting Project"

VERSION = "1.0.0"