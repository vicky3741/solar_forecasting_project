"""
=========================================================
Solar Forecasting Project
File Manager
=========================================================
Generic save/load/append helpers for the CSV and JSON
output files the orchestrator produces, plus the handful of
path lookups that must resolve per plant. No forecasting or
scheduling logic lives here.
=========================================================
"""

import json
import re
from pathlib import Path

import pandas as pd

from config.config import settings


# --------------------------------------------------

def processed_data_path():
    """
    This plant's combined processed dataset - the file
    tests/test_preprocessing.py writes and every backtest and
    tuning experiment reads.

    Resolved through config rather than hard-coded, because the
    string "data/processed/processed_data.csv" appeared in a dozen
    scripts: with three plants in the codebase, each of those would
    otherwise silently read SIRMOUR's data no matter which plant it
    was asked to analyse, and produce a plausible-looking wrong
    answer instead of an error.
    """

    return Path(settings["paths"]["processed_data"]) / "processed_data.csv"


def reports_path(*parts):
    """
    A file inside this plant's reports folder.
    """

    return Path(settings["outputs"]["reports"]).joinpath(*parts)


def ensure_parent(path):

    Path(path).parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------

# A calendar date anywhere in a filename, in any of the separators the
# three plants' vendors use: 2026_08_06_SOLAR_INV.csv (Sirmour),
# kasipet_20260806.csv and bhupalpally_20260806.csv (Telangana).
_FILENAME_DATE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


def date_in_name(name):
    """
    The calendar date encoded in a filename, or None.
    """

    match = _FILENAME_DATE.search(str(name))

    if not match:
        return None

    try:
        return pd.Timestamp(
            year=int(match.group(1)),
            month=int(match.group(2)),
            day=int(match.group(3)),
        ).date()
    except ValueError:
        return None


def find_daily_file(folder, day, extension="*.csv"):
    """
    The meter file for one calendar day inside a plant's historical
    folder, found by the date in its NAME rather than by a fixed
    naming template.

    Each plant's vendor names its daily export differently
    (2026_08_06_SOLAR_INV.csv vs kasipet_20260806.csv), and hard-coding
    one template is what would otherwise have to be duplicated into
    every report script. Returns None when that day is not present.
    """

    day = pd.Timestamp(day).date()

    for path in sorted(Path(folder).glob(extension)):
        if date_in_name(path.name) == day:
            return path

    return None


# --------------------------------------------------

def save_dataframe(dataframe, path):

    ensure_parent(path)

    dataframe.to_csv(path, index=False)


# --------------------------------------------------

def load_dataframe(path, parse_dates=None):

    path = Path(path)

    if not path.exists():
        return None

    return pd.read_csv(path, parse_dates=parse_dates)


# --------------------------------------------------

def append_dataframe(dataframe, path):
    """
    Appends rows to an existing CSV, creating it if it does
    not exist yet. Used for the all-runs forecast archive.
    """

    path = Path(path)

    ensure_parent(path)

    write_header = not path.exists()

    dataframe.to_csv(path, mode="a", index=False, header=write_header)


# --------------------------------------------------

def save_json(data, path):

    ensure_parent(path)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


# --------------------------------------------------

def load_json(path):

    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
