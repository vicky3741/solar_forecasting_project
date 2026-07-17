"""
=========================================================
Solar Forecasting Project
File Manager
=========================================================
Generic save/load/append helpers for the CSV and JSON
output files the orchestrator produces. Pure I/O - no
forecasting or scheduling logic lives here.
=========================================================
"""

import json
from pathlib import Path

import pandas as pd


def ensure_parent(path):

    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
