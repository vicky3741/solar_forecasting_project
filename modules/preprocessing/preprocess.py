"""
=========================================================
Solar Forecasting Project
Data Preprocessing Module
=========================================================
Turns one plant's raw meter CSV into the single canonical
shape the rest of the pipeline forecasts on:

    timestamp, active_power_kw, ghi_w_m2, poa_w_m2, ...
    on a gap-free 15-minute grid, with is_real_measurement
    marking which blocks the meter actually recorded.

MULTI-PLANT (2026-08-08)
------------------------
Sirmour and the two Telangana plants come from different
vendors, and their exports disagree on three things that all
fail silently rather than loudly:

  * the timestamp header ("TimeStamp" vs "Timestamp");
  * the date order - Telangana writes dd-mm-yyyy, which pandas
    reads as mm-dd-yyyy unless told otherwise, quietly moving
    06-08-2026 from 6 August to 8 June;
  * the column names for generation and irradiance
    ("Active Power (kW)" vs "Active Power-Avg MFM-OUT (KW)",
    "GHI (W/m2)" vs "GHI_W (W/m2)").

Rather than sniff the file, each plant DECLARES its schema in
config (data_schema), and this module applies it. The defaults
in settings.yaml are Sirmour's, so Sirmour's behaviour is
unchanged; callers may still pass the old explicit arguments.
=========================================================
"""

import pandas as pd

from config.config import settings
from modules.preprocessing.feature_engineering import FeatureEngineering
from modules.preprocessing.validator import DataValidator
from modules.preprocessing.time_alignment import TimeAlignment
from modules.preprocessing.outlier_handler import OutlierHandler


def normalize_name(name):
    """
    The one column-name convention the whole pipeline speaks:
    lower_snake_case with brackets and slashes stripped.
    """

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


class DataPreprocessor:

    def __init__(self):

        self.validator = DataValidator()

        self.aligner = TimeAlignment()

        self.feature_engineering = FeatureEngineering()

        self.outlier_handler = OutlierHandler()

        schema = settings.get("data_schema", {})

        self.timestamp_column = schema.get("timestamp_column", "TimeStamp")
        self.dayfirst = bool(schema.get("dayfirst", False))
        self.column_map = {
            normalize_name(source): normalize_name(target)
            for source, target in (schema.get("column_map") or {}).items()
        }
        self.timestamp_shift_minutes = schema.get("timestamp_shift_minutes", 0)

    # --------------------------------------------------

    def standardize_columns(self, dataframe):
        """
        Normalises every header, then applies this plant's rename
        map so downstream code always sees active_power_kw /
        ghi_w_m2 regardless of what the vendor called them.
        """

        dataframe.columns = [
            normalize_name(column) for column in dataframe.columns
        ]

        if self.column_map:
            dataframe = dataframe.rename(columns=self.column_map)

        return dataframe

    # --------------------------------------------------

    def preprocess(self,
                   file_path,
                   required_columns=None,
                   timestamp_column=None):
        """
        `required_columns` and `timestamp_column` default to this
        plant's declared schema. They stay as arguments so the
        existing experiment scripts that pass "TimeStamp"
        explicitly keep working unchanged on Sirmour.
        """

        timestamp_column = timestamp_column or self.timestamp_column

        if required_columns is None:
            required_columns = [timestamp_column]

        result = self.validator.validate(
            file_path,
            required_columns
        )

        dataframe = result["dataframe"]

        dataframe = self.standardize_columns(
            dataframe
        )

        timestamp_column = normalize_name(timestamp_column)

        # dayfirst matters only for the ambiguous dd-mm-yyyy files.
        # Sirmour's ISO timestamps parse identically either way, but
        # the flag is passed explicitly rather than left to inference
        # so a vendor changing its export format cannot silently
        # relabel a month as a day.
        dataframe[timestamp_column] = pd.to_datetime(
            dataframe[timestamp_column],
            dayfirst=self.dayfirst
        )

        if self.timestamp_shift_minutes:
            dataframe[timestamp_column] = (
                dataframe[timestamp_column]
                + pd.Timedelta(minutes=self.timestamp_shift_minutes)
            )

        dataframe = dataframe.drop_duplicates()

        dataframe = self.outlier_handler.handle(dataframe)

        dataframe = dataframe.ffill().bfill()

        dataframe = self.aligner.align(
            dataframe,
            timestamp_column
        )

        dataframe = self.feature_engineering.generate_features(
            dataframe

        )
        return dataframe
