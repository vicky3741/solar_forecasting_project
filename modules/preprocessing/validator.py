"""
=========================================================
Solar Forecasting Project
Data Validator
=========================================================
Validates all incoming datasets before preprocessing.
=========================================================
"""

from pathlib import Path
import pandas as pd


class DataValidator:

    def __init__(self):
        pass

    # ----------------------------------------------------

    def file_exists(self, file_path):

        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path} not found.")

        return True

    # ----------------------------------------------------

    def load_csv(self, file_path):

        self.file_exists(file_path)

        return pd.read_csv(file_path)

    # ----------------------------------------------------

    def check_empty(self, dataframe):

        if dataframe.empty:
            raise ValueError("Dataset is empty.")

        return True

    # ----------------------------------------------------

    def check_required_columns(self,
                               dataframe,
                               required_columns):

        missing = []

        for column in required_columns:

            if column not in dataframe.columns:
                missing.append(column)

        if missing:
            raise ValueError(
                f"Missing Columns : {missing}"
            )

        return True

    # ----------------------------------------------------

    def check_duplicates(self, dataframe):

        duplicates = dataframe.duplicated().sum()

        return duplicates

    # ----------------------------------------------------

    def check_missing_values(self,
                             dataframe):

        return dataframe.isnull().sum()

    # ----------------------------------------------------

    def validate(self,
                 file_path,
                 required_columns):

        dataframe = self.load_csv(file_path)

        self.check_empty(dataframe)

        self.check_required_columns(
            dataframe,
            required_columns
        )

        duplicates = self.check_duplicates(dataframe)

        missing = self.check_missing_values(dataframe)

        return {

            "dataframe": dataframe,

            "duplicates": duplicates,

            "missing": missing

        }