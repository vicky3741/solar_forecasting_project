"""
=========================================================
Solar Forecasting Project
Data Preprocessing Module
=========================================================
"""

import pandas as pd

from modules.preprocessing.feature_engineering import FeatureEngineering
from modules.preprocessing.validator import DataValidator
from modules.preprocessing.time_alignment import TimeAlignment
from modules.preprocessing.outlier_handler import OutlierHandler


class DataPreprocessor:

    def __init__(self):

        self.validator = DataValidator()

        self.aligner = TimeAlignment()

        self.feature_engineering = FeatureEngineering()

        self.outlier_handler = OutlierHandler()

    # --------------------------------------------------

    def standardize_columns(self, dataframe):

        dataframe.columns = (

            dataframe.columns

            .str.strip()

            .str.lower()

            .str.replace(" ", "_")

            .str.replace("(", "")

            .str.replace(")", "")

            .str.replace("/", "_")

        )

        return dataframe

    # --------------------------------------------------

    def preprocess(self,
                   file_path,
                   required_columns,
                   timestamp_column):

        result = self.validator.validate(
            file_path,
            required_columns
        )

        dataframe = result["dataframe"]

        dataframe = self.standardize_columns(
            dataframe
        )

        timestamp_column = (
            timestamp_column
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

        dataframe[timestamp_column] = pd.to_datetime(
            dataframe[timestamp_column]
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