"""
=========================================================
Solar Forecasting Project
Data Loader
=========================================================
Loads all project datasets.
=========================================================
"""

from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self):

        self.supported_extension = ".csv"

    # --------------------------------------------------

    def load_csv(self, file_path):

        return pd.read_csv(file_path)

    # --------------------------------------------------

    def load_folder(self, folder_path):

        folder = Path(folder_path)

        csv_files = sorted(folder.glob("*.csv"))

        dataframes = []

        for csv in csv_files:

            df = pd.read_csv(csv)

            dataframes.append(df)

        return dataframes

    # --------------------------------------------------

    def merge_folder(self, folder_path):

        dataframes = self.load_folder(folder_path)

        dataframe = pd.concat(
            dataframes,
            ignore_index=True
        )

        return dataframe