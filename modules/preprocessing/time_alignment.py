"""
=========================================================
Solar Forecasting Project
Time Alignment Module
=========================================================
Aligns all datasets to a common 15-minute timeline.
=========================================================
"""

import pandas as pd


class TimeAlignment:

    def __init__(self, frequency="15min"):
        self.frequency = frequency

    # --------------------------------------------------

    def sort_by_time(self, dataframe, timestamp_column):

        dataframe = dataframe.sort_values(timestamp_column)

        return dataframe

    # --------------------------------------------------

    def set_timestamp_index(self,
                            dataframe,
                            timestamp_column):

        dataframe = dataframe.set_index(timestamp_column)

        return dataframe

    # --------------------------------------------------

    def resample_data(self,
                      dataframe):

        dataframe = dataframe.resample(
            self.frequency
        ).mean()

        return dataframe

    # --------------------------------------------------

    def interpolate_missing(self,
                            dataframe):

        dataframe = dataframe.interpolate()

        return dataframe

    # --------------------------------------------------

    def reset_index(self,
                    dataframe):

        dataframe = dataframe.reset_index()

        return dataframe

    # --------------------------------------------------

    def align(self,
              dataframe,
              timestamp_column):

        dataframe = self.sort_by_time(
            dataframe,
            timestamp_column
        )

        dataframe = self.set_timestamp_index(
            dataframe,
            timestamp_column
        )

        dataframe = self.resample_data(
            dataframe
        )

        dataframe = self.interpolate_missing(
            dataframe
        )

        dataframe = self.reset_index(
            dataframe
        )

        return dataframe