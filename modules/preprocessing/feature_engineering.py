"""
=========================================================
Solar Forecasting Project
Feature Engineering Module
=========================================================
Creates additional numerical features for forecasting.
=========================================================
"""

import pandas as pd


class FeatureEngineering:

    def __init__(self):
        pass

    # --------------------------------------------------

    def add_time_features(self, dataframe):

        dataframe["hour"] = dataframe["timestamp"].dt.hour
        dataframe["minute"] = dataframe["timestamp"].dt.minute
        dataframe["day"] = dataframe["timestamp"].dt.day
        dataframe["month"] = dataframe["timestamp"].dt.month
        dataframe["weekday"] = dataframe["timestamp"].dt.dayofweek

        return dataframe

    # --------------------------------------------------

    def add_lag_features(self, dataframe):

        dataframe["power_lag_1"] = dataframe["active_power_kw"].shift(1)
        dataframe["power_lag_2"] = dataframe["active_power_kw"].shift(2)

        return dataframe

    # --------------------------------------------------

    def add_rolling_features(self, dataframe):

        dataframe["rolling_mean_3"] = (
            dataframe["active_power_kw"]
            .rolling(window=3)
            .mean()
        )

        dataframe["rolling_std_3"] = (
            dataframe["active_power_kw"]
            .rolling(window=3)
            .std()
        )

        return dataframe

    # --------------------------------------------------

    def add_solar_features(self, dataframe):

        dataframe["is_daylight"] = (
            dataframe["ghi_w_m2"] > 20
        ).astype(int)

        dataframe["is_morning"] = (
            dataframe["hour"] < 12
        ).astype(int)

        dataframe["is_afternoon"] = (
            dataframe["hour"] >= 12
        ).astype(int)

        return dataframe

    # --------------------------------------------------

    def generate_features(self, dataframe):

        dataframe = self.add_time_features(dataframe)
        dataframe = self.add_lag_features(dataframe)
        dataframe = self.add_rolling_features(dataframe)
        dataframe = self.add_solar_features(dataframe)

        dataframe = dataframe.bfill()

        return dataframe