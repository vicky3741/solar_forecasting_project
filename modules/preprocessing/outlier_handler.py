"""
=========================================================
Solar Forecasting Project
Outlier Handler
=========================================================
Cleans physically-impossible sensor readings before they
reach feature engineering or forecasting:

  - Inverter night-time "phantom power" (a small negative
    baseline reading, e.g. -5.7 kW, from standby/parasitic
    draw and sensor offset) and negative irradiance noise -
    both physically impossible, clipped to zero.
  - Generation spikes above what the plant can physically
    produce, capped at a small tolerance over capacity.
=========================================================
"""

import numpy as np

from config.config import settings


class OutlierHandler:

    def __init__(self, capacity_tolerance=1.05):

        self.capacity_kw = settings["plant"]["capacity_mw"] * 1000
        self.capacity_tolerance = capacity_tolerance

    # --------------------------------------------------

    def clip_negative_values(self, dataframe, columns):
        """
        Power and irradiance cannot physically be negative -
        negative readings are sensor noise or inverter
        standby draw, not real generation.
        """

        for column in columns:
            if column in dataframe.columns:
                dataframe[column] = np.clip(dataframe[column], 0, None)

        return dataframe

    # --------------------------------------------------

    def clip_capacity_ceiling(self, dataframe, column="active_power_kw"):
        """
        Caps generation readings at a small tolerance above
        nameplate capacity - anything higher is a sensor or
        unit-conversion fault, not real output.
        """

        if column in dataframe.columns:

            ceiling = self.capacity_kw * self.capacity_tolerance

            dataframe[column] = np.clip(dataframe[column], None, ceiling)

        return dataframe

    # --------------------------------------------------

    def handle(self, dataframe):

        dataframe = self.clip_negative_values(
            dataframe,
            columns=["active_power_kw", "ghi_w_m2", "poa_w_m2"]
        )

        dataframe = self.clip_capacity_ceiling(dataframe)

        return dataframe
