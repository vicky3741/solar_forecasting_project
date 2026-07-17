"""
=========================================================
Solar Forecasting Project
Clear-Sky Model
=========================================================
Computes theoretical clear-sky irradiance and expected
generation for the plant using pvlib.

This is a physics-based baseline driven only by solar
geometry and plant configuration (latitude, longitude,
tilt, azimuth, capacity) - it needs no historical training
data, which makes it a useful foundation while historical
generation data is still limited.
=========================================================
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from config.config import settings


class ClearSkyModel:

    def __init__(self):

        plant = settings["plant"]

        self.latitude = plant["latitude"]
        self.longitude = plant["longitude"]
        self.altitude = plant.get("altitude_m", 0)
        self.timezone = plant["timezone"]

        self.tilt = plant["tilt_deg"]
        self.azimuth = plant["azimuth_deg"]
        self.capacity_kw = plant["capacity_mw"] * 1000

        clearsky_settings = settings["clearsky"]

        self.model = clearsky_settings["model"]
        self.performance_ratio = clearsky_settings["performance_ratio"]

        self.location = Location(
            self.latitude,
            self.longitude,
            tz=self.timezone,
            altitude=self.altitude,
            name=plant["name"]
        )

    # --------------------------------------------------

    def to_local_datetime_index(self, timestamps):

        times = pd.DatetimeIndex(pd.to_datetime(timestamps))

        if times.tz is None:
            times = times.tz_localize(self.timezone)
        else:
            times = times.tz_convert(self.timezone)

        return times

    # --------------------------------------------------

    def get_clearsky_irradiance(self, timestamps):
        """
        Returns clear-sky GHI, DNI, DHI for the given timestamps.
        """

        times = self.to_local_datetime_index(timestamps)

        clearsky = self.location.get_clearsky(
            times,
            model=self.model
        )

        return clearsky

    # --------------------------------------------------

    def get_poa_irradiance(self, timestamps):
        """
        Transposes clear-sky irradiance onto the plane of array
        using the configured tilt and azimuth.
        """

        times = self.to_local_datetime_index(timestamps)

        clearsky = self.location.get_clearsky(
            times,
            model=self.model
        )

        solar_position = self.location.get_solarposition(times)

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=self.azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=clearsky["dni"],
            ghi=clearsky["ghi"],
            dhi=clearsky["dhi"]
        )

        result = clearsky.copy()
        result["poa_global"] = poa["poa_global"]

        return result

    # --------------------------------------------------

    def estimate_clearsky_generation(self, timestamps):
        """
        Estimates expected clear-sky generation (kW) from POA
        irradiance, scaled by plant capacity and performance ratio.

        Gives an expected generation curve for any date/time using
        only solar geometry and plant configuration.
        """

        irradiance = self.get_poa_irradiance(timestamps)

        irradiance["expected_power_kw"] = (
            (irradiance["poa_global"] / 1000)
            * self.capacity_kw
            * self.performance_ratio
        ).clip(lower=0)

        return irradiance

    # --------------------------------------------------

    def compute_clear_sky_index(self,
                                dataframe,
                                ghi_column,
                                timestamp_column):
        """
        Computes the clear-sky index (kt) = actual GHI / clear-sky GHI.

        kt near 1   -> sky is clear
        kt well < 1 -> clouds are attenuating irradiance

        Used to validate cloud impact against irradiance evidence
        instead of treating cloud coverage as a direct solar loss.
        """

        dataframe = dataframe.copy()

        clearsky = self.get_clearsky_irradiance(
            dataframe[timestamp_column]
        )

        clearsky_ghi = clearsky["ghi"].to_numpy()
        actual_ghi = dataframe[ghi_column].to_numpy(dtype=float)

        # Below this threshold clear-sky GHI is near zero (dawn/dusk/night)
        # and the ratio becomes noise rather than signal.
        kt = np.divide(
            actual_ghi,
            clearsky_ghi,
            out=np.full_like(actual_ghi, np.nan, dtype=float),
            where=clearsky_ghi > 20
        )

        dataframe["clear_sky_ghi"] = clearsky_ghi
        dataframe["clear_sky_index"] = np.clip(kt, 0, 1.2)

        return dataframe
