"""
=========================================================
Solar Forecasting Project
Evaluation Metrics
=========================================================
Error and deviation metrics used to score a forecast
against actual generation, per the mentor's evaluation
brief (schedule deviation, forecast error, scheduling
penalty).
=========================================================
"""

import numpy as np


def mean_absolute_error(forecast, actual):

    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)

    return float(np.mean(np.abs(forecast - actual)))


# --------------------------------------------------

def root_mean_squared_error(forecast, actual):

    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)

    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


# --------------------------------------------------

def percentage_deviation(forecast, actual, capacity_kw):
    """
    Deviation of each block's forecast from actual, as a
    percentage of plant capacity rather than of the actual
    value itself - actual generation is near zero at dawn
    and dusk, which would make a %-of-actual metric blow up.
    """

    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)

    return np.abs(forecast - actual) / capacity_kw * 100


# --------------------------------------------------

def average_percentage_deviation(forecast, actual, capacity_kw):

    return float(
        np.mean(percentage_deviation(forecast, actual, capacity_kw))
    )


# --------------------------------------------------

def scheduling_penalty(forecast,
                       actual,
                       capacity_kw,
                       free_band_pct=15,
                       penalty_rate_per_pct=1.0):
    """
    PLACEHOLDER - NOT the real grid scheduling regulation.

    Real intraday scheduling penalties (e.g. India's DSM
    regulations) use specific deviation slabs and rupee-per-MWh
    rates set by the regulator, which are not available here.
    This is a simple stand-in so the pipeline has a working
    penalty signal end-to-end: no penalty within `free_band_pct`
    deviation, then a flat rate per percentage point beyond it.

    Replace `free_band_pct` / `penalty_rate_per_pct` with the
    actual regulation's slabs once available.
    """

    deviation_pct = percentage_deviation(forecast, actual, capacity_kw)

    excess = np.clip(deviation_pct - free_band_pct, 0, None)

    return float(np.sum(excess * penalty_rate_per_pct))
