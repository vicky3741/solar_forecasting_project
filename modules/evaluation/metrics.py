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

from modules.evaluation import dsm_penalty


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

def scheduling_penalty(forecast, actual, capacity_kw):
    """
    The day's DSM penalty for these blocks, IN RUPEES.

    REPLACED 2026-08-14. This used to be a placeholder - a 15%
    free band and a flat "rate per percentage point", returning
    a unitless score - written before the regulator's slabs
    were known. Every other penalty figure this project quotes
    (the penalty report, every experiment, both pipelines' cost
    comparisons) already used the real slabs, so this function
    was the one number in the codebase that disagreed with the
    rest, and it is the number the run report and the backtest
    CSV print. That is the likeliest source of the mentor's
    "your Enercast penalty not matching with actual penalty":
    the same Enercast schedule priced here and priced in the
    penalty report came out different, because the free band
    was 15% instead of 10% and the answer was not in rupees.

    It now delegates to modules/evaluation/dsm_penalty.py, the
    single implementation of the mentor's SIRMOUR penalty
    logic. Figures printed by this function BEFORE 2026-08-14
    are on the old placeholder basis and are not comparable.
    """

    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # DSM signs the deviation as actual - scheduled; only its size is
    # charged, so this matches the report's column either way.
    deviation_mw = (actual - forecast) / 1000.0

    return float(np.sum(dsm_penalty.penalty_rs(
        deviation_mw, capacity_mw=capacity_kw / 1000.0
    )))
