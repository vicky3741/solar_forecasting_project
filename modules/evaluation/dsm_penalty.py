"""
=========================================================
Solar Forecasting Project
DSM Penalty - the ONE implementation
=========================================================
The mentor's "SIRMOUR Penalty Logic" (2026-08-14), written
down once so every report, experiment and pipeline prices a
block the same way. Before this module the same slab table
was retyped in eight places (both pipelines' penalty
reports, four experiment scripts, the new pipeline's
scorer); they all agreed, but nothing forced them to.

THE RULE
--------
Sirmour settles on the STANDARD DSM calculation, not the
OSEPL payable/receivable settlement. Under-generation and
over-generation cost the same, because only the SIZE of the
deviation is charged:

    deviation_mw      = actual_meter_mw - scheduled_mw
    deviation_percent = deviation_mw / capacity_mw * 100
    deviation_kwh     = |deviation_mw| * 0.25 * 1000

Madhya Pradesh Solar bands (config: dsm.bands):

    0-10%   free
    10-15%  Rs 0.50 / kWh
    15-20%  Rs 0.75 / kWh
    20%+    Rs 1.00 / kWh

The mentor states the charge as "energy apportioned by the
share of the deviation percentage that falls in each band":

    penalty = SUM over bands of
              deviation_kwh * (band_span_pct / |deviation_pct|) * rate

That is the same number as the marginal-slab form this
project has always used, and the algebra is worth keeping
because the two look nothing alike:

    deviation_kwh * span/|dev_pct|
      = 250 * |dev_mw| * span / (|dev_mw| / cap * 100)
      = 250 * cap * span / 100
      = 250 * (the band's WIDTH IN MW)

i.e. the |dev_mw| cancels and each band charges its own MW
width, which is exactly `min(dev, upper_edge) - lower_edge`.
`penalty_rs()` below computes the marginal form (vectorized,
no divide-by-zero at dev = 0) and tests/test_dsm_penalty.py
asserts it against the mentor's summation form block by
block.

MISSING BLOCKS ARE NOT ZERO
---------------------------
A block with no meter reading, or no schedule, has NO
penalty - not a penalty of 0. `day_settlement()` returns
None for those and reports the day as Pending / Partially
Calculated, per the mentor's daily logic.

WHAT IS NOT IN HERE
-------------------
The OSEPL special settlement (separate payable/receivable
legs). Sirmour does not use it; `dsm.settlement` is carried
in config so a plant that does can be spotted rather than
silently priced with the wrong rule.
=========================================================
"""

import numpy as np
import pandas as pd

from config.config import settings


# Bands as (from_pct, to_pct_or_None, rate_rs_per_kwh), from config so a
# plant whose regulator publishes a different table overrides it in its
# overlay instead of editing code.
BANDS = [
    (float(low), None if high is None else float(high), float(rate))
    for low, high, rate in settings["dsm"]["bands"]
]

SETTLEMENT = settings["dsm"].get("settlement", "standard")

# Display only. The PPA rate never enters the penalty - it is what the
# scheduled energy would have earned, shown alongside so a rupee penalty
# can be read against the rupees the block was worth.
PPA_RATE = float(settings["dsm"].get("ppa_rate_rs_per_kwh", 0.0))

CAPACITY_MW = float(settings["plant"]["capacity_mw"])

# MW deviation held for one block -> kWh. 0.25 h x 1000 kW/MW for the
# standard 15-minute block; read from config so it follows the interval.
BLOCK_HOURS = float(settings["forecast"]["interval_minutes"]) / 60.0
BLOCK_ENERGY_FACTOR = BLOCK_HOURS * 1000.0

BLOCKS_PER_DAY = int(round(24 * 60 / float(settings["forecast"]["interval_minutes"])))


# --------------------------------------------------

def band_edges_mw(capacity_mw=None):
    """
    The band boundaries in MW for this plant: [lower_mw, upper_mw_or_None,
    rate]. 10% of a 5.1 MW plant is 0.51 MW - the free dead band every
    penalty chart draws.
    """

    capacity = CAPACITY_MW if capacity_mw is None else float(capacity_mw)

    return [
        (low / 100 * capacity,
         None if high is None else high / 100 * capacity,
         rate)
        for low, high, rate in BANDS
    ]


# --------------------------------------------------

def penalty_rs(deviation_mw, capacity_mw=None):
    """
    Rupees for one block, or for a whole array of blocks.

    `deviation_mw` is (actual - scheduled); only its magnitude is
    charged, so the sign is irrelevant to the amount. Scalars come back
    as a float, arrays/Series as an ndarray, so this can be used both
    per block and as a column expression.
    """

    deviation = np.abs(np.asarray(deviation_mw, dtype=float))

    total = np.zeros_like(deviation, dtype=float)

    for lower_mw, upper_mw, rate in band_edges_mw(capacity_mw):

        if rate == 0:
            continue

        capped = deviation if upper_mw is None else np.minimum(deviation, upper_mw)

        total += np.clip(capped - lower_mw, 0, None) * rate

    total = total * BLOCK_ENERGY_FACTOR

    return float(total) if np.isscalar(deviation_mw) or total.ndim == 0 else total


# --------------------------------------------------

def block_settlement(scheduled_mw, actual_mw, capacity_mw=None):
    """
    One block, in the mentor's result shape.

    Returns None for `penalty_amount` (and every derived figure) when
    either side is missing - a block nobody scheduled, or one the meter
    has not reported yet, is Pending, NOT free.

    payable/receivable stay 0 on purpose: they are the OSEPL settlement's
    legs, and this plant does not use it. They are present so a report
    laid out for either plant type has the columns it expects.
    """

    capacity = CAPACITY_MW if capacity_mw is None else float(capacity_mw)

    missing = (
        scheduled_mw is None or actual_mw is None
        or pd.isna(scheduled_mw) or pd.isna(actual_mw)
    )

    if missing:
        return {
            "scheduled_mw": None if scheduled_mw is None or pd.isna(scheduled_mw)
                            else float(scheduled_mw),
            "actual_meter_mw": None if actual_mw is None or pd.isna(actual_mw)
                               else float(actual_mw),
            "deviation_mw": None,
            "deviation_percent": None,
            "penalty_amount": None,
            "payable_amount": None,
            "receivable_amount": None,
            "net_settlement": None,
            "ppa_amount": None,
            "status": "Pending",
        }

    scheduled_mw = float(scheduled_mw)
    actual_mw = float(actual_mw)

    deviation_mw = actual_mw - scheduled_mw
    penalty = penalty_rs(deviation_mw, capacity)

    return {
        "scheduled_mw": scheduled_mw,
        "actual_meter_mw": actual_mw,
        "deviation_mw": deviation_mw,
        "deviation_percent": deviation_mw / capacity * 100,
        "penalty_amount": penalty,
        "payable_amount": 0.0,
        "receivable_amount": 0.0,
        "net_settlement": -penalty,
        "ppa_amount": scheduled_mw * BLOCK_ENERGY_FACTOR * PPA_RATE,
        "status": "Calculated",
    }


# --------------------------------------------------

def day_settlement(blocks, capacity_mw=None, expected_blocks=None):
    """
    A whole day, priced block by block.

    `blocks` is either a DataFrame carrying `scheduled_mw` and
    `actual_mw` (a `block` column is used for the block number when
    present), or an iterable of (block_number, scheduled_mw, actual_mw).

    Day status, per the mentor's daily logic:

        no block priced          -> Pending
        some blocks priced       -> Partially Calculated
        all 96 priced, all zero  -> Zero Penalty
        all 96 priced, some cost -> Calculated

    "all 96" means every block of the day, which is why the count is a
    parameter: a report covering only the scheduled window is Partially
    Calculated by this rule even when nothing is missing inside it, and
    that is the honest label for it.
    """

    expected = BLOCKS_PER_DAY if expected_blocks is None else int(expected_blocks)

    rows = []

    if isinstance(blocks, pd.DataFrame):
        for position, (_, record) in enumerate(blocks.iterrows(), start=1):
            number = record["block"] if "block" in blocks.columns else position
            rows.append((number, record.get("scheduled_mw"), record.get("actual_mw")))
    else:
        rows = [tuple(row) for row in blocks]

    priced = []

    for number, scheduled, actual in rows:
        result = block_settlement(scheduled, actual, capacity_mw)
        result["block"] = int(number)
        priced.append(result)

    calculated = [b for b in priced if b["penalty_amount"] is not None]

    total = float(sum(b["penalty_amount"] for b in calculated))

    if not calculated:
        status = "Pending"
    elif len(calculated) < expected:
        status = "Partially Calculated"
    elif total == 0:
        status = "Zero Penalty"
    else:
        status = "Calculated"

    highest = max(
        calculated, key=lambda b: abs(b["penalty_amount"])
    ) if calculated else None

    return {
        "blocks": priced,
        "total_penalty": total,
        "calculated_blocks": len(calculated),
        "pending_blocks": len(priced) - len(calculated),
        "expected_blocks": expected,
        "status": status,
        "highest_penalty_block": highest,
        "settlement": SETTLEMENT,
    }
