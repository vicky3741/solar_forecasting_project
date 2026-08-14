"""
=========================================================
Solar Forecasting Project
DSM Penalty - checked against the mentor's own arithmetic
=========================================================
The mentor sent the SIRMOUR penalty logic on 2026-08-14
with a worked example, after "your Enercast penalty not
matching with actual penalty". This script is how we show
the code computes HIS number, not one of ours that happens
to look similar.

Three things are checked:

1. His worked example, block by block, to the paisa.

2. His SUMMATION form against our MARGINAL-SLAB form, over
   a sweep of deviations from 0 to well past 20% of
   capacity. They are the same formula written two ways
   (see modules/evaluation/dsm_penalty.py); this proves it
   numerically rather than only on paper.

3. The rules that are easy to get wrong and expensive to
   get wrong: symmetry (over- and under-generation cost the
   same), missing blocks priced as Pending and not as zero,
   and the daily status labels.

Run:  python -m tests.test_dsm_penalty
=========================================================
"""

import numpy as np

from config.config import settings
from modules.evaluation import dsm_penalty


CAPACITY_MW = settings["plant"]["capacity_mw"]

TOLERANCE = 1e-9

failures = []


def check(label, got, expected, tolerance=TOLERANCE):

    ok = abs(got - expected) <= tolerance

    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} "
          f"got {got:14.6f}   expected {expected:14.6f}")

    if not ok:
        failures.append(label)


def mentor_summation_form(scheduled_mw, actual_mw, capacity_mw):
    """
    The mentor's algorithm, transcribed from his message with no
    simplification: apportion the block's deviation ENERGY across the
    bands by the share of the deviation PERCENTAGE that falls in each.
    """

    deviation_mw = actual_mw - scheduled_mw
    absolute_percent = abs(deviation_mw / capacity_mw * 100)
    deviation_energy_kwh = abs(deviation_mw) * 0.25 * 1000

    if absolute_percent == 0:
        return 0.0

    penalty = 0.0

    for lower, upper, rate in [(0, 10, 0.0), (10, 15, 0.5),
                               (15, 20, 0.75), (20, np.inf, 1.0)]:

        span = min(absolute_percent, upper) - lower

        if span > 0:
            penalty += deviation_energy_kwh * (span / absolute_percent) * rate

    return penalty


def main():

    print("=" * 92)
    print("DSM PENALTY - mentor's SIRMOUR logic (2026-08-14)")
    print("=" * 92)
    print(f"capacity {CAPACITY_MW} MW   bands {dsm_penalty.BANDS}   "
          f"settlement '{dsm_penalty.SETTLEMENT}'")

    # ---------------- 1. the mentor's worked example ----------------
    #
    # scheduled 4.0, actual 3.0 -> deviation -1.0 MW = -19.6078% of 5.1 MW.
    #
    #   0-10%   250 * (10.0000 / 19.6078) * 0    = 0
    #   10-15%  250 * ( 5.0000 / 19.6078) * 0.5  = 31.8750
    #   15-20%  250 * ( 4.6078 / 19.6078) * 0.75 = 44.0625
    #                                              --------
    #                                              75.9375
    #
    # His message totals this to 75.9566, from rounding 4.6078/19.6078
    # in the last line; carried exactly it is 44.0625, and the block is
    # Rs 75.9375. Everything else in his example reproduces exactly.
    print("\n1. The mentor's worked example (scheduled 4.0 MW, actual 3.0 MW)")

    result = dsm_penalty.block_settlement(4.0, 3.0)

    check("deviation_mw", result["deviation_mw"], -1.0)
    check("deviation_percent", result["deviation_percent"], -1.0 / 5.1 * 100, 1e-6)
    check("penalty_amount", result["penalty_amount"], 75.9375, 1e-6)
    check("net_settlement", result["net_settlement"], -75.9375, 1e-6)
    check("payable_amount", result["payable_amount"], 0.0)
    check("receivable_amount", result["receivable_amount"], 0.0)
    check("ppa_amount", result["ppa_amount"], 4.0 * 250 * 2.94)

    check("his own summation form agrees",
          mentor_summation_form(4.0, 3.0, CAPACITY_MW), 75.9375, 1e-6)

    # ---------------- 2. summation form == marginal-slab form ----------------
    print("\n2. His summation form vs our marginal-slab form, across the range")

    worst = 0.0
    worst_at = 0.0

    for deviation in np.arange(-1.6, 1.6001, 0.01):

        ours = dsm_penalty.penalty_rs(deviation)
        theirs = mentor_summation_form(0.0, deviation, CAPACITY_MW)

        gap = abs(ours - theirs)

        if gap > worst:
            worst, worst_at = gap, deviation

    print(f"  deviations swept    : -1.60 to +1.60 MW in 0.01 steps "
          f"(0 to {1.6 / CAPACITY_MW * 100:.1f}% of capacity)")
    check(f"largest disagreement (at {worst_at:+.2f} MW)", worst, 0.0, 1e-9)

    # ---------------- 3. the rules worth protecting ----------------
    print("\n3. Rules the logic must preserve")

    over = dsm_penalty.penalty_rs(+1.0)
    under = dsm_penalty.penalty_rs(-1.0)
    check("over-generation costs the same as under", over, under)

    # 10% of 5.1 MW = 0.51 MW: the free band, and the first paisa above it.
    check("0.51 MW deviation (exactly 10%) is free", dsm_penalty.penalty_rs(0.51), 0.0)
    check("0.61 MW deviation costs (0.61-0.51)*250*0.5",
          dsm_penalty.penalty_rs(0.61), 0.10 * 250 * 0.5, 1e-9)
    check("zero deviation is free", dsm_penalty.penalty_rs(0.0), 0.0)

    # A block above 20% pays all three paid bands: 0.255 MW at 0.50,
    # 0.255 at 0.75, the rest at 1.00.
    deep = 1.5
    expected_deep = 250 * (
        (CAPACITY_MW * 0.05) * 0.5
        + (CAPACITY_MW * 0.05) * 0.75
        + (deep - CAPACITY_MW * 0.20) * 1.0
    )
    check("1.50 MW deviation (>20%) pays all three bands",
          dsm_penalty.penalty_rs(deep), expected_deep, 1e-9)

    missing = dsm_penalty.block_settlement(2.0, None)
    ok = missing["penalty_amount"] is None and missing["status"] == "Pending"
    print(f"  {'PASS' if ok else 'FAIL'}  "
          f"{'missing meter -> Pending, not a penalty of 0':52s} "
          f"got {missing['penalty_amount']}, {missing['status']}")
    if not ok:
        failures.append("missing meter -> Pending")

    # ---------------- 4. daily status labels ----------------
    print("\n4. Daily aggregation")

    full_day_free = [(n, 1.0, 1.0) for n in range(1, 97)]
    day = dsm_penalty.day_settlement(full_day_free)
    check("96 blocks, no deviation -> total", day["total_penalty"], 0.0)
    print(f"  status: {day['status']} (expected Zero Penalty)")
    if day["status"] != "Zero Penalty":
        failures.append("Zero Penalty status")

    partial = [(n, 1.0, 1.0) for n in range(1, 50)]
    partial += [(n, 1.0, None) for n in range(50, 97)]
    day = dsm_penalty.day_settlement(partial)
    print(f"  status: {day['status']} (expected Partially Calculated), "
          f"{day['calculated_blocks']} priced / {day['pending_blocks']} pending")
    if day["status"] != "Partially Calculated":
        failures.append("Partially Calculated status")

    nothing = dsm_penalty.day_settlement([(n, None, None) for n in range(1, 97)])
    print(f"  status: {nothing['status']} (expected Pending)")
    if nothing["status"] != "Pending":
        failures.append("Pending status")

    costly = [(n, 1.0, 1.0) for n in range(1, 96)] + [(96, 4.0, 3.0)]
    day = dsm_penalty.day_settlement(costly)
    check("one costly block in a clean day", day["total_penalty"], 75.9375, 1e-6)
    highest = day["highest_penalty_block"]
    print(f"  status: {day['status']} (expected Calculated); "
          f"highest penalty block = {highest['block']} "
          f"at Rs {highest['penalty_amount']:.2f}")
    if day["status"] != "Calculated" or highest["block"] != 96:
        failures.append("Calculated status / highest block")

    print("\n" + "=" * 92)

    if failures:
        print(f"FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    else:
        print("All checks passed - the code prices a block exactly as the mentor's "
              "logic does.")

    print("=" * 92)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
