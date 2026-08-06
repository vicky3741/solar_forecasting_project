"""
=========================================================
Solar Forecasting Project
Weather-Weight Re-tune
=========================================================
`weather_weight` ships at 0.65. That value was tuned in
July on roughly three weeks of data, BEFORE the walk-forward
weather bias correction existed. That correction now scales
the weather signal down on its own (0.67-0.82 on recent
days), so part of what 0.65 was compensating for may already
be handled - meaning the blend could now be double-counting
the weather forecast.

The adaptive-weight experiment (2026-08-04) hinted at this:
its leave-one-day-out baseline kept choosing ~0.20, not
0.65. That was a side observation there, not the question
being asked. This script asks it directly.

Prints the FULL deviation-vs-weight curve, so the shape is
visible rather than just an argmin - a flat curve means the
exact value barely matters and 0.65 is fine to leave alone;
a steep one means it is genuinely costing accuracy.

Vision is OFF, as in the adaptive experiment: 238 Gemini
calls is neither free (20/day shared quota) nor
reproducible. Any recommended change has to be re-checked
with vision ON before it ships, because vision shifts the
persistence side of the blend and could move the optimum.

Run:  python -m tests.test_weather_weight_retune
=========================================================
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.evaluation import metrics
from tests.test_adaptive_weather_weight_experiment import (
    CAPACITY_KW, collapse_days, load_data, weights_constant,
)

CACHE = Path("outputs/weather_weight_runs.pkl")

# Finer than the adaptive experiment's grid - the question here is the
# exact value, so 0.05 steps would be too coarse to trust an argmin.
W_GRID = np.round(np.arange(0.0, 0.951, 0.025), 4)

SHIPPED = 0.65
CANDIDATE = 0.50          # the value under discussion


def deviation_for(day_data, w, performance_ratio):

    weights = weights_constant(day_data, w)

    blended = weights * day_data["f1"] + (1.0 - weights) * day_data["f0"]
    forecast = np.clip(blended * performance_ratio, 0, CAPACITY_KW)

    return metrics.average_percentage_deviation(
        forecast, day_data["actual"], CAPACITY_KW
    )


def main():

    data = load_data()

    actual = data[data["is_real_measurement"].fillna(False)]
    actual = actual.set_index("timestamp")["active_power_kw"]

    days = sorted(data["timestamp"].dt.date.unique())

    if CACHE.exists():
        print(f"Reusing cached runs: {CACHE}")
        collapsed = pickle.loads(CACHE.read_bytes())
    else:
        print("Running Chronos once per scheduling time...")
        collapsed = collapse_days(data, days, actual)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(collapsed))
        print(f"Cached: {CACHE}")

    scored = sorted(collapsed)
    pr = settings["clearsky"]["performance_ratio"]

    print()
    print("=" * 72)
    print(f"WEATHER-WEIGHT RE-TUNE   ({len(scored)} days, PR {pr}, vision OFF)")
    print("=" * 72)

    table = {(w, d): deviation_for(collapsed[d], w, pr)
             for w in W_GRID for d in scored}

    curve = {w: float(np.mean([table[(w, d)] for d in scored])) for w in W_GRID}

    best_w = min(curve, key=curve.get)

    print()
    print("  weight   mean deviation")
    for w in W_GRID:
        mark = ""
        if w == best_w:
            mark = "  <- best"
        elif abs(w - SHIPPED) < 1e-9:
            mark = "  <- shipped"
        elif abs(w - CANDIDATE) < 1e-9:
            mark = "  <- proposed"
        print(f"   {w:0.3f}      {curve[w]:6.3f}%{mark}")

    # ---------------- leave-one-day-out ----------------
    # Each fold picks its weight from the OTHER days, so the reported
    # number is what an unseen day would really have got.
    loo_best, loo_shipped, loo_candidate, picks = [], [], [], []

    for held_out in scored:
        train = [d for d in scored if d != held_out]
        pick = min(W_GRID, key=lambda w: np.mean([table[(w, d)] for d in train]))
        picks.append(pick)
        loo_best.append(table[(pick, held_out)])
        loo_shipped.append(table[(SHIPPED, held_out)])
        loo_candidate.append(table[(CANDIDATE, held_out)])

    m_best = float(np.mean(loo_best))
    m_ship = float(np.mean(loo_shipped))
    m_cand = float(np.mean(loo_candidate))

    print()
    print("-" * 72)
    print("LEAVE-ONE-DAY-OUT (held-out days only)")
    print("-" * 72)
    print(f"  shipped  weight {SHIPPED:.2f}          : {m_ship:.3f}%")
    print(f"  proposed weight {CANDIDATE:.2f}          : {m_cand:.3f}%   "
          f"({m_ship - m_cand:+.3f} pts vs shipped)")
    print(f"  tuned per fold (median {np.median(picks):.3f}) : {m_best:.3f}%   "
          f"({m_ship - m_best:+.3f} pts vs shipped)")

    better_cand = sum(1 for a, b in zip(loo_shipped, loo_candidate) if b < a)
    better_best = sum(1 for a, b in zip(loo_shipped, loo_best) if b < a)
    print()
    print(f"  days {CANDIDATE:.2f} beat {SHIPPED:.2f} : {better_cand}/{len(scored)}")
    print(f"  days tuned beat {SHIPPED:.2f}  : {better_best}/{len(scored)}")

    # How flat is the curve? If moving the weight a long way barely moves
    # the number, the exact value is not worth arguing about.
    span = max(curve.values()) - min(curve.values())
    print()
    print(f"  curve span across the whole grid : {span:.3f} pts")

    print()
    if m_ship - m_best > 0.10 and better_best > len(scored) / 2:
        print(f"VERDICT: {SHIPPED} is costing real accuracy. Best is around "
              f"{np.median(picks):.2f}.")
        print("         Re-check with vision ON, then change it.")
    else:
        print(f"VERDICT: no meaningful gain from moving {SHIPPED}. Leave it.")

    out = Path("outputs/reports/weather_weight_retune.csv")
    pd.DataFrame(
        [{"weather_weight": w, "mean_deviation_pct": round(curve[w], 4)}
         for w in W_GRID]
    ).to_csv(out, index=False)
    print(f"\nSaved curve: {out}")


if __name__ == "__main__":
    main()
