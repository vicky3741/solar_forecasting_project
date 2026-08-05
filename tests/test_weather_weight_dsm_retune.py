"""
=========================================================
Solar Forecasting Project
Weather-Weight Re-tune - DSM Rs PENALTY OBJECTIVE
=========================================================
Every prior tune (this one's predecessor, the adaptive-
weight experiment) optimized average % deviation. DSM does
not charge % deviation - it charges Rs, in RISING SLABS:
0-10% free, 10-15% = 0.5 Rs/kWh, 15-20% = 0.75, >20% = 1.0.
A model with slightly worse average accuracy but no big
slab-4 spikes can cost LESS real money than one with better
average accuracy but occasional large misses.

Every report built this project has shown worst-single-
block penalty, not average deviation, driving the day's
total cost. This asks directly: does the weight that
minimizes Rs differ from the weight that minimizes %
deviation (0.20-0.25, found by the two prior scripts)?

Same cache, same grid, same leave-one-day-out discipline -
only the scoring function changes. Vision OFF (inherited
from the cached runs); re-check with vision ON before
shipping anything this finds.

Run:  python -m tests.test_weather_weight_dsm_retune
=========================================================
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from tests.test_adaptive_weather_weight_experiment import CAPACITY_KW, weights_constant

CACHE = Path("outputs/weather_weight_runs.pkl")

W_GRID = np.round(np.arange(0.0, 0.951, 0.025), 4)
SHIPPED = 0.65
CURRENT = 0.25          # what's live in production right now

CAPACITY_MW = settings["plant"]["capacity_mw"]
BLOCK_ENERGY_FACTOR = 250      # 0.25 h x 1000 kW/MW

# Same DSM slabs used in every penalty report this project has built.
EDGE1 = CAPACITY_MW * 0.10
EDGE2 = CAPACITY_MW * 0.15
EDGE3 = CAPACITY_MW * 0.20
RATE2, RATE3, RATE4 = 0.5, 0.75, 1.0


def dsm_penalty_rs(deviation_mw):
    """Vectorized DSM slab penalty for an array of |deviation| in MW."""

    dev = np.abs(deviation_mw)

    slab2 = np.clip(np.minimum(dev, EDGE2) - EDGE1, 0, None) * RATE2
    slab3 = np.clip(np.minimum(dev, EDGE3) - EDGE2, 0, None) * RATE3
    slab4 = np.clip(dev - EDGE3, 0, None) * RATE4

    return BLOCK_ENERGY_FACTOR * (slab2 + slab3 + slab4)


def total_penalty_for(day_data, w, performance_ratio):
    """Total Rs penalty for one day at one weather_weight - the real cost."""

    weights = weights_constant(day_data, w)

    blended_kw = weights * day_data["f1"] + (1.0 - weights) * day_data["f0"]
    forecast_kw = np.clip(blended_kw * performance_ratio, 0, CAPACITY_KW)

    deviation_mw = (day_data["actual"] - forecast_kw) / 1000.0

    return float(dsm_penalty_rs(deviation_mw).sum())


def main():

    if not CACHE.exists():
        print(f"{CACHE} not found - run tests.test_weather_weight_retune once "
              "first to build the cache (same Chronos runs, reused here).")
        return

    collapsed = pickle.loads(CACHE.read_bytes())
    scored = sorted(collapsed)
    pr = settings["clearsky"]["performance_ratio"]

    print("=" * 72)
    print(f"WEATHER-WEIGHT RE-TUNE - DSM Rs PENALTY OBJECTIVE   "
          f"({len(scored)} days, PR {pr}, vision OFF)")
    print("=" * 72)

    table = {(w, d): total_penalty_for(collapsed[d], w, pr)
             for w in W_GRID for d in scored}

    curve = {w: float(np.mean([table[(w, d)] for d in scored])) for w in W_GRID}
    best_w = min(curve, key=curve.get)

    print()
    print("  weight   mean daily penalty (Rs)")
    for w in W_GRID:
        mark = ""
        if w == best_w:
            mark = "  <- best for Rs"
        elif abs(w - SHIPPED) < 1e-9:
            mark = "  <- old shipped"
        elif abs(w - CURRENT) < 1e-9:
            mark = "  <- CURRENT (live now)"
        print(f"   {w:0.3f}      {curve[w]:8.1f}{mark}")

    # ---------------- leave-one-day-out ----------------
    loo_current, loo_shipped, loo_best, picks = [], [], [], []

    for held_out in scored:
        train = [d for d in scored if d != held_out]
        pick = min(W_GRID, key=lambda w: np.mean([table[(w, d)] for d in train]))
        picks.append(pick)
        loo_best.append(table[(pick, held_out)])
        loo_current.append(table[(CURRENT, held_out)])
        loo_shipped.append(table[(SHIPPED, held_out)])

    m_best = float(np.mean(loo_best))
    m_curr = float(np.mean(loo_current))
    m_ship = float(np.mean(loo_shipped))

    print()
    print("-" * 72)
    print("LEAVE-ONE-DAY-OUT (held-out days only) - Rs/day")
    print("-" * 72)
    print(f"  old shipped 0.65          : Rs {m_ship:7.1f} / day")
    print(f"  CURRENT (live) 0.25       : Rs {m_curr:7.1f} / day")
    print(f"  best for Rs (median {np.median(picks):.3f})  : Rs {m_best:7.1f} / day   "
          f"({m_curr - m_best:+.1f} Rs/day vs current)")

    better_than_current = sum(1 for a, b in zip(loo_current, loo_best) if b < a)
    print()
    print(f"  days Rs-tuned beat CURRENT (0.25) : {better_than_current}/{len(scored)}")

    print()
    if m_curr - m_best > 20 and better_than_current > len(scored) / 2:
        print(f"VERDICT: even the CURRENT (0.25, already better than old 0.65) is not")
        print(f"         the Rs-optimal point. Rs-optimal is near {np.median(picks):.2f}.")
        print("         Re-check with vision ON, then consider moving it.")
    else:
        print("VERDICT: the weight that minimizes % deviation (current 0.25) is ALSO")
        print("         close to the weight that minimizes Rs penalty. No separate")
        print("         Rs-driven change needed here - the earlier retune already")
        print("         captured most of the available money.")

    out = Path("outputs/reports/weather_weight_dsm_retune.csv")
    pd.DataFrame(
        [{"weather_weight": w, "mean_daily_penalty_rs": round(curve[w], 2)}
         for w in W_GRID]
    ).to_csv(out, index=False)
    print(f"\nSaved curve: {out}")


if __name__ == "__main__":
    main()
