"""
=========================================================
Solar Forecasting Project
What Actually Drives DSM Penalty - Horizon vs Volatility
=========================================================
Two ideas on the table before any pipeline change:

  A. Refresh more often than every ~90 min (shorter horizon
     per block). Real cost: more EC2 compute, more Gemini
     calls against a 20/day shared quota. Before spending
     that, test the PREMISE: does penalty actually rise with
     how stale a block's forecast is? If not, faster refresh
     buys nothing.

  B. cloud_field_structure (patchy/broken/solid) as a spike
     predictor. Only 1 day of real data exists so far (added
     2026-08-03) - can't test it directly yet. Proxy instead:
     recent kt VOLATILITY (already computed in the adaptive-
     weight experiment) stands in for "is the sky patchy right
     now". If volatility already predicts penalty, that is
     real evidence the full feature is worth wiring in once
     enough data exists - not a guess.

At the CURRENT live weight (0.25), fixed - not sweeping
weights again, that question is closed. Vision OFF, same
reason as every other script this session: reproducibility
over the free-tier quota.

Run:  python -m tests.test_penalty_drivers_analysis
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from tests.test_adaptive_weather_weight_experiment import collapse_days, load_data
from tests.test_weather_weight_dsm_retune import CAPACITY_MW, dsm_penalty_rs

CURRENT_WEIGHT = 0.25


def main():

    data = load_data()

    actual = data[data["is_real_measurement"].fillna(False)]
    actual_series = actual.set_index("timestamp")["active_power_kw"]

    days = sorted(data["timestamp"].dt.date.unique())

    print("=" * 72)
    print("WHAT DRIVES DSM PENALTY - horizon vs volatility")
    print("=" * 72)
    print("Running Chronos once per scheduling time (same as prior scripts)...")

    collapsed = collapse_days(data, days, actual_series)

    pr = settings["clearsky"]["performance_ratio"]

    rows = []
    for day, d in collapsed.items():

        blended = CURRENT_WEIGHT * d["f1"] + (1 - CURRENT_WEIGHT) * d["f0"]
        forecast_kw = np.clip(blended * pr, 0, 5100)
        deviation_mw = (d["actual"] - forecast_kw) / 1000.0
        penalty = dsm_penalty_rs(deviation_mw)

        for h, v, p, dev in zip(d["horizon"], d["volatility"], penalty, deviation_mw):
            rows.append({"day": day, "horizon_min": h, "volatility": v,
                         "penalty_rs": p, "abs_dev_mw": abs(dev)})

    df = pd.DataFrame(rows)

    print(f"\nblocks analyzed: {len(df)}  across {df['day'].nunique()} days")

    # ---------------- A: horizon ----------------
    print()
    print("-" * 72)
    print("A. PENALTY vs HOW STALE THE FORECAST WAS (minutes since last run)")
    print("-" * 72)
    bins = [0, 30, 60, 90, 120, 999]
    labels = ["0-30 min", "30-60 min", "60-90 min", "90-120 min", "120+ min"]
    df["horizon_bucket"] = pd.cut(df["horizon_min"], bins=bins, labels=labels)

    horizon_table = df.groupby("horizon_bucket", observed=True).agg(
        blocks=("penalty_rs", "size"),
        mean_penalty_rs=("penalty_rs", "mean"),
        pct_with_penalty=("penalty_rs", lambda s: (s > 0).mean() * 100),
    ).round(2)
    print(horizon_table.to_string())

    corr_h = df["horizon_min"].corr(df["penalty_rs"])
    print(f"\ncorrelation (horizon vs penalty): {corr_h:.3f}")

    # ---------------- B: volatility ----------------
    print()
    print("-" * 72)
    print("B. PENALTY vs RECENT SKY VOLATILITY (proxy for patchy/broken cloud)")
    print("-" * 72)
    df["vol_quartile"] = pd.qcut(df["volatility"], 4, labels=["Q1 calmest", "Q2", "Q3", "Q4 jumpiest"])

    vol_table = df.groupby("vol_quartile", observed=True).agg(
        blocks=("penalty_rs", "size"),
        mean_penalty_rs=("penalty_rs", "mean"),
        pct_with_penalty=("penalty_rs", lambda s: (s > 0).mean() * 100),
    ).round(2)
    print(vol_table.to_string())

    corr_v = df["volatility"].corr(df["penalty_rs"])
    print(f"\ncorrelation (volatility vs penalty): {corr_v:.3f}")

    # ---------------- verdicts ----------------
    print()
    print("=" * 72)
    print("VERDICTS")
    print("=" * 72)

    h_ratio = horizon_table["mean_penalty_rs"].iloc[-1] / max(horizon_table["mean_penalty_rs"].iloc[0], 0.01)
    if corr_h > 0.15 or h_ratio > 2:
        print(f"A. Refresh-frequency idea: SUPPORTED. Stale blocks (120+ min old)")
        print(f"   cost {h_ratio:.1f}x the penalty of fresh ones (0-30 min).")
        print("   Worth costing out a shorter refresh interval for real.")
    else:
        print("A. Refresh-frequency idea: NOT supported by this data. Penalty does")
        print("   not rise meaningfully with staleness - don't spend EC2/quota on")
        print("   more frequent runs expecting this to fix it.")

    v_ratio = vol_table["mean_penalty_rs"].iloc[-1] / max(vol_table["mean_penalty_rs"].iloc[0], 0.01)
    if corr_v > 0.15 or v_ratio > 2:
        print(f"\nB. Cloud-structure idea: SUPPORTED. Jumpiest-sky blocks (Q4) cost")
        print(f"   {v_ratio:.1f}x the penalty of calmest blocks (Q1). Real evidence -")
        print("   worth wiring in cloud_field_structure once ~1 week of data exists.")
    else:
        print("\nB. Cloud-structure idea: NOT supported by this volatility proxy.")
        print("   Real cloud_field_structure field may still differ (it reads the")
        print("   actual sky image, not just recent kt jumpiness) - re-test once")
        print("   real data accumulates rather than dropping the idea now.")

    out = Path("outputs/reports/penalty_drivers_analysis.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved per-block data: {out}")


if __name__ == "__main__":
    main()
