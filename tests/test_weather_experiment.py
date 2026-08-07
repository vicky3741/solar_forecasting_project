"""
=========================================================
Solar Forecasting Project
Open-Meteo Weather Signal Experiment
=========================================================
Measures - honestly - whether folding the forward-looking
Open-Meteo forecast into the blend actually improves
accuracy.

Method:
  1. Run the point-in-time backtest once. compute_signals now
     also fetches the Open-Meteo historical forecast (the
     forecast ISSUED at the time, not the recorded truth - no
     lookahead) and stores a weather_kt per run.
  2. Cheaply re-blend the cached signals at several
     weather_weight values (0 = weather off) and report the
     average deviation for each.
  3. Leave-one-day-out: for the best weight, check it still
     helps on days it was not chosen on - the same
     out-of-sample discipline used for the residual model. A
     weight that only helps in-sample is not shipped.
=========================================================
"""

import numpy as np
import pandas as pd

from config.config import settings
from modules.evaluation.backtester import Backtester
from modules.evaluation import metrics
from utils.file_manager import processed_data_path


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
WEATHER_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def avg_deviation_at_weight(backtester, cache, weather_weight, days=None):

    devs = []
    for (date, _), cached in cache.items():
        if days is not None and date not in days:
            continue
        forecast = backtester.predictor.blend_signals(
            cached["signals"],
            weather_weight=weather_weight,
            vision_adjustment=cached["vision_adjustment"],
        )
        merged = forecast.merge(cached["actual"], on="timestamp", how="inner")
        if merged.empty:
            continue
        devs.append(metrics.average_percentage_deviation(
            merged["final_forecast_kw"], merged["active_power_kw"], CAPACITY_KW
        ))
    return float(np.mean(devs)) if devs else None


def main():

    df = pd.read_csv(processed_data_path(), parse_dates=["timestamp"])

    backtester = Backtester()

    print("Running backtest (fetches Open-Meteo forecast per day, cached)...")
    _, cache, _ = backtester.run(df)

    print("\n=== Weather weight sweep (all days) ===")
    print(f"{'weather_weight':>15} {'avg deviation %':>17}")
    sweep = {}
    for w in WEATHER_WEIGHTS:
        dev = avg_deviation_at_weight(backtester, cache, w)
        sweep[w] = dev
        tag = "  <- weather OFF (baseline)" if w == 0 else ""
        print(f"{w:>15.2f} {dev:>17.3f}{tag}")

    baseline = sweep[0.0]
    best_w = min(sweep, key=lambda k: sweep[k])
    best_dev = sweep[best_w]

    print(f"\nBaseline (no weather) : {baseline:.3f} %")
    print(f"Best weight {best_w:.2f}       : {best_dev:.3f} %  ({baseline - best_dev:+.3f} pts)")

    # --- Leave-one-day-out: does best_w generalize? ---
    days = sorted({d for (d, _) in cache.keys()})
    helped = 0
    total = 0
    for held in days:
        others = [d for d in days if d != held]
        # pick the best weight on the OTHER days, apply to the held day
        best_other = min(
            WEATHER_WEIGHTS,
            key=lambda w: avg_deviation_at_weight(backtester, cache, w, days=set(others))
        )
        dev_off = avg_deviation_at_weight(backtester, cache, 0.0, days={held})
        dev_on = avg_deviation_at_weight(backtester, cache, best_other, days={held})
        if dev_off is None or dev_on is None:
            continue
        total += 1
        if dev_on < dev_off:
            helped += 1

    print("\n=== Leave-one-day-out (out-of-sample honesty) ===")
    print(f"Weather (best weight chosen on other days) helped on {helped} of {total} days")

    print("\n=== VERDICT ===")
    if best_w > 0 and best_dev < baseline and helped > total / 2:
        print(f"Weather HELPS. Recommend weather_weight = {best_w:.2f} "
              f"({baseline - best_dev:+.3f} pts, generalizes {helped}/{total}).")
    else:
        print("Weather does NOT reliably help on unseen days - keep weather_weight = 0.")


if __name__ == "__main__":
    main()
