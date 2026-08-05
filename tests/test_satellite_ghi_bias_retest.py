"""
=========================================================
Solar Forecasting Project
Satellite GHI - Bias-Corrected Retest
=========================================================
Follow-up to test_satellite_ghi_experiment.py, which found
raw satellite GHI 66.4% off (moderate correlation 0.738,
noisy in both directions). Before dropping the candidate,
this applies the SAME bias-correction trick already used on
Open-Meteo (modules/weather/open_meteo.py bias_factor):
correction factor = median(real/sat) over the last 5 days,
applied WALK-FORWARD (never using today's own answer to
correct today - that would be cheating).

Run:  python -m tests.test_satellite_ghi_bias_retest
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 5


def main():

    path = Path("outputs/reports/satellite_ghi_experiment.csv")

    if not path.exists():
        print("Run tests.test_satellite_ghi_experiment first.")
        return

    d = pd.read_csv(path, parse_dates=["timestamp"])
    d["date"] = d["timestamp"].dt.date

    days = sorted(d["date"].unique())

    raw_err, corrected_err, factors = [], [], []

    for i, day in enumerate(days):

        window_days = days[max(0, i - WINDOW_DAYS):i]  # PRIOR days only

        if not window_days:
            continue  # no history yet - can't correct the first days

        window = d[d["date"].isin(window_days)]
        window = window[window["sat_ghi"] > 20]

        if window.empty:
            continue

        factor = float((window["ghi_w_m2"] / window["sat_ghi"]).median())
        factor = np.clip(factor, 0.4, 2.5)  # sane bounds, same spirit as open_meteo.py

        today = d[d["date"] == day]

        raw_err.extend((today["sat_ghi"] - today["ghi_w_m2"]).abs().tolist())
        corrected_err.extend(
            (today["sat_ghi"] * factor - today["ghi_w_m2"]).abs().tolist()
        )
        factors.append(factor)

    raw_err = np.array(raw_err)
    corrected_err = np.array(corrected_err)

    print("=" * 70)
    print("SATELLITE GHI - WALK-FORWARD BIAS CORRECTION RETEST")
    print("=" * 70)
    print(f"days with a prior 5-day window : {len(factors)} / {len(days)}")
    print(f"correction factor range        : {min(factors):.2f} .. {max(factors):.2f}")
    print(f"median correction factor       : {np.median(factors):.2f}")
    print()
    print(f"points compared                : {len(raw_err)}")
    print(f"RAW mean absolute error        : {raw_err.mean():.1f} W/m2")
    print(f"CORRECTED mean absolute error  : {corrected_err.mean():.1f} W/m2")
    print(f"improvement                    : {raw_err.mean() - corrected_err.mean():+.1f} W/m2 "
          f"({(1 - corrected_err.mean()/raw_err.mean())*100:+.1f}%)")

    better = (corrected_err < raw_err).sum()
    print(f"points improved                : {better}/{len(raw_err)} "
          f"({better/len(raw_err)*100:.0f}%)")

    print()
    if corrected_err.mean() < raw_err.mean() * 0.85:
        print("VERDICT: bias correction meaningfully helps. Worth a real")
        print("         forecast-blend test (plug into blend_signals, score")
        print("         leave-one-day-out) before any pipeline change.")
    else:
        print("VERDICT: bias correction does not fix the core problem - the")
        print("         error here is noisy/random (monsoon cloud speed), not")
        print("         a steady over/under bias. Same fix that worked for")
        print("         Open-Meteo does not transfer. DROP this candidate,")
        print("         try CAMS or INSAT next.")


if __name__ == "__main__":
    main()
