"""
=========================================================
Solar Forecasting Project - NEW APPROACH
How much should the model actually move the number?
=========================================================
Sweeps fusion.blend_weight over saved schedules WITHOUT
spending any Gemini quota.

Every saved schedule already carries both numbers - the
model's forecast_mw and the physics anchor_mw - so the
published value for any blend weight is just

    w * forecast_mw + (1 - w) * anchor_mw

and the whole day can be restitched and repriced for each w.
That makes the weight a measurement rather than a guess, and
it costs nothing.

WHY THIS EXISTS

Seven-runs-a-day scoring over 12 days:

    model     Rs 27,663
    anchor    Rs 17,411     cheaper on 11 of 12 days

and the three-run sweep was closer (Rs 15,420 v 15,296).
The pattern is not subtle: the more often the model's number
is applied, the more it costs. That is what a signal looks
like when it is adding noise rather than judgement.

If the sweep bottoms out at w = 0, the model is not earning
its place at all and the honest answer is to publish the
anchor. If it bottoms out somewhere in between, that is the
weight to ship - the same way weather sits at 0.25 and
Chronos at 0.2 in the production blend.

Run:  python -m tests.sweep_blend_weight
=========================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from tests.score_day_schedules import (
    dsm_penalty, run_time_of, stitch_day
)


CAPACITY_MW = settings["plant"]["capacity_mw"]
TIMEZONE = settings["plant"]["timezone"]


def load_meter():

    meter = load_meter_history()

    if meter is None or meter.empty:
        raise SystemExit("No meter history to score against.")

    meter = meter.copy()

    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        meter["timestamp"] = meter["timestamp"].dt.tz_convert(TIMEZONE)

    if "is_real_measurement" in meter.columns:
        meter = meter[meter["is_real_measurement"].fillna(False)]

    meter["actual_mw"] = meter["active_power_kw"] / 1000.0

    return meter[["timestamp", "actual_mw"]].dropna()


def score_at(paths_by_day, meter, weight):
    """
    Total penalty and mean deviation across all days at one blend
    weight. The blend is applied per RUN before stitching, which is
    what a live run would publish - blending after stitching would mix
    blocks that different runs owned.
    """

    penalty = 0.0
    deviations = []
    within_band = []

    for day, paths in paths_by_day.items():

        model = stitch_day(paths, "forecast_mw")
        anchor = stitch_day(paths, "anchor_mw")

        if not model:
            continue

        frame = pd.DataFrame({
            "timestamp": list(model),
            "model_mw": list(model.values()),
        })

        frame["anchor_mw"] = frame["timestamp"].map(anchor)
        frame = frame.dropna(subset=["anchor_mw"])

        if frame.empty:
            continue

        frame["published_mw"] = np.clip(
            weight * frame["model_mw"] + (1 - weight) * frame["anchor_mw"],
            0.0, CAPACITY_MW,
        )

        merged = frame.merge(meter, on="timestamp", how="inner").dropna(
            subset=["actual_mw"]
        )

        if merged.empty:
            continue

        deviation = merged["actual_mw"] - merged["published_mw"]

        penalty += float(deviation.map(dsm_penalty).sum())
        deviations.append(float(deviation.abs().mean() / CAPACITY_MW * 100))
        within_band.append(
            float((deviation.abs() / CAPACITY_MW * 100 <= 10).mean() * 100)
        )

    return {
        "penalty_rs": penalty,
        "deviation_pct": float(np.mean(deviations)) if deviations else np.nan,
        "accuracy_pct": float(np.mean(within_band)) if within_band else np.nan,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Sweep the model/anchor blend weight over saved schedules"
    )
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )

    args = parser.parse_args()

    paths = sorted(Path(args.folder).glob("*_schedule.csv"))

    if not paths:
        raise SystemExit(f"No schedules in {args.folder}")

    meter = load_meter()

    by_day = {}

    for path in paths:

        run_time = run_time_of(path)

        if run_time is not None:
            by_day.setdefault(run_time.date(), []).append(path)

    print("=" * 78)
    print("BLEND WEIGHT SWEEP")
    print(f"  {len(paths)} saved schedules over {len(by_day)} days, "
          "no API calls")
    print("  w = 0.0 is the anchor alone; w = 1.0 is the model alone")
    print("=" * 78)
    print(f"{'w':>5} {'penalty Rs':>12} {'deviation %':>13} {'accuracy %':>12}")
    print("-" * 78)

    rows = []

    for weight in np.round(np.arange(0.0, 1.01, 0.1), 2):

        result = score_at(by_day, meter, float(weight))

        rows.append({"weight": float(weight), **result})

        print(f"{weight:5.1f} {result['penalty_rs']:12,.0f} "
              f"{result['deviation_pct']:13.2f} {result['accuracy_pct']:12.1f}")

    table = pd.DataFrame(rows)

    best = table.loc[table["penalty_rs"].idxmin()]

    print("-" * 78)
    print(f"cheapest at w = {best['weight']:.1f}: "
          f"Rs {best['penalty_rs']:,.0f}, "
          f"deviation {best['deviation_pct']:.2f}%")

    anchor_only = table.loc[table["weight"] == 0.0].iloc[0]
    model_only = table.loc[table["weight"] == 1.0].iloc[0]

    print(f"\nanchor alone (w=0.0) : Rs {anchor_only['penalty_rs']:,.0f}")
    print(f"model alone  (w=1.0) : Rs {model_only['penalty_rs']:,.0f}")

    if best["weight"] <= 0.0:
        print(
            "\nVERDICT: no blend beats the anchor by itself. On this data the "
            "model is\n         adding noise, not judgement - publishing the "
            "anchor is the honest\n         answer until that changes."
        )
    else:
        saved = anchor_only["penalty_rs"] - best["penalty_rs"]
        print(
            f"\nVERDICT: blending at w = {best['weight']:.1f} saves "
            f"Rs {saved:,.0f} against the anchor alone.\n"
            f"         Set fusion.blend_weight to {best['weight']:.1f} and "
            "re-score on new days before shipping."
        )

    output = Path("outputs/reports/blend_weight_sweep.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
