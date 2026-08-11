"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Publish for the BAND, not for the average
=========================================================
tests/diagnose_error.py found the contradiction this file
exists to resolve: over the same six days the model's raw call
has the LOWEST block error (7.78% against the anchor's 8.89%)
and the HIGHEST rupee cost (Rs 7,351 against Rs 4,882).

Both are true because the DSM penalty is BANDED. Deviation
inside the free band costs nothing at all; outside it, it is
charged by slab. So a forecast can cut mean error and still
cost more, by trading many cheap small misses for a few
expensive large ones. We optimise accuracy; we are graded on
staying inside a line, and nothing in the pipeline knows the
line is there.

WHAT IS TUNED
-------------
One published value per block:

    published = level * [ anchor + weight * (model - anchor) ] + shift

  weight  how much of the model's departure from the anchor
          survives. This is fusion.blend_weight.
  level   one multiplicative scale for the whole run - the
          lever fusion.level_calibration already pulls, but
          tuned here against RUPEES rather than against error.
          diagnose_error.py measured 41% of our error as pure
          level, so this is where the cheap money is.
  shift   a flat MW offset. Small, and it is the only genuinely
          band-aware term: when the residual distribution sits
          off-centre, moving the whole schedule a little
          recentres it inside the free band and pays for
          itself even though it makes the MEAN error worse.

EVERYTHING IS SCORED IN RUPEES, through the same slab function
tests/score_day_schedules.py uses, on schedules stitched under
the real freeze horizon. No API calls: this replays saved runs,
so it can be swept as often as wanted for free.

WALK-FORWARD, NOT FITTED
------------------------
Thirteen days is not enough to trust a three-parameter fit. So
each day is scored with parameters chosen on the days BEFORE
it only. The headline number is therefore what this would have
earned in production, not what it can achieve in hindsight -
and the in-hindsight number is printed beside it so the gap
between the two is visible rather than hidden.

Run:  python -m tests.tune_band_aware
      python -m tests.tune_band_aware --free-band 5
=========================================================
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import tests.score_day_schedules as scoring
import tests.sweep_blend_weight as sweep
from config.config import settings


CAPACITY_MW = settings["plant"]["capacity_mw"]


def stitch(paths_by_day, meter, weight, level, shift):
    """
    One day's published schedule under the freeze horizon, with the
    three knobs applied, priced in rupees.

    Reuses sweep_blend_weight's stitcher so the freeze horizon and the
    run-ordering are treated exactly as the scoring everyone else quotes
    already treats them.
    """

    penalty = 0.0
    inside = []
    errors = []

    free_band = scoring.SLABS[0][1]

    for day, paths in sorted(paths_by_day.items()):

        # stitch_day returns {timestamp: value} for one column, already
        # resolved under the freeze horizon. Two passes give the two
        # ends of the blend.
        #
        # NOTE ON `weight`, which is easy to misread: forecast_mw is
        # what the pipeline ALREADY published, blend, calibration,
        # validator and freeze included. So weight = 1.0 is "as
        # shipped", not "the raw model", and weight = 0.0 is the bare
        # anchor. This matches tests/sweep_blend_weight.py so the two
        # sets of numbers can be compared.
        shipped = scoring.stitch_day(paths, "forecast_mw")
        anchor = scoring.stitch_day(paths, "anchor_mw")

        if not shipped:
            continue

        frame = pd.DataFrame({
            "timestamp": list(shipped),
            "shipped_mw": list(shipped.values()),
        })

        frame["anchor_mw"] = frame["timestamp"].map(anchor)
        frame = frame.dropna(subset=["anchor_mw"])

        if frame.empty:
            continue

        merged = frame.merge(meter, on="timestamp", how="inner").dropna(
            subset=["actual_mw"]
        )

        if merged.empty:
            continue

        blended = (
            weight * merged["shipped_mw"].to_numpy(dtype=float)
            + (1.0 - weight) * merged["anchor_mw"].to_numpy(dtype=float)
        )

        published = np.clip(blended * level + shift, 0.0, CAPACITY_MW)

        deviation = merged["actual_mw"].to_numpy(dtype=float) - published

        penalty += float(
            pd.Series(deviation).map(scoring.dsm_penalty).sum()
        )

        deviation_pct = np.abs(deviation) / CAPACITY_MW * 100

        inside.append(float((deviation_pct <= free_band).mean() * 100))
        errors.append(float(np.mean(deviation_pct)))

    return {
        "penalty_rs": penalty,
        "inside_band_pct": float(np.mean(inside)) if inside else np.nan,
        "mae_pct": float(np.mean(errors)) if errors else np.nan,
    }


def best_parameters(paths_by_day, meter, grid):
    """Cheapest (weight, level, shift) over the days given."""

    best = (np.inf, None)

    for weight, level, shift in grid:

        result = stitch(paths_by_day, meter, weight, level, shift)

        if result["penalty_rs"] < best[0]:
            best = (result["penalty_rs"], (weight, level, shift))

    return best[1]


def main():

    parser = argparse.ArgumentParser(
        description="Tune the published schedule against the penalty bands"
    )
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )
    parser.add_argument(
        "--free-band", type=float, default=None,
        help="override the free band %% of capacity, to test how much the "
             "answer depends on a regulation we have assumed rather than "
             "verified (default: whatever score_day_schedules.py uses)"
    )
    parser.add_argument(
        "--min-train-days", type=int, default=4,
        help="days of history required before a day is scored walk-forward"
    )
    parser.add_argument("--out", default="outputs/reports/band_aware_tuning.csv")

    args = parser.parse_args()

    if args.free_band is not None:

        first = scoring.SLABS[0]
        second = scoring.SLABS[1]

        scoring.SLABS = (
            [(0, args.free_band, 0.0),
             (args.free_band, second[1], second[2])]
            + list(scoring.SLABS[2:])
        )

    sweep.dsm_penalty = scoring.dsm_penalty

    paths = sorted(Path(args.folder).glob("*_schedule.csv"))

    if not paths:
        raise SystemExit(f"No schedules in {args.folder}")

    meter = sweep.load_meter()

    by_day = {}

    for path in paths:

        run_time = sweep.run_time_of(path)

        if run_time is not None:
            by_day.setdefault(run_time.date(), []).append(path)

    days = sorted(by_day)

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    levels = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    shifts = [-0.20, -0.10, 0.0, 0.10, 0.20]

    grid = list(itertools.product(weights, levels, shifts))

    print("=" * 78)
    print("BAND-AWARE TUNING")
    print(f"  {len(paths)} schedules over {len(days)} days")
    print(f"  free band {scoring.SLABS[0][1]}% of capacity; "
          f"{len(grid)} parameter combinations, all priced in rupees")
    print("=" * 78)

    # ---------- baselines ----------
    print("\nBASELINES (whole period, no tuning)")
    print("-" * 78)
    print(f"{'setting':<40} {'penalty Rs':>12} {'in band %':>11} {'MAE %':>8}")

    baselines = {
        "what we ship today (w=1.0)": (1.0, 1.0, 0.0),
        "anchor alone (w=0.0)": (0.0, 1.0, 0.0),
        "half way (w=0.5)": (0.5, 1.0, 0.0),
    }

    for label, (weight, level, shift) in baselines.items():

        result = stitch(by_day, meter, weight, level, shift)

        print(f"{label:<40} {result['penalty_rs']:>12,.0f} "
              f"{result['inside_band_pct']:>11.1f} {result['mae_pct']:>8.2f}")

    # ---------- in hindsight ----------
    weight, level, shift = best_parameters(by_day, meter, grid)

    hindsight = stitch(by_day, meter, weight, level, shift)

    print(f"\n{'BEST IN HINDSIGHT (sees every day - a ceiling)':<40} "
          f"{hindsight['penalty_rs']:>12,.0f} "
          f"{hindsight['inside_band_pct']:>11.1f} "
          f"{hindsight['mae_pct']:>8.2f}")
    print(f"{'':<40} weight {weight}, level {level}, shift {shift:+.2f} MW")

    # ---------- walk-forward ----------
    print("\nWALK-FORWARD (each day tuned on earlier days only)")
    print("-" * 78)
    print(f"{'day':<12} {'w':>5} {'level':>6} {'shift':>7} "
          f"{'tuned Rs':>10} {'shipped Rs':>11} {'anchor Rs':>10}")

    rows = []

    for index, day in enumerate(days):

        if index < args.min_train_days:
            continue

        history = {d: by_day[d] for d in days[:index]}
        today = {day: by_day[day]}

        chosen = best_parameters(history, meter, grid)

        tuned = stitch(today, meter, *chosen)
        shipped = stitch(today, meter, 1.0, 1.0, 0.0)
        anchor = stitch(today, meter, 0.0, 1.0, 0.0)

        rows.append({
            "day": str(day),
            "weight": chosen[0], "level": chosen[1], "shift": chosen[2],
            "tuned_rs": tuned["penalty_rs"],
            "shipped_rs": shipped["penalty_rs"],
            "anchor_rs": anchor["penalty_rs"],
            "tuned_in_band_pct": tuned["inside_band_pct"],
            "shipped_in_band_pct": shipped["inside_band_pct"],
        })

        print(f"{day!s:<12} {chosen[0]:>5} {chosen[1]:>6.2f} "
              f"{chosen[2]:>+7.2f} {tuned['penalty_rs']:>10,.0f} "
              f"{shipped['penalty_rs']:>11,.0f} {anchor['penalty_rs']:>10,.0f}")

    if not rows:
        raise SystemExit("Not enough days to walk forward.")

    frame = pd.DataFrame(rows)

    tuned_total = frame["tuned_rs"].sum()
    shipped_total = frame["shipped_rs"].sum()
    anchor_total = frame["anchor_rs"].sum()

    print("-" * 78)
    print(f"{'TOTAL over ' + str(len(frame)) + ' unseen days':<25} "
          f"tuned Rs {tuned_total:>9,.0f}   "
          f"shipped Rs {shipped_total:>9,.0f}   "
          f"anchor Rs {anchor_total:>9,.0f}")

    print(f"\n  tuned vs shipped : {tuned_total - shipped_total:>+10,.0f} Rs "
          f"({(tuned_total / shipped_total - 1) * 100:+.1f}%)")
    print(f"  tuned vs anchor  : {tuned_total - anchor_total:>+10,.0f} Rs "
          f"({(tuned_total / anchor_total - 1) * 100:+.1f}%)")

    print(f"\n  blocks inside the free band: "
          f"{frame['shipped_in_band_pct'].mean():.1f}% shipped -> "
          f"{frame['tuned_in_band_pct'].mean():.1f}% tuned")

    print("\n  Beating the ANCHOR is the bar that matters. Beating what we")
    print("  ship today only means the current settings are wrong, which")
    print("  diagnose_error.py already established.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
