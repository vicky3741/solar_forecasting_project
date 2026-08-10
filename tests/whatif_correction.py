"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Price a candidate BEFORE importing it
=========================================================
Standing rule from the user, 2026-08-09: nothing gets added
to this pipeline until it has been shown to lower the DSM
penalty. This is the harness that shows it.

Every candidate is applied to the ALREADY SAVED schedules and
the whole thing re-priced, so a candidate costs no Gemini
quota to evaluate and can be tested against the same 12 days
as everything else.

CANDIDATES

  block-bias   the production pipeline's block bias correction:
               shift each 15-minute block by `strength` x the
               median (actual - scheduled) that block showed over
               the last few finished days.

               Mentor guidance 2026-08-06, and worth Rs 40/day in
               the production blend. The control matters as much
               as the result: a FLAT whole-day shift made the
               penalty worse there, so the money is in the
               time-of-day SHAPE, not the overall level.

               Relevant here because our failure mode is exactly
               that shape - on 2026-07-27 the model was over on 31
               of 35 blocks, then under on 33 of 35 the other way.

WALK-FORWARD, ALWAYS

The profile for day D is learned only from days STRICTLY BEFORE
D. Learning it from the day being corrected would produce a
beautiful number that means nothing - the correction would be
fitting the answer.

Run:  python -m tests.whatif_correction
      python -m tests.whatif_correction --strength 0.5 --lookback 5
=========================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from modules.scheduling.effective_time import block_number
from tests.score_day_schedules import dsm_penalty, run_time_of, stitch_day


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


def day_frame(paths, meter, column="forecast_mw"):
    """One stitched day, joined to actual generation."""

    stitched = stitch_day(paths, column)

    if not stitched:
        return None

    frame = pd.DataFrame({
        "timestamp": list(stitched),
        "scheduled_mw": list(stitched.values()),
    }).sort_values("timestamp")

    frame["block"] = frame["timestamp"].map(block_number)

    merged = frame.merge(meter, on="timestamp", how="inner").dropna(
        subset=["actual_mw"]
    )

    return merged if not merged.empty else None


def learn_profile(history, smooth_blocks=6, max_shift_mw=1.0):
    """
    One bias number per block: the median (actual - scheduled) over the
    history, smoothed across neighbouring blocks.

    Smoothing is not cosmetic. A raw per-block median off five days is
    five noisy samples; unsmoothed it made the production penalty WORSE.
    The recoverable pattern is a broad time-of-day shape, not a
    per-block offset.
    """

    if not history:
        return {}

    combined = pd.concat(history, ignore_index=True)

    combined["error_mw"] = combined["actual_mw"] - combined["scheduled_mw"]

    profile = combined.groupby("block")["error_mw"].median()

    counts = combined.groupby("block")["error_mw"].size()

    # A block seen on fewer than two days is guesswork; drop it before
    # smoothing so it cannot bleed into its neighbours.
    profile = profile[counts >= 2]

    if profile.empty:
        return {}

    profile = profile.sort_index()

    if smooth_blocks:
        profile = profile.rolling(
            2 * smooth_blocks + 1, center=True, min_periods=1
        ).mean()

    return profile.clip(-max_shift_mw, max_shift_mw).to_dict()


def learn_level(history, bounds=(0.5, 1.5), min_blocks=20):
    """
    A single multiplier: total actual over total scheduled across the
    history. Team 1's step 9 - "0.7 means the formula has been running
    30% too high, on average".

    Deliberately ONE number, where learn_profile gives one per block.
    They fix different things: a level correction moves the whole day,
    a shape correction moves the hours differently. The production
    pipeline's control test found a flat whole-day shift made its
    penalty WORSE while the block shape helped, so the two must be
    priced separately rather than assumed to compose.

    Returns None below `min_blocks`, which keeps a ratio built from a
    handful of blocks from rescaling a whole day.
    """

    if not history:
        return None

    combined = pd.concat(history, ignore_index=True)

    if len(combined) < min_blocks:
        return None

    scheduled = float(combined["scheduled_mw"].sum())

    if scheduled <= 0:
        return None

    ratio = float(combined["actual_mw"].sum()) / scheduled

    return float(np.clip(ratio, *bounds))


def price(by_day, meter, strength, lookback, smooth_blocks, min_days,
          candidate="block-bias", shape_strength=0.25):
    """
    Total penalty with the correction applied walk-forward, and without.
    """

    days = sorted(by_day)

    rows = []
    history = []

    for day in days:

        frame = day_frame(by_day[day], meter)

        if frame is None:
            continue

        before = float(
            (frame["actual_mw"] - frame["scheduled_mw"]).map(dsm_penalty).sum()
        )

        corrected = frame["scheduled_mw"].to_numpy()
        applied = False

        # BOTH, in the order the pipeline would apply them: level first,
        # then shape. They fix different things, but they are learned
        # from the SAME residuals, so some of what the level correction
        # removes is residual the shape correction would also have
        # removed. Applying both is not the sum of applying each - which
        # is why this is priced rather than assumed.
        if candidate == "both":

            factor = (
                learn_level(history[-lookback:])
                if len(history) >= min_days else None
            )

            if factor is not None:
                corrected = np.clip(
                    corrected * (1 + strength * (factor - 1)),
                    0.0, CAPACITY_MW,
                )
                applied = True

            profile = (
                learn_profile(history[-lookback:], smooth_blocks)
                if len(history) >= min_days else {}
            )

            if profile:
                shift = frame["block"].map(profile).fillna(0.0).to_numpy()
                corrected = np.clip(
                    corrected + shape_strength * shift, 0.0, CAPACITY_MW
                )
                applied = True

        elif candidate == "level":

            factor = (
                learn_level(history[-lookback:])
                if len(history) >= min_days else None
            )

            if factor is not None:
                corrected = np.clip(
                    corrected * (1 + strength * (factor - 1)),
                    0.0, CAPACITY_MW,
                )
                applied = True

        else:

            profile = (
                learn_profile(history[-lookback:], smooth_blocks)
                if len(history) >= min_days else {}
            )

            if profile:
                shift = frame["block"].map(profile).fillna(0.0).to_numpy()
                corrected = np.clip(
                    corrected + strength * shift, 0.0, CAPACITY_MW
                )
                applied = True

        after = float(
            pd.Series(frame["actual_mw"].to_numpy() - corrected)
            .map(dsm_penalty).sum()
        )

        rows.append({
            "day": day,
            "blocks": len(frame),
            "penalty_before": before,
            "penalty_after": after,
            "change_rs": after - before,
            "corrected": applied,
        })

        # History is appended AFTER scoring, so the profile that
        # corrected this day never saw it.
        history.append(frame[["block", "scheduled_mw", "actual_mw"]])

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser(
        description="Price a candidate correction before importing it"
    )
    parser.add_argument("--folder", default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--smooth", type=int, default=None)
    parser.add_argument(
        "--candidate", default="block-bias",
        choices=("block-bias", "level", "both"),
        help="block-bias = one shift per block (time-of-day shape); "
             "level = one multiplier for the whole day; both = level "
             "then shape, as the pipeline would apply them"
    )
    parser.add_argument(
        "--shape-strength", type=float, default=0.25,
        help="strength of the block-shape part when --candidate both "
             "(--strength then sets the level part)"
    )

    args = parser.parse_args()

    cfg = settings.get("block_bias_correction", {})

    strength = args.strength if args.strength is not None else cfg.get("strength", 0.5)
    lookback = args.lookback if args.lookback is not None else cfg.get("lookback_days", 5)
    smooth = args.smooth if args.smooth is not None else cfg.get("smooth_blocks", 6)
    min_days = cfg.get("min_days", 4)

    folder = Path(
        args.folder or settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        )
    )

    paths = sorted(folder.glob("*_schedule.csv"))

    if not paths:
        raise SystemExit(f"No schedules in {folder}")

    meter = load_meter()

    by_day = {}

    for path in paths:
        run_time = run_time_of(path)
        if run_time is not None:
            by_day.setdefault(run_time.date(), []).append(path)

    results = price(
        by_day, meter, strength, lookback, smooth, min_days,
        args.candidate, args.shape_strength,
    )

    if results.empty:
        raise SystemExit("Nothing could be priced.")

    title = (
        "anchor self-correction (one multiplier for the whole day)"
        if args.candidate == "level"
        else "block bias correction (one shift per block)"
    )

    print("=" * 86)
    print(f"CANDIDATE: {title}")
    print(f"  strength {strength}, lookback {lookback} days, "
          f"smoothed +/-{smooth} blocks, needs {min_days} days of history")
    print("  walk-forward: each day corrected only by days strictly before it")
    print("=" * 86)

    print(results.to_string(index=False, float_format="%.2f"))

    corrected = results[results["corrected"]]

    print("-" * 86)
    print(f"days priced          : {len(results)} "
          f"({len(corrected)} actually corrected)")
    print(f"penalty before       : Rs {results['penalty_before'].sum():,.0f}")
    print(f"penalty after        : Rs {results['penalty_after'].sum():,.0f}")

    change = results["penalty_after"].sum() - results["penalty_before"].sum()

    if not corrected.empty:
        better = int((corrected["change_rs"] < 0).sum())
        print(f"cheaper on           : {better} of {len(corrected)} "
              "corrected day(s)")

    print()

    if change < 0:
        print(f"VERDICT: IMPORT IT. Saves Rs {abs(change):,.0f} over "
              f"{len(results)} days "
              f"(Rs {abs(change) / max(len(results), 1):,.0f}/day).")
    else:
        print(f"VERDICT: DO NOT IMPORT. Costs Rs {change:,.0f} more over "
              f"{len(results)} days.")

    output = Path("outputs/reports/whatif_block_bias.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
