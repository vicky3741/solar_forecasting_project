"""
=========================================================
Solar Forecasting Project
Block Bias Correction - walk-forward validation
=========================================================
tests/test_block_penalty_pattern.py found the pattern the
mentor asked for: the penalty carries a repeating
time-of-day SHAPE (late morning over-forecast, mid-
afternoon under-forecast). This script decides whether
feeding that shape back into the model actually saves
money on days it has never seen - the same bar every other
signal in this project had to clear.

DISCIPLINE (why this is not just a hyperparameter search):

  * RECURSIVE walk-forward. Day D is corrected using only
    days before it, and those training days are themselves
    the CORRECTED versions - which is what deployment
    actually looks like once the correction is live. A
    non-recursive test would learn from a bias the live
    system no longer has.
  * Two CONTROLS. "flat" shifts the whole day by one
    number; "none" is today's model. If the per-block
    version cannot beat a flat shift, then the pattern is
    just a level error and none of this block talk is
    earning anything.
  * Scored in RUPEES (the real DSM slabs), not in percent
    deviation - the mentor's question was about penalty.
  * A setting is only worth shipping if it helps across
    the WHOLE lookback range, not at one lucky cell. With
    7-9 test days, the best cell of a 60-cell grid means
    nothing on its own.

Run:  python -m tests.test_block_bias_experiment
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.block_bias_correction import BlockBiasCorrector
from tests.test_block_penalty_pattern import dsm_penalty_rs


CAPACITY_MW = settings["plant"]["capacity_mw"]
SCHEDULE_DIR = Path(settings["outputs"]["schedules"])

LOOKBACKS = (3, 4, 5, 7)
STRENGTHS = (0.25, 0.4, 0.5, 0.75)
SMOOTHINGS = (2, 4, 6, 8)


def load_days():
    """date string -> scored blocks of that day's reconstructed schedule."""

    days = {}

    for path in sorted(SCHEDULE_DIR.glob("day_schedule_*.csv")):

        stem = path.stem.replace("day_schedule_", "")
        if "_" in stem:
            continue

        frame = pd.read_csv(path)

        if "actual_is_real" in frame.columns:
            frame = frame[frame["actual_is_real"].astype(bool)]

        frame = frame.dropna(subset=["actual_mw", "scheduled_mw"])

        if not frame.empty:
            days[stem] = frame.reset_index(drop=True)

    return days


def deviation_pct(scheduled, actual):

    return float(np.mean(np.abs(scheduled - actual) / CAPACITY_MW) * 100)


def walk_forward(days, lookback, strength, smooth, mode):
    """
    mode: "block" = the shipped per-block profile
          "flat"  = one median shift for the whole day (control)
          "none"  = no correction (baseline)
    """

    corrector = BlockBiasCorrector(
        lookback_days=lookback, strength=strength, smooth_blocks=smooth
    )

    ordered = sorted(days)
    corrected = {}
    rows = []

    for index, day in enumerate(ordered):

        today = days[day].copy()

        if index < lookback or mode == "none":
            corrected[day] = today
            if mode == "none" or index >= lookback:
                rows.append(score(day, today, today["scheduled_mw"]))
            continue

        history = [(d, corrected[d]) for d in ordered[index - lookback:index]]

        if mode == "block":
            profile_kw = corrector.learn_profile(history)
            shift_mw = today["block"].map(profile_kw / 1000.0).fillna(0.0).to_numpy()
        else:
            past = pd.concat([frame for _, frame in history], ignore_index=True)
            flat = float((past["actual_mw"] - past["scheduled_mw"]).median())
            shift_mw = np.full(len(today), flat)

        new_scheduled = np.clip(
            today["scheduled_mw"] + strength * shift_mw, 0, CAPACITY_MW
        )

        rows.append(score(day, today, new_scheduled))

        applied = today.copy()
        applied["scheduled_mw"] = new_scheduled
        corrected[day] = applied

    return pd.DataFrame(rows)


def score(day, frame, scheduled):

    return {
        "day": day,
        "penalty_rs": float(dsm_penalty_rs(frame["actual_mw"] - scheduled).sum()),
        "deviation_pct": deviation_pct(scheduled, frame["actual_mw"]),
    }


def main():

    days = load_days()

    if len(days) < max(LOOKBACKS) + 2:
        print(f"Only {len(days)} day schedules available in {SCHEDULE_DIR}; "
              f"need at least {max(LOOKBACKS) + 2} to walk forward. Build more "
              "days with tests.generate_schedule_for_day first.")
        return

    print("=" * 84)
    print(f"BLOCK BIAS CORRECTION - recursive walk-forward over "
          f"{len(days)} days ({min(days)} to {max(days)})")
    print("=" * 84)

    results = []

    for lookback in LOOKBACKS:

        baseline = walk_forward(days, lookback, 0.0, 0, "none")
        # Baseline must be scored on exactly the same test days as the
        # variants, so it is re-derived per lookback (the first
        # `lookback` days are training-only and never scored).
        baseline = baseline[baseline["day"].isin(sorted(days)[lookback:])]

        base_rs = baseline["penalty_rs"].mean()
        base_dev = baseline["deviation_pct"].mean()

        for mode in ("block", "flat"):
            for strength in STRENGTHS:
                for smooth in (SMOOTHINGS if mode == "block" else (0,)):

                    run = walk_forward(days, lookback, strength, smooth, mode)

                    merged = run.merge(baseline, on="day", suffixes=("", "_base"))

                    results.append({
                        "mode": mode,
                        "lookback_days": lookback,
                        "strength": strength,
                        "smooth_blocks": smooth,
                        "test_days": len(merged),
                        "base_rs_per_day": round(base_rs, 1),
                        "new_rs_per_day": round(run["penalty_rs"].mean(), 1),
                        "saving_rs_per_day": round(base_rs - run["penalty_rs"].mean(), 1),
                        "saving_pct": round(
                            (1 - run["penalty_rs"].mean() / base_rs) * 100, 1),
                        "days_cheaper": int(
                            (merged["penalty_rs"] < merged["penalty_rs_base"]).sum()),
                        "deviation_change_pts": round(
                            run["deviation_pct"].mean() - base_dev, 3),
                    })

    table = pd.DataFrame(results)

    # ---------------- the control question first ----------------

    print()
    print("-" * 84)
    print("CONTROL: does the per-block SHAPE beat a flat whole-day shift?")
    print("-" * 84)

    control = table.groupby("mode")["saving_rs_per_day"].agg(["mean", "max"]).round(1)
    print(control.to_string())

    flat_best = table[table["mode"] == "flat"]["saving_rs_per_day"].max()
    block_mean = table[table["mode"] == "block"]["saving_rs_per_day"].mean()

    if flat_best >= block_mean:
        print()
        print("A flat shift does as well as the per-block profile. The error is a")
        print("LEVEL problem, not a block-pattern problem - stop here and tune the")
        print("level (performance_ratio / weather weight) instead.")

    # ---------------- stability across lookbacks ----------------

    block = table[table["mode"] == "block"]

    grid = block.pivot_table(
        index=["strength", "smooth_blocks"],
        columns="lookback_days",
        values="saving_rs_per_day",
    ).round(1)

    print()
    print("-" * 84)
    print("STABILITY: Rs/day SAVED (positive = cheaper) by setting, at every lookback")
    print("-" * 84)
    print(grid.to_string())

    summary = pd.DataFrame({
        "worst": grid.min(axis=1),
        "mean": grid.mean(axis=1).round(1),
    })

    # A setting ships only if it saves money at EVERY lookback - one
    # lucky cell in a 64-cell grid over 7 test days is noise.
    robust = summary[summary["worst"] > 0].sort_values("mean", ascending=False)

    print()
    print("-" * 84)
    print("SETTINGS THAT SAVE MONEY AT EVERY LOOKBACK")
    print("-" * 84)

    if robust.empty:
        print("None. No (strength, smoothing) pair helps consistently -")
        print("do NOT enable block_bias_correction on this evidence.")
        save(table)
        return

    print(robust.to_string())

    best_strength, best_smooth = robust.index[0]

    # The mentor asked for the last 4-5 days specifically, and the
    # weather bias correction already uses a 5-day window - so among
    # robust settings, report the 5-day one as the shipping candidate.
    candidate = block[
        (block["strength"] == best_strength)
        & (block["smooth_blocks"] == best_smooth)
        & (block["lookback_days"] == 5)
    ].iloc[0]

    print()
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"  strength      : {best_strength}")
    print(f"  smooth_blocks : {best_smooth}  (+/-{best_smooth * 15} minutes)")
    print(f"  lookback_days : 5")
    print()
    print(f"  penalty now       : Rs {candidate['base_rs_per_day']:,.0f} / day")
    print(f"  penalty corrected : Rs {candidate['new_rs_per_day']:,.0f} / day")
    print(f"  saving            : Rs {candidate['saving_rs_per_day']:,.0f} / day "
          f"({candidate['saving_pct']:.1f}%)")
    print(f"  days cheaper      : {candidate['days_cheaper']}/{candidate['test_days']} "
          "unseen days")
    print(f"  deviation change  : {candidate['deviation_change_pts']:+.3f} pts")
    print()
    print("Modest, and stated as modest: this trims a systematic slice off the")
    print("penalty, it does not fix a badly forecast day. The big misses are")
    print("still weather the model did not see coming.")

    per_day = walk_forward(days, 5, best_strength, best_smooth, "block")
    base_day = walk_forward(days, 5, 0.0, 0, "none")
    base_day = base_day[base_day["day"].isin(per_day["day"])]

    comparison = per_day.merge(base_day, on="day", suffixes=("_corrected", "_now"))
    comparison["saving_rs"] = (
        comparison["penalty_rs_now"] - comparison["penalty_rs_corrected"]
    )

    print()
    print("Per unseen day (Rs):")
    print(comparison[[
        "day", "penalty_rs_now", "penalty_rs_corrected", "saving_rs"
    ]].round(0).to_string(index=False))

    save(table)


def save(table):

    out = Path(settings["outputs"]["reports"]) / "block_bias_experiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print()
    print(f"Saved full grid: {out}")


if __name__ == "__main__":
    main()
