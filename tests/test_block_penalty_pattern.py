"""
=========================================================
Solar Forecasting Project
Block-Level Penalty Pattern (last N finished days)
=========================================================
MENTOR GUIDANCE 2026-08-06: "Try to analyze the pattern of
the results for the last 4-5 days & finetune the AI model
using that analysis. You can identify the pattern among the
blocks causing higher penalties & convey the same to the
model."

This is the ANALYSIS half. It answers three questions off
the reconstructed day schedules already in outputs/schedules
(the same files the daily mentor report is built from):

  1. WHERE does the money go - which of the 96 blocks carry
     the DSM penalty, and what share each one carries.
  2. Is a penalised block MISSING HIGH or MISSING LOW - i.e.
     is there a repeatable signed bias per block (fixable by
     shifting the forecast) or is it symmetric scatter
     (fixable only by forecasting the weather better)?
  3. Is that bias CONSISTENT day to day, or one bad day
     dragging an average around?

It changes nothing on its own. The correction it justifies
is modules/forecasting/block_bias_correction.py, validated
walk-forward in tests/test_block_bias_experiment.py.

Run:  python -m tests.test_block_penalty_pattern [N_DAYS]
=========================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings


CAPACITY_MW = settings["plant"]["capacity_mw"]
SCHEDULE_DIR = Path(settings["outputs"]["schedules"])

DEFAULT_DAYS = 5

# The DSM slabs used in every penalty report this project has
# built: 0-10% of capacity free, then 0.5 / 0.75 / 1.0 Rs per kWh.
EDGE1 = CAPACITY_MW * 0.10
EDGE2 = CAPACITY_MW * 0.15
EDGE3 = CAPACITY_MW * 0.20
BLOCK_ENERGY_FACTOR = 250      # 0.25 h x 1000 kW/MW


def dsm_penalty_rs(deviation_mw):

    dev = np.abs(deviation_mw)

    return BLOCK_ENERGY_FACTOR * (
        np.clip(np.minimum(dev, EDGE2) - EDGE1, 0, None) * 0.50
        + np.clip(np.minimum(dev, EDGE3) - EDGE2, 0, None) * 0.75
        + np.clip(dev - EDGE3, 0, None) * 1.00
    )


def load_days(n_days):
    """The most recent n_days reconstructed day schedules."""

    paths = [
        path for path in sorted(SCHEDULE_DIR.glob("day_schedule_*.csv"))
        if "_" not in path.stem.replace("day_schedule_", "")
    ]

    frames = []
    for path in paths[-n_days:]:

        frame = pd.read_csv(path)
        frame["date"] = path.stem.replace("day_schedule_", "")

        if "actual_is_real" in frame.columns:
            frame = frame[frame["actual_is_real"].astype(bool)]

        frames.append(frame.dropna(subset=["actual_mw", "scheduled_mw"]))

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # Signed the way an operator reads it: positive means the plant
    # generated MORE than we scheduled (we under-forecast).
    data["deviation_mw"] = data["actual_mw"] - data["scheduled_mw"]
    data["penalty_rs"] = dsm_penalty_rs(data["deviation_mw"])

    return data


def main():

    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS

    data = load_days(n_days)

    if data.empty:
        print(f"No day schedules found in {SCHEDULE_DIR} - run "
              "tests.generate_schedule_for_day for a few days first.")
        return

    days = sorted(data["date"].unique())
    total_rs = data["penalty_rs"].sum()

    print("=" * 78)
    print(f"BLOCK-LEVEL PENALTY PATTERN - last {len(days)} finished days")
    print("=" * 78)
    print(f"days      : {', '.join(days)}")
    print(f"blocks    : {len(data)} scored (real meter readings only)")
    print(f"penalty   : Rs {total_rs:,.0f} total, Rs {total_rs / len(days):,.0f} per day")
    print(f"dead band : +/-{EDGE1:.2f} MW (10% of {CAPACITY_MW} MW) is free of charge")

    # ---------------- 1. where the money goes ----------------

    per_block = data.groupby(["block", "block_time"]).agg(
        days_scored=("penalty_rs", "size"),
        penalty_rs=("penalty_rs", "sum"),
        days_penalised=("penalty_rs", lambda s: int((s > 0).sum())),
        mean_deviation_mw=("deviation_mw", "mean"),
        mean_abs_deviation_mw=("deviation_mw", lambda s: s.abs().mean()),
        days_under_forecast=("deviation_mw", lambda s: int((s > 0).sum())),
    ).reset_index()

    per_block["share_pct"] = per_block["penalty_rs"] / total_rs * 100

    print()
    print("-" * 78)
    print("1. WHERE THE PENALTY LIVES")
    print("-" * 78)

    charged = per_block[per_block["penalty_rs"] > 0]

    if charged.empty:
        print("No block was penalised in this window.")
        return

    first, last = charged["block"].min(), charged["block"].max()
    first_time = charged.loc[charged["block"].idxmin(), "block_time"]
    last_time = charged.loc[charged["block"].idxmax(), "block_time"]

    print(f"Every rupee falls in blocks {first}-{last} ({first_time}-{last_time}).")
    print(f"{len(charged)} of the {len(per_block)} scored blocks were ever charged; "
          f"the other {len(per_block) - len(charged)} never were.")
    print()
    print("Blocks outside that window are not accurate - they are SMALL. Around")
    print("sunrise and sunset the plant produces so little that even a total miss")
    print(f"stays inside the free +/-{EDGE1:.2f} MW dead band. Chasing accuracy there")
    print(f"cannot save money; only blocks {first}-{last} can.")

    top = per_block.sort_values("penalty_rs", ascending=False).head(12)

    print()
    print("Top 12 blocks by penalty:")
    print(top[[
        "block", "block_time", "penalty_rs", "share_pct",
        "days_penalised", "days_scored", "mean_deviation_mw",
    ]].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    cumulative = per_block.sort_values("penalty_rs", ascending=False)
    cumulative["cum_share"] = cumulative["share_pct"].cumsum()
    blocks_for_half = int((cumulative["cum_share"] < 50).sum() + 1)

    print()
    print(f"Half of the whole window's penalty comes from just {blocks_for_half} "
          f"blocks out of {len(per_block)}.")

    # ---------------- 2. bias or scatter ----------------

    print()
    print("-" * 78)
    print("2. BIAS OR SCATTER - is the miss repeatable in one direction?")
    print("-" * 78)
    print("mean deviation = actual - scheduled, averaged over the days.")
    print("  positive -> we UNDER-forecast that block (plant beat the schedule)")
    print("  negative -> we OVER-forecast it")
    print("A |mean| close to the mean ABSOLUTE deviation means the miss is")
    print("systematically one-sided and a shift can fix it. A |mean| near zero")
    print("with a large absolute means it scatters both ways - only a better")
    print("weather signal helps there, not a shift.")

    data["hour"] = pd.to_datetime(data["block_time"], format="%H:%M").dt.hour

    windows = [
        ("06:00-09:45  early", data["hour"] < 9.75),
        ("09:45-11:45  late morning", (data["hour"] >= 9.75) & (data["hour"] < 11.75)),
        ("11:45-14:00  midday", (data["hour"] >= 11.75) & (data["hour"] < 14)),
        ("14:00-16:30  afternoon", (data["hour"] >= 14) & (data["hour"] < 16.5)),
        ("16:30-19:00  evening", data["hour"] >= 16.5),
    ]

    rows = []
    for label, mask in windows:

        chunk = data[mask]
        if chunk.empty:
            continue

        # Consistency: on how many of the days was this window's own
        # mean deviation the same sign as the overall mean?
        per_day = chunk.groupby("date")["deviation_mw"].mean()
        overall = per_day.mean()
        agree = int((np.sign(per_day) == np.sign(overall)).sum())

        rows.append({
            "window": label,
            "blocks": len(chunk),
            "penalty_rs": round(chunk["penalty_rs"].sum(), 0),
            "share_pct": round(chunk["penalty_rs"].sum() / total_rs * 100, 1),
            "mean_dev_mw": round(overall, 3),
            "mean_abs_dev_mw": round(chunk["deviation_mw"].abs().mean(), 3),
            "days_same_sign": f"{agree}/{len(per_day)}",
        })

    window_table = pd.DataFrame(rows)

    print()
    print(window_table.to_string(index=False))

    # ---------------- 3. day-to-day consistency ----------------

    print()
    print("-" * 78)
    print("3. DAY-TO-DAY CONSISTENCY (mean deviation MW per window, per day)")
    print("-" * 78)

    labelled = pd.Series(index=data.index, dtype=object)
    for label, mask in windows:
        labelled[mask] = label

    data["window"] = labelled

    per_day_window = data.pivot_table(
        index="date", columns="window", values="deviation_mw", aggfunc="mean"
    ).round(3)

    print(per_day_window.to_string())

    # ---------------- verdict ----------------

    print()
    print("=" * 78)
    print("WHAT THIS TELLS THE MODEL")
    print("=" * 78)

    fixable = window_table[
        (window_table["penalty_rs"] > 0)
        & (window_table["mean_dev_mw"].abs()
           > 0.35 * window_table["mean_abs_dev_mw"])
    ]

    if fixable.empty:
        print("No window shows a one-sided miss big enough to shift out. The")
        print("penalty here is scatter, not bias - a per-block shift would only")
        print("add noise. Spend the effort on the weather/vision signal instead.")
    else:
        print("These windows miss in a repeatable direction, so shifting their")
        print("blocks by the recent median is worth testing:")
        print()
        for _, row in fixable.iterrows():
            direction = "UNDER-forecast" if row["mean_dev_mw"] > 0 else "OVER-forecast"
            print(f"  {row['window']:28s} {direction} by "
                  f"{abs(row['mean_dev_mw']):.2f} MW on average "
                  f"({row['days_same_sign']} days agree), "
                  f"Rs {row['penalty_rs']:,.0f} at stake")
        print()
        print("That correction is implemented in")
        print("modules/forecasting/block_bias_correction.py and validated")
        print("walk-forward in tests/test_block_bias_experiment.py - which is")
        print("where the decision to ship it is actually made, not here.")

    out = Path(settings["outputs"]["reports"]) / "block_penalty_pattern.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    per_block.round(4).to_csv(out, index=False)

    print()
    print(f"Saved per-block table: {out}")


if __name__ == "__main__":
    main()
