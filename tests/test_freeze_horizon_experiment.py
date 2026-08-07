"""
=========================================================
Solar Forecasting Project
Freeze Horizon (Effective Time) Experiment
=========================================================
What does the mentor's effective-time rule actually COST?

The guide (2026-08-08) says a newly generated schedule may not
touch the next few blocks, because they are already declared
to the grid operator: 6 blocks / 90 min for Sirmour, 3 / 45
for the Kasipet family. This rebuilds the same days with the
freeze OFF and ON and scores both with the same DSM slab
formula the penalty report uses, so the trade-off is a number
rather than an opinion.

HOW TO READ THE RESULT
----------------------
The freeze can only ever make accuracy WORSE. It forces the
schedule to keep older numbers for blocks that fresher meter
data and a newer cloud reading disagree with. On a clear day
that costs nothing; on a changing day it locks in a wrong
value for exactly the blocks that carry the penalty.

So this is NOT a tuning experiment and a negative result is
not a reason to skip the rule. It is a COMPLIANCE question:
if the operator really enforces the horizon, then a schedule
built without it is not one they would accept, and the
no-freeze accuracy we report is flattered by information we
were never allowed to use. What this script measures is how
much of our reported accuracy is that flattery.

VERDICT 2026-08-08, Sirmour, 20 days (Jul 19 - Aug 7):
    deviation  4.190% -> 4.654%   (+0.46 pts)
    penalty    Rs 178 -> Rs 263 per day  (+48%, ~Rs 31k/year)
    13 of 20 days worse on deviation, 15 of 20 dearer
Which is why Sirmour is still configured at 0 pending the
mentor's answer on whether the rule binds there.

CONTROLS
--------
  * Vision is OFF in both arms. The question is about WHEN a
    revision lands, and a cloud signal that is cached for some
    clips and not others adds noise that has nothing to do
    with it. (It also stops the experiment burning the daily
    Gemini free-tier quota, which is what happened on the
    first attempt.)
  * The block-bias corrector reads the SAME real schedule
    history in both arms, so the freeze is the only thing that
    differs. Second-order effect not modelled: in a world that
    had always frozen, that history would itself look
    different.

Run:
    SOLAR_PLANT=sirmour python -m tests.test_freeze_horizon_experiment
    SOLAR_PLANT=kasipet python -m tests.test_freeze_horizon_experiment \
        --off 0 --on 3 --from 2026-07-19 --to 2026-08-06
=========================================================
"""

import argparse
import contextlib
import importlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from config.config import settings
from utils.file_manager import reports_path


BLOCK_ENERGY_FACTOR = 250   # 0.25 h x 1000 kW/MW - MW deviation -> kWh
# DSM slabs, same as tests/build_penalty_report.py:
# 0-10% of capacity is free, then 0.50 / 0.75 / 1.00 Rs per kWh.
SLAB_EDGES_PCT = (10, 15, 20)
SLAB_RATES = (0.5, 0.75, 1.0)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Measure what this plant's freeze horizon costs."
    )
    parser.add_argument("--off", type=int, default=0,
                        help="freeze_blocks for the control arm (default 0)")
    parser.add_argument("--on", type=int, default=None,
                        help="freeze_blocks for the test arm "
                             "(default: this plant's guide value - 6 for "
                             "Sirmour, 3 for the Telangana plants)")
    parser.add_argument("--from", dest="start", default="2026-07-19")
    parser.add_argument("--to", dest="end", default=None)
    return parser.parse_args()


def penalty_rs(error_mw, capacity_mw):
    """Per-block DSM penalty in rupees."""

    deviation = abs(float(error_mw))
    edges = [capacity_mw * pct / 100 for pct in SLAB_EDGES_PCT]

    return BLOCK_ENERGY_FACTOR * (
        max(0.0, min(deviation, edges[1]) - edges[0]) * SLAB_RATES[0]
        + max(0.0, min(deviation, edges[2]) - edges[1]) * SLAB_RATES[1]
        + max(0.0, deviation - edges[2]) * SLAB_RATES[2]
    )


def build_arm(freeze, out_dir, days):
    """Rebuild every day with this freeze horizon, into out_dir."""

    out_dir.mkdir(parents=True, exist_ok=True)

    settings["schedule_rules"]["freeze_blocks"] = freeze
    settings["outputs"]["schedules"] = str(out_dir)

    # Re-import so the generator's module-level constants pick up the
    # settings we just changed.
    sys.modules.pop("tests.generate_schedule_for_day", None)
    generator = importlib.import_module("tests.generate_schedule_for_day")

    data = generator.load_data()

    for day in days:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                generator.main(day_str=str(day), data=data)
        except Exception as error:
            print(f"   ! {day}: {error}")


def score_arm(out_dir, capacity_mw):

    rows = []

    for path in sorted(out_dir.glob("day_schedule_2026-*.csv")):

        if "_" in path.stem.replace("day_schedule_", ""):
            continue

        frame = pd.read_csv(path)
        frame = frame[frame["actual_is_real"].astype(bool)]
        frame = frame.dropna(subset=["actual_mw", "scheduled_mw"])

        if frame.empty:
            continue

        error = frame["scheduled_mw"] - frame["actual_mw"]

        rows.append({
            "day": path.stem.replace("day_schedule_", ""),
            "blocks": len(frame),
            "dev": error.abs().mean() / capacity_mw * 100,
            "pen": sum(penalty_rs(e, capacity_mw) for e in error),
        })

    return pd.DataFrame(rows)


def main():

    args = parse_args()

    plant = settings["plant"]
    capacity_mw = plant["capacity_mw"]

    freeze_on = args.on
    if freeze_on is None:
        freeze_on = 6 if plant["key"] == "sirmour" else 3

    end = args.end or str(pd.Timestamp.now().date())
    days = [d.date() for d in pd.date_range(args.start, end)]

    print("=" * 92)
    print(f"FREEZE HORIZON EXPERIMENT - {plant['name']} ({capacity_mw} MW)")
    print("=" * 92)
    print(f"control arm : freeze_blocks = {args.off}")
    print(f"test arm    : freeze_blocks = {freeze_on} "
          f"({freeze_on * 15} minutes)")
    print(f"days        : {days[0]} .. {days[-1]}")
    print("vision      : OFF in both arms (see the note at the top)")
    print()

    # Both arms must learn their block-bias shape from the same real
    # history, so the freeze is the only difference between them.
    settings["block_bias_correction"]["schedule_dir"] = \
        settings["outputs"]["schedules"]

    settings["vision"]["provider"] = ""
    settings["vision"]["use_s3_video_fallback"] = False

    workspace = Path(tempfile.mkdtemp(prefix="freeze_ab_"))

    results = {}

    for freeze in (args.off, freeze_on):
        print(f"building {len(days)} days with freeze_blocks={freeze} ...")
        arm_dir = workspace / f"freeze_{freeze}"
        build_arm(freeze, arm_dir, days)
        results[freeze] = score_arm(arm_dir, capacity_mw)
        print(f"  {len(results[freeze])} days scored")

    control, test = results[args.off], results[freeze_on]

    merged = control.merge(test, on="day", suffixes=("_off", "_on"))
    merged["dev_change"] = merged["dev_on"] - merged["dev_off"]
    merged["pen_change"] = merged["pen_on"] - merged["pen_off"]

    print()
    print(f"{'day':12s} {'blocks':>6s} {'dev off':>8s} {'dev on':>8s} "
          f"{'change':>8s} {'Rs off':>8s} {'Rs on':>8s} {'change':>8s}")

    for _, row in merged.iterrows():
        print(f"{row['day']:12s} {row['blocks_off']:6.0f} "
              f"{row['dev_off']:8.2f} {row['dev_on']:8.2f} "
              f"{row['dev_change']:+8.2f} {row['pen_off']:8.0f} "
              f"{row['pen_on']:8.0f} {row['pen_change']:+8.0f}")

    dev_change = merged["dev_change"].mean()
    pen_change = merged["pen_change"].mean()

    print()
    print("=" * 92)
    print(f"days compared        : {len(merged)}")
    print(f"deviation, freeze {args.off}  : {merged['dev_off'].mean():.3f} %")
    print(f"deviation, freeze {freeze_on}  : {merged['dev_on'].mean():.3f} %")
    print(f"  change             : {dev_change:+.3f} pct points "
          f"({'WORSE' if dev_change > 0 else 'BETTER'})")
    print()
    print(f"penalty/day, freeze {args.off}: Rs {merged['pen_off'].mean():.0f}")
    print(f"penalty/day, freeze {freeze_on}: Rs {merged['pen_on'].mean():.0f}")
    print(f"  change             : Rs {pen_change:+.0f}/day "
          f"({'WORSE' if pen_change > 0 else 'BETTER'}), "
          f"Rs {pen_change * 365:+.0f}/year")
    print()
    print(f"days deviation worse : "
          f"{int((merged['dev_change'] > 1e-9).sum())} / {len(merged)}")
    print(f"days penalty dearer  : "
          f"{int((merged['pen_change'] > 1e-9).sum())} / {len(merged)}")

    print()
    print("The freeze can only ever cost accuracy - it withholds fresher")
    print("information from blocks that are already declared. The question")
    print("this answers is not 'is it worth turning on' but 'how much of our")
    print("reported accuracy comes from revisions the operator would not")
    print("have accepted'.")

    output_path = reports_path("freeze_horizon_experiment.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    shutil.rmtree(workspace, ignore_errors=True)

    print()
    print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
