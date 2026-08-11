"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Where does the error actually come from?
=========================================================
Reads every scored schedule and asks four questions that
between them decide what is worth working on next. All of it
is arithmetic over saved files - no API calls, no quota, and
it can be re-run after any change.

  1. HORIZON.   Error against how far ahead the block was.
                A flat curve means the forecast is limited by
                something it never knew; a rising one means it
                is limited by not seeing weather coming.

  2. INPUT x HORIZON.  Which input is closest in each horizon
                band. This is the question the prompt already
                asserts an answer to - "weather should carry
                most of the weight for blocks several hours
                out" - and which nothing has ever checked.

  3. LEVEL vs SHAPE.  Split each run's error into a single
                whole-run scale factor and what is left. If
                most of it is level, the fix is calibration and
                is cheap. If most is shape, the fix is a better
                forecast and is not.

  4. HEADROOM.  The best fixed convex blend of the available
                inputs, fitted in hindsight over all runs. It
                is not achievable - it peeks - but it bounds
                what any smarter COMBINATION of these same
                inputs could deliver. If the oracle blend is
                barely better than the best single input, more
                combination logic is wasted effort and the
                answer has to be a new input.

Run:  python -m tests.diagnose_error
=========================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings


_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})_schedule\.csv$")

# Horizon bands in minutes. Chosen to match how the schedule is
# actually used: the first band is inside the 6-block freeze horizon
# and cannot be changed by a new run at all.
BANDS = [(0, 90), (90, 180), (180, 360), (360, 720), (720, 10_000)]

BAND_LABELS = ["0-90 min (frozen)", "90-180 min", "3-6 h", "6-12 h", "12 h+"]


def load_runs(folder):
    """Every scored schedule, with horizon and the candidate inputs."""

    capacity = settings["plant"]["capacity_mw"]
    timezone = settings["plant"]["timezone"]

    frames = []

    for path in sorted(Path(folder).glob("*_schedule.csv")):

        match = _STAMP.search(path.name)

        if match is None:
            continue

        frame = pd.read_csv(path, parse_dates=["timestamp"])

        if "actual_mw" not in frame.columns:
            continue

        frame = frame[frame["actual_mw"].notna()].copy()

        if frame.empty:
            continue

        run_time = pd.Timestamp(
            f"{match.group(1)} {match.group(2)}:{match.group(3)}",
            tz=timezone,
        )

        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(timezone)

        frame["run"] = path.stem
        frame["day"] = match.group(1)
        frame["run_time"] = run_time
        frame["horizon_min"] = (
            (frame["timestamp"] - run_time).dt.total_seconds() / 60.0
        )
        frame["hour"] = frame["timestamp"].dt.hour

        # Every candidate expressed in MW, so one error metric covers
        # them all. kt columns are turned back into power through the
        # same clear-sky curve the pipeline uses.
        clearsky = frame["clearsky_power_mw"].astype(float)

        frame["cand_anchor"] = frame.get("anchor_mw", np.nan)
        frame["cand_llm"] = frame.get("llm_raw_mw", np.nan)
        frame["cand_published"] = frame["forecast_mw"]
        frame["cand_clearsky"] = clearsky

        for column, name in (("weather_kt", "cand_weather"),
                             ("windy_kt", "cand_windy")):
            frame[name] = (
                frame[column].astype(float) * clearsky
                if column in frame.columns else np.nan
            )

        frames.append(frame)

    if not frames:
        raise SystemExit(f"No scored schedules in {folder}")

    everything = pd.concat(frames, ignore_index=True)
    everything["capacity"] = capacity

    return everything


def pct_error(predicted, actual, capacity):
    """Mean absolute error as % of capacity - the project's metric."""

    ok = np.isfinite(predicted) & np.isfinite(actual)

    if not ok.any():
        return np.nan

    return float(np.mean(np.abs(predicted[ok] - actual[ok])) / capacity * 100)


def oracle_blend(frame, columns, capacity, step=0.05):
    """
    Best fixed convex blend of `columns`, fitted over every block at
    once. It sees the answers, so it is a CEILING and not a proposal -
    the point is what it says about headroom, not about weights.
    """

    usable = frame[columns + ["actual_mw"]].dropna()

    if usable.empty or len(columns) < 2:
        return None, None

    actual = usable["actual_mw"].to_numpy()
    stack = [usable[c].to_numpy() for c in columns]

    best = (np.inf, None)

    grid = np.arange(0.0, 1.0 + step / 2, step)

    def walk(index, remaining, weights):

        nonlocal best

        if index == len(columns) - 1:

            full = weights + [remaining]

            predicted = sum(w * s for w, s in zip(full, stack))

            error = np.mean(np.abs(predicted - actual)) / capacity * 100

            if error < best[0]:
                best = (error, list(full))

            return

        for weight in grid:
            if weight <= remaining + 1e-9:
                walk(index + 1, remaining - weight, weights + [weight])

    walk(0, 1.0, [])

    return best[0], best[1]


def main():

    parser = argparse.ArgumentParser(description="Diagnose forecast error")
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )
    parser.add_argument("--out", default="outputs/reports/error_diagnosis.csv")

    args = parser.parse_args()

    data = load_runs(args.folder)

    capacity = float(data["capacity"].iloc[0])
    actual = data["actual_mw"].to_numpy(dtype=float)

    candidates = [
        ("published (what we ship)", "cand_published"),
        ("anchor / persistence", "cand_anchor"),
        ("LLM unblended", "cand_llm"),
        ("ECMWF weather", "cand_weather"),
        ("Windy scraped", "cand_windy"),
        ("clear sky (no cloud at all)", "cand_clearsky"),
    ]

    print("=" * 78)
    print("ERROR DIAGNOSIS")
    print(f"  {data['run'].nunique()} scored runs over "
          f"{data['day'].nunique()} days, {len(data):,} blocks")
    print(f"  capacity {capacity} MW; error is MAE as % of capacity")
    print("=" * 78)

    # ---------- 1. overall + coverage ----------
    print("\n1. OVERALL, and how much of the data each input even covers")
    print("-" * 78)
    print(f"{'input':<30} {'MAE %':>8} {'bias %':>9} {'blocks':>9} "
          f"{'days':>6}")

    for label, column in candidates:

        values = data[column].to_numpy(dtype=float)
        ok = np.isfinite(values) & np.isfinite(actual)

        if not ok.any():
            print(f"{label:<30} {'-':>8} {'-':>9} {0:>9} {0:>6}")
            continue

        bias = float(np.mean(values[ok] - actual[ok]) / capacity * 100)

        print(f"{label:<30} {pct_error(values, actual, capacity):>8.2f} "
              f"{bias:>+9.2f} {int(ok.sum()):>9,} "
              f"{data.loc[ok, 'day'].nunique():>6}")

    # ---------- 2. horizon ----------
    print("\n2. ERROR BY HORIZON  (does it grow with how far ahead we look?)")
    print("-" * 78)

    header = f"{'band':<20} {'blocks':>8}"
    for label, _ in candidates[:5]:
        header += f" {label.split()[0]:>12}"
    print(header)

    rows = []

    for (low, high), label in zip(BANDS, BAND_LABELS):

        band = data[
            (data["horizon_min"] >= low) & (data["horizon_min"] < high)
        ]

        if band.empty:
            continue

        line = f"{label:<20} {len(band):>8,}"
        record = {"band": label, "blocks": len(band)}

        band_actual = band["actual_mw"].to_numpy(dtype=float)

        for name, column in candidates[:5]:

            error = pct_error(
                band[column].to_numpy(dtype=float), band_actual, capacity
            )

            record[name] = error
            line += f" {error:>12.2f}" if np.isfinite(error) else f" {'-':>12}"

        rows.append(record)
        print(line)

    # ---------- 3. level vs shape ----------
    print("\n3. LEVEL vs SHAPE  (is the error scale, or is it pattern?)")
    print("-" * 78)

    level_errors = []
    raw_errors = []

    for _, run in data.groupby("run"):

        predicted = run["cand_published"].to_numpy(dtype=float)
        truth = run["actual_mw"].to_numpy(dtype=float)

        ok = np.isfinite(predicted) & np.isfinite(truth)

        if ok.sum() < 4 or predicted[ok].sum() <= 0:
            continue

        # The single scale factor that would have been best for this
        # run. What survives it is shape error, which no calibration
        # can remove.
        scale = truth[ok].sum() / predicted[ok].sum()

        raw_errors.append(
            np.mean(np.abs(predicted[ok] - truth[ok])) / capacity * 100
        )
        level_errors.append(
            np.mean(np.abs(predicted[ok] * scale - truth[ok]))
            / capacity * 100
        )

    raw = float(np.mean(raw_errors))
    shape = float(np.mean(level_errors))

    print(f"  published error                        {raw:>6.2f}%")
    print(f"  after a PERFECT per-run scale factor   {shape:>6.2f}%")
    print(f"  removable by level calibration alone   {raw - shape:>6.2f}%"
          f"   ({(raw - shape) / raw * 100:.0f}% of the error)")
    print(f"  irreducible shape error                {shape:>6.2f}%"
          f"   ({shape / raw * 100:.0f}%)")

    # ---------- 4. headroom ----------
    print("\n4. HEADROOM  (best fixed blend of the inputs, fitted in hindsight)")
    print("-" * 78)

    combos = [
        ("anchor + weather", ["cand_anchor", "cand_weather"]),
        ("anchor + LLM", ["cand_anchor", "cand_llm"]),
        ("anchor + weather + LLM",
         ["cand_anchor", "cand_weather", "cand_llm"]),
    ]

    for label, columns in combos:

        subset = data.dropna(subset=columns + ["actual_mw"])

        if subset.empty:
            print(f"  {label:<26} no overlapping blocks")
            continue

        error, weights = oracle_blend(subset, columns, capacity)

        if error is None:
            continue

        singles = min(
            pct_error(
                subset[c].to_numpy(dtype=float),
                subset["actual_mw"].to_numpy(dtype=float),
                capacity,
            )
            for c in columns
        )

        print(f"  {label:<26} {error:>6.2f}%  "
              f"(best single {singles:.2f}%, gain {singles - error:.2f} pts)  "
              f"weights {[round(w, 2) for w in weights]}  "
              f"on {len(subset):,} blocks")

    print("\n  These blends SEE THE ANSWERS. They bound what better")
    print("  combination of the SAME inputs could ever give. A small gain")
    print("  over the best single input means the answer is a new input,")
    print("  not smarter weighting.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
