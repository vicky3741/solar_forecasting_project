"""
=========================================================
Solar Forecasting Project
Module-Temperature Derate Experiment
=========================================================
The mentor's "Static Data of Solar Power Plant" sheet
(2026-08-01) confirmed the array is MONOCRYSTALLINE silicon
- 9968 x 550 Wp modules, 5.4824 MWp DC behind a 5.1 MW AC
inverter, on a FIXED structure. Mono c-Si loses roughly
0.35-0.40 % of its output per degree C above the 25 C STC
rating, and the meter file has carried a `MOD TEMP` column
all along that the pipeline has never read.

Midday module temperature at this plant runs 40.6 C median
and 58.1 C peak, and the DAY-TO-DAY spread of the midday
median is 19.8 C - worth ~7 % of output between a cool day
and a hot one. `kt` cannot see any of that: it is a pure
irradiance ratio. So the question this experiment answers
is whether a temperature term buys anything the existing
signals do not already carry.

THE TRAP THIS EXPERIMENT IS BUILT TO AVOID
------------------------------------------
`performance_ratio` (0.80) is an EMPIRICALLY TUNED constant.
It already absorbs the average temperature loss, along with
soiling, wiring, inverter efficiency and the DC/AC ratio.
Bolting a derate on top of the tuned PR would lower every
forecast and, on a dataset where we over-forecast, would
"help" for entirely the wrong reason.

So PR is RE-TUNED JOINTLY with the temperature coefficient,
and the baseline is the best PR with no temperature term at
all. The only thing being tested is whether the temperature
term earns its place ON TOP of a PR that has been given the
same chance to fit.

STAGE 1 (this script): ORACLE CEILING
-------------------------------------
Future blocks are derated using the MEASURED module
temperature - which is lookahead and could never ship. It
is deliberately the most generous possible version. If even
the oracle cannot beat a re-tuned PR, no forecastable
version ever will, and the idea is dead for the cost of one
script. Only if the oracle wins is it worth building the
Faiman/NOCT chain that predicts module temperature from
forecast ambient temperature and POA.

Vision is OFF in both arms: the two arms must differ only
in the temperature term, and holding Gemini constant across
217 runs is neither free nor reproducible.

Verdict is LEAVE-ONE-DAY-OUT, matching the discipline that
disabled residual_correction (-0.37 pts) and capped
case_based_correction at strength 0.5 (+0.47 pts).

Run:  python -m tests.test_module_temperature_experiment
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.evaluation import metrics
from modules.forecasting.predictor import HybridPredictor
from modules.preprocessing.preprocess import DataPreprocessor

CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]

STC_TEMP_C = 25.0

# Candidate grids. PR brackets the tuned 0.80 generously so the
# no-temperature baseline is never handicapped by a grid that is too
# narrow. Gamma brackets the -0.0035..-0.0040 /C datasheet range for
# mono c-Si, with 0.0 (no derate) as the baseline member.
PR_GRID = np.round(np.arange(0.60, 0.95001, 0.01), 4)
GAMMA_GRID = np.round(np.arange(0.0, -0.00701, -0.0005), 5)

# Scaling probe: blend_signals clips at plant capacity, which would
# break the linear "scale the unit curve by PR" shortcut. Blending at a
# deliberately low PR keeps every block far below the clip, so the
# unclipped unit curve can be recovered by dividing it back out. Asserted
# at runtime rather than assumed.
PROBE_PR = 0.30


def load_data():
    """Every daily meter CSV, preprocessed per-day then concatenated."""

    pre = DataPreprocessor()

    frames = [
        pre.preprocess(
            file_path=path,
            required_columns=["TimeStamp"],
            timestamp_column="TimeStamp",
        )
        for path in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def collapse_days(data, days, actual):
    """
    One Chronos pass per (day, scheduling time), immediately collapsed
    into each day's FINAL published schedule.

    The collapse is the whole trick that makes the grid cheap. Which run
    ends up owning a block is decided by timestamp alone - the last run
    before it wins - and that ordering does not depend on PR or gamma at
    all. So the expensive reconstruction happens ONCE per day here, and
    every (PR x gamma) candidate afterwards is a single multiply over a
    ~50-element array instead of a rescan of every run in the dataset.

    Returns {day: (unit_kw, mod_temp, actual_kw)} aligned block-for-block
    over the blocks that have a real meter reading.
    """

    predictor = HybridPredictor()

    temps = data.set_index("timestamp")["mod_temp"]

    collapsed = {}

    for day_index, day in enumerate(days, start=1):

        print(f"  [{day_index}/{len(days)}] {day}", flush=True)

        schedule = {}          # timestamp -> unit kW, last writer wins

        for run_str in RUN_TIMES:

            hour, minute = map(int, run_str.split(":"))
            run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            signals = predictor.compute_signals(data, run_time)

            # No vision: both arms must differ only in the temperature term.
            probe = predictor.blend_signals(
                signals, performance_ratio=PROBE_PR, vision_adjustment=0.0
            )

            unit_kw = probe["final_forecast_kw"].to_numpy() / PROBE_PR

            if unit_kw.max() * PR_GRID.max() >= CAPACITY_KW:
                raise AssertionError(
                    "Capacity clipping would bind inside the PR grid - the "
                    "linear unit-curve shortcut is invalid; lower PROBE_PR."
                )

            timestamps = pd.DatetimeIndex(probe["timestamp"])

            for timestamp, value in zip(timestamps, unit_kw):
                if timestamp > run_time:
                    schedule[timestamp] = value

        # Keep only blocks the meter really measured - both arms then
        # always cover exactly the same blocks.
        keep = [t for t in sorted(schedule) if t in actual.index]

        if not keep:
            continue

        measured = actual.reindex(keep).to_numpy(dtype=float)
        good = ~np.isnan(measured)

        if not good.any():
            continue

        keep = list(np.array(keep)[good])

        collapsed[day] = (
            np.array([schedule[t] for t in keep]),
            temps.reindex(keep).to_numpy(dtype=float),
            measured[good],
        )

    return collapsed


def deviation_for(collapsed_day, performance_ratio, gamma):
    """
    Average percentage deviation for one day at one (PR, gamma).

    The derate is (1 + gamma * (T_module - 25)). Blocks with no module
    temperature reading fall back to no derate rather than being dropped,
    so both arms always cover exactly the same blocks.
    """

    unit_kw, mod_temp, measured = collapsed_day

    derate = np.where(
        np.isnan(mod_temp), 1.0, 1.0 + gamma * (mod_temp - STC_TEMP_C)
    )

    forecast = np.clip(unit_kw * performance_ratio * derate, 0, CAPACITY_KW)

    return metrics.average_percentage_deviation(forecast, measured, CAPACITY_KW)


def main():

    data = load_data()

    actual = data[data["is_real_measurement"].fillna(False)]
    actual = actual.set_index("timestamp")["active_power_kw"]

    days = sorted(data["timestamp"].dt.date.unique())

    print("=" * 78)
    print("MODULE-TEMPERATURE DERATE - ORACLE CEILING TEST")
    print("=" * 78)
    print(f"days: {len(days)}   scheduling times/day: {len(RUN_TIMES)}")
    print(f"PR grid: {PR_GRID.min():.2f}..{PR_GRID.max():.2f}   "
          f"gamma grid: {GAMMA_GRID.min():.5f}..0.0 per degC")
    print("Module temperature is MEASURED (lookahead) - this is a ceiling,")
    print("not a shippable configuration.")
    print()
    # The Chronos pass is the only expensive part and does not depend on
    # PR or gamma, so it is cached - re-running to widen the grid or
    # change the verdict rule then costs seconds instead of minutes.
    cache = Path("outputs/module_temperature_runs.pkl")

    if cache.exists():
        print(f"Reusing cached Chronos runs: {cache}")
        collapsed = pd.read_pickle(cache)
    else:
        print("Running Chronos once per scheduling time...")
        collapsed = collapse_days(data, days, actual)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(collapsed, cache)
        print(f"Cached: {cache}")

    # Deviation for every (day, PR, gamma) - computed once, reused by
    # every leave-one-day-out fold below.
    print()
    print("Scoring the parameter grid...")

    table = {}

    for pr in PR_GRID:
        for gamma in GAMMA_GRID:
            for day, collapsed_day in collapsed.items():
                table[(pr, gamma, day)] = deviation_for(collapsed_day, pr, gamma)

    scored_days = sorted(collapsed)

    def mean_over(days_subset, pr, gamma):
        values = [table[(pr, gamma, d)] for d in days_subset
                  if (pr, gamma, d) in table]
        return float(np.mean(values)) if values else None

    # ---- leave-one-day-out ----
    # Two arms tuned identically: the baseline may pick any PR but is
    # locked to gamma = 0; the candidate may pick any (PR, gamma) pair.
    baseline_out, candidate_out, picks = [], [], []

    for held_out in scored_days:

        train = [d for d in scored_days if d != held_out]

        best_baseline = min(
            PR_GRID, key=lambda p: mean_over(train, p, 0.0) or 1e9
        )

        best_candidate = min(
            ((p, g) for p in PR_GRID for g in GAMMA_GRID),
            key=lambda pg: mean_over(train, pg[0], pg[1]) or 1e9,
        )

        baseline_out.append(table[(best_baseline, 0.0, held_out)])
        candidate_out.append(
            table[(best_candidate[0], best_candidate[1], held_out)]
        )
        picks.append((held_out, best_baseline, best_candidate))

    baseline_mean = float(np.mean(baseline_out))
    candidate_mean = float(np.mean(candidate_out))
    gain = baseline_mean - candidate_mean
    better = sum(1 for b, c in zip(baseline_out, candidate_out) if c < b)

    print()
    print("=" * 78)
    print("LEAVE-ONE-DAY-OUT RESULT (held-out days only)")
    print("=" * 78)
    print(f"  baseline  (best PR, no temperature term) : {baseline_mean:.3f}%")
    print(f"  candidate (best PR + temperature derate) : {candidate_mean:.3f}%")
    print(f"  gain                                     : {gain:+.3f} pts")
    print(f"  days improved                            : {better}/{len(scored_days)}")

    chosen = [pg for _, _, pg in picks]
    gammas = [g for _, g in chosen]
    prs = [p for p, _ in chosen]
    print(f"  gamma chosen (median)                    : {np.median(gammas):+.5f} /degC")
    print(f"  PR chosen with temperature (median)      : {np.median(prs):.3f}")
    print(f"  PR chosen without temperature (median)   : "
          f"{np.median([p for _, p, _ in picks]):.3f}")

    print()
    if gain > 0.10 and better > len(scored_days) / 2:
        print("VERDICT: the oracle beats a re-tuned PR. Worth building the")
        print("         forecastable module-temperature chain (Faiman from")
        print("         Open-Meteo ambient temperature + forecast POA) and")
        print("         re-running this test against PREDICTED temperature.")
    else:
        print("VERDICT: the oracle does NOT beat a re-tuned PR. Even with")
        print("         perfect knowledge of module temperature the term earns")
        print("         nothing, so a forecastable version cannot either.")
        print("         Record the negative result; leave the model alone.")

    out = Path("outputs/reports/module_temperature_experiment.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"performance_ratio": pr, "gamma_per_degC": g, "date": str(d),
             "deviation_pct": round(v, 4)}
            for (pr, g, d), v in table.items()
        ]
    ).to_csv(out, index=False)

    print()
    print(f"Saved full grid: {out}")


if __name__ == "__main__":
    main()
