"""
=========================================================
Solar Forecasting Project
Adaptive Weather-Weight Experiment
=========================================================
Mentor question (2026-08-04): the blend weights are fixed -
can the model decide for itself which signal to trust more,
given the conditions?

Most of the blend is ALREADY adaptive. Chronos scales with
how much of today has been observed (context_ratio), the
vision adjustment is per-block and scales with the model's
own confidence, and the weather bias factor is recomputed
from the last 5 days of measured error. The one genuinely
static knob is `weather_weight` (0.65), and it is applied
identically to every block.

That is the part worth attacking, because it is fixed in a
way that is clearly wrong in principle:

  * 15 minutes ahead, what the meter is reading RIGHT NOW is
    hugely informative.
  * 5 hours ahead, it is nearly worthless, and the weather
    model is essentially all we have.

Yet both blocks currently get weather_weight = 0.65.

TWO ARMS ARE TESTED
-------------------
  A. HORIZON-DEPENDENT - the weight ramps from w_near (for
     blocks just ahead) to w_far (for distant blocks).
  B. VOLATILITY-DEPENDENT - the weight rises when today's
     recent clear-sky index has been jumpy, on the theory
     that persistence is worthless on an unstable day.
     2026-08-03 is the motivating case: actual generation
     swung 3.40 -> 2.42 -> 3.98 -> 2.15 -> 4.06 MW across
     midday and every signal smoothed straight through it.

THE TRAP THIS AVOIDS
--------------------
Both arms have MORE freedom than a single fixed number, so
comparing them against the shipped 0.65 would be rigged -
they could win purely by re-tuning. The baseline is
therefore the BEST CONSTANT weather_weight, tuned over the
same grid and scored the same way. The only thing being
tested is whether letting the weight VARY earns anything
beyond letting it be re-fitted.

Verdict is leave-one-day-out, the same rule that disabled
residual_correction (-0.37 pts) and rejected the module
temperature derate (+0.000 pts, 0/31 days).

Vision is OFF in every arm. It has to be held constant for
the comparison to mean anything, and 238 Gemini calls is
neither free (20/day shared quota) nor reproducible. The
weight that wins here should be re-checked with vision on
before shipping.

Run:  python -m tests.test_adaptive_weather_weight_experiment
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

# Blending at a deliberately low performance ratio keeps every block far
# below the capacity clip, so the two probe curves stay LINEAR and any
# weight can be recovered by interpolating between them. Asserted at
# runtime rather than assumed.
PROBE_PR = 0.30

# Constant-weight grid - also the baseline arm.
W_GRID = np.round(np.arange(0.0, 0.951, 0.05), 3)

# Horizon ramp: weight moves from w_near to w_far as the horizon grows,
# saturating at HORIZON_SCALE minutes. 240 min is chosen because the
# forecast horizon runs to ~12 h but the useful spread is the first few
# hours; it is not tuned, so it cannot absorb the result.
HORIZON_SCALE = 240.0

# Volatility reference: the std of today's recent kt readings at which
# the weight reaches w_volatile. 0.20 is roughly the spread seen on a
# broken-cloud day versus ~0.02 on a clear one.
VOL_REF = 0.20
VOL_LOOKBACK = 8          # readings (~2 h at 15 min) before the run


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


def recent_kt_volatility(predictor, data, run_time):
    """
    Standard deviation of today's last few clear-sky-index readings
    before the run - a cheap "is the sky jumpy right now" measure that
    uses only data genuinely available at that instant.
    """

    kt_series = predictor.get_today_kt_series(data, run_time)

    recent = kt_series["clear_sky_index"].tail(VOL_LOOKBACK)

    if len(recent) < 3:
        return 0.0

    return float(recent.std())


def collapse_days(data, days, actual):
    """
    One Chronos pass per (day, scheduling time), collapsed into each
    day's FINAL published schedule.

    The key trick: with the capacity clip out of the way, the blend is
    LINEAR in weather_weight -

        forecast(w) = w * forecast(w=1) + (1 - w) * forecast(w=0)

    so blending twice per run (weight 0 and weight 1) is enough to
    reconstruct ANY weight scheme afterwards with pure arithmetic - per
    block, and without re-running Chronos. Which run owns a block is
    decided by timestamp alone and never depends on the weights, so the
    expensive reconstruction happens once.

    Each surviving block carries the horizon and volatility of the run
    that WROTE it, since that is what an adaptive weight would have
    keyed off at that moment.

    Returns {day: dict of aligned arrays}.
    """

    predictor = HybridPredictor()

    collapsed = {}

    for day_index, day in enumerate(days, start=1):

        print(f"  [{day_index}/{len(days)}] {day}", flush=True)

        schedule = {}     # timestamp -> (f0, f1, horizon_min, volatility)

        for run_str in RUN_TIMES:

            hour, minute = map(int, run_str.split(":"))
            run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            signals = predictor.compute_signals(data, run_time)

            if signals.get("weather_kt") is None:
                # No weather forecast for this run - the weight has
                # nothing to act on, so this run is not comparable
                # across arms. Skip the whole day rather than mix.
                schedule = None
                break

            volatility = recent_kt_volatility(predictor, data, run_time)

            probe0 = predictor.blend_signals(
                signals, performance_ratio=PROBE_PR,
                weather_weight=0.0, vision_adjustment=0.0
            )
            probe1 = predictor.blend_signals(
                signals, performance_ratio=PROBE_PR,
                weather_weight=1.0, vision_adjustment=0.0
            )

            # The guard has to look at the PROBE's own output, not the
            # unit curve: dividing by PROBE_PR scales the numbers back up
            # by ~3x, so comparing THAT against capacity fires on blocks
            # that never actually clipped. What matters is only whether
            # blend_signals clipped while producing the probe - if it did
            # not, f0/f1 are clean unclipped curves and interpolating
            # between them is exact. The final clip is applied later, in
            # deviation_for, after the interpolation.
            if max(probe0["final_forecast_kw"].max(),
                   probe1["final_forecast_kw"].max()) >= CAPACITY_KW:
                raise AssertionError(
                    "Capacity clipping bound while building the probe - the "
                    "linear-interpolation shortcut is invalid; lower PROBE_PR."
                )

            f0 = probe0["final_forecast_kw"].to_numpy() / PROBE_PR
            f1 = probe1["final_forecast_kw"].to_numpy() / PROBE_PR

            timestamps = pd.DatetimeIndex(probe0["timestamp"])
            horizons = (timestamps - run_time).total_seconds().to_numpy() / 60.0

            for ts, a, b, h in zip(timestamps, f0, f1, horizons):
                if ts > run_time:
                    schedule[ts] = (a, b, h, volatility)

        if not schedule:
            continue

        keep = [t for t in sorted(schedule) if t in actual.index]

        if not keep:
            continue

        measured = actual.reindex(keep).to_numpy(dtype=float)
        good = ~np.isnan(measured)

        if not good.any():
            continue

        keep = list(np.array(keep)[good])
        rows = [schedule[t] for t in keep]

        collapsed[day] = {
            "f0": np.array([r[0] for r in rows]),
            "f1": np.array([r[1] for r in rows]),
            "horizon": np.array([r[2] for r in rows]),
            "volatility": np.array([r[3] for r in rows]),
            "actual": measured[good],
        }

    return collapsed


# ---------------- weight schemes ----------------

def weights_constant(day_data, w):
    return np.full_like(day_data["horizon"], float(w))


def weights_horizon(day_data, w_near, w_far):
    """Ramps from w_near at zero horizon to w_far far out."""

    ramp = np.clip(day_data["horizon"] / HORIZON_SCALE, 0.0, 1.0)

    return w_near + (w_far - w_near) * ramp


def weights_volatility(day_data, w_calm, w_volatile):
    """Rises toward w_volatile as today's recent kt gets jumpier."""

    ramp = np.clip(day_data["volatility"] / VOL_REF, 0.0, 1.0)

    return w_calm + (w_volatile - w_calm) * ramp


def deviation_for(day_data, weights, performance_ratio):
    """Average percentage deviation for one day under a weight array."""

    blended = weights * day_data["f1"] + (1.0 - weights) * day_data["f0"]

    forecast = np.clip(blended * performance_ratio, 0, CAPACITY_KW)

    return metrics.average_percentage_deviation(
        forecast, day_data["actual"], CAPACITY_KW
    )


def main():

    data = load_data()

    actual = data[data["is_real_measurement"].fillna(False)]
    actual = actual.set_index("timestamp")["active_power_kw"]

    days = sorted(data["timestamp"].dt.date.unique())

    print("=" * 78)
    print("ADAPTIVE WEATHER-WEIGHT EXPERIMENT")
    print("=" * 78)
    print(f"days: {len(days)}   scheduling times/day: {len(RUN_TIMES)}")
    print("Baseline  : best CONSTANT weather_weight (re-tuned, not the shipped 0.65)")
    print("Arm A     : weight ramps with forecast horizon")
    print("Arm B     : weight ramps with today's recent kt volatility")
    print("Vision OFF in every arm so the arms differ only in the weight.")
    print()
    print("Running Chronos once per scheduling time...")

    collapsed = collapse_days(data, days, actual)

    scored_days = sorted(collapsed)

    if len(scored_days) < 5:
        print(f"\nOnly {len(scored_days)} usable days - not enough to judge.")
        return

    pr = settings["clearsky"]["performance_ratio"]

    print()
    print(f"Scoring the grids over {len(scored_days)} days (PR fixed at {pr})...")

    # Pre-compute every candidate's per-day deviation once.
    const_scores, horizon_scores, vol_scores = {}, {}, {}

    for w in W_GRID:
        for d in scored_days:
            const_scores[(w, d)] = deviation_for(
                collapsed[d], weights_constant(collapsed[d], w), pr)

    pairs = [(a, b) for a in W_GRID for b in W_GRID]

    for a, b in pairs:
        for d in scored_days:
            horizon_scores[(a, b, d)] = deviation_for(
                collapsed[d], weights_horizon(collapsed[d], a, b), pr)
            vol_scores[(a, b, d)] = deviation_for(
                collapsed[d], weights_volatility(collapsed[d], a, b), pr)

    def mean_over(table, key_fn, subset):
        return float(np.mean([table[key_fn(d)] for d in subset]))

    # ---------------- leave-one-day-out ----------------
    results = {"baseline": [], "horizon": [], "volatility": []}
    picks = {"baseline": [], "horizon": [], "volatility": []}

    for held_out in scored_days:

        train = [d for d in scored_days if d != held_out]

        best_w = min(W_GRID, key=lambda w: mean_over(
            const_scores, lambda d, w=w: (w, d), train))
        results["baseline"].append(const_scores[(best_w, held_out)])
        picks["baseline"].append(best_w)

        best_h = min(pairs, key=lambda p: mean_over(
            horizon_scores, lambda d, p=p: (p[0], p[1], d), train))
        results["horizon"].append(horizon_scores[(best_h[0], best_h[1], held_out)])
        picks["horizon"].append(best_h)

        best_v = min(pairs, key=lambda p: mean_over(
            vol_scores, lambda d, p=p: (p[0], p[1], d), train))
        results["volatility"].append(vol_scores[(best_v[0], best_v[1], held_out)])
        picks["volatility"].append(best_v)

    base_mean = float(np.mean(results["baseline"]))

    print()
    print("=" * 78)
    print("LEAVE-ONE-DAY-OUT RESULT (held-out days only)")
    print("=" * 78)
    print(f"  baseline (best constant weight)  : {base_mean:.3f}%")

    for arm in ("horizon", "volatility"):
        arm_mean = float(np.mean(results[arm]))
        gain = base_mean - arm_mean
        better = sum(1 for b, c in zip(results["baseline"], results[arm]) if c < b)
        print(f"  {arm:<32}: {arm_mean:.3f}%   "
              f"gain {gain:+.3f} pts   improved {better}/{len(scored_days)} days")

    print()
    print(f"  median constant weight chosen    : {np.median(picks['baseline']):.2f}")
    hn = np.median([p[0] for p in picks['horizon']])
    hf = np.median([p[1] for p in picks['horizon']])
    print(f"  median horizon pair (near, far)  : ({hn:.2f}, {hf:.2f})")
    vc = np.median([p[0] for p in picks['volatility']])
    vv = np.median([p[1] for p in picks['volatility']])
    print(f"  median volatility pair (calm,vol): ({vc:.2f}, {vv:.2f})")

    print()
    best_arm = min(("horizon", "volatility"), key=lambda a: np.mean(results[a]))
    best_gain = base_mean - float(np.mean(results[best_arm]))
    improved = sum(1 for b, c in zip(results["baseline"], results[best_arm]) if c < b)

    if best_gain > 0.10 and improved > len(scored_days) / 2:
        print(f"VERDICT: the {best_arm} weighting beats a re-tuned constant weight")
        print(f"         by {best_gain:.3f} pts on held-out days. Worth wiring into")
        print("         the blend - then re-check with vision ON before shipping.")
    else:
        print("VERDICT: neither adaptive scheme beats a re-tuned CONSTANT weight.")
        print("         Letting the weight vary earns nothing here, so the fixed")
        print("         weight stays. Record the negative result.")

    out = Path("outputs/reports/adaptive_weather_weight_experiment.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, d in enumerate(scored_days):
        rows.append({
            "date": str(d),
            "baseline_pct": round(results["baseline"][i], 4),
            "horizon_pct": round(results["horizon"][i], 4),
            "volatility_pct": round(results["volatility"][i], 4),
        })
    pd.DataFrame(rows).to_csv(out, index=False)

    print()
    print(f"Saved per-day results: {out}")


if __name__ == "__main__":
    main()
