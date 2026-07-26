"""
=========================================================
Solar Forecasting Project
Weather Bias Correction Experiment
=========================================================
The audit of 2026-07-27 found Open-Meteo systematically
over-forecasts sunlight at Sirmour: +10% in early July,
+49% during the deep-monsoon days (Jul 19-26), with 18 of
21 days over-forecast. The blend trusts this signal 65%,
so its optimism is imported straight into our schedule -
this, not the model core, is the recent over-forecasting.

Candidate fix: a walk-forward bias factor. Before each run,
compare Open-Meteo's archived forecasts with the plant's
measured GHI over the last N days, and scale today's
weather signal by the inverse of that recent bias
(clipped, so one weird day cannot swing it).

Honesty rules, as always:
  * factor for day D uses ONLY days before D (walk-forward);
  * archived model runs, never reanalysis - each simulated
    run sees only forecasts that existed at that moment;
  * scored on real measured blocks only;
  * ships only if it helps on unseen days.

Run:  python -m tests.test_weather_bias_experiment
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.predictor import HybridPredictor
from modules.preprocessing.preprocess import DataPreprocessor
from modules.weather.open_meteo import OpenMeteoClient
from modules.evaluation import metrics


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
RUN_TIMES = settings["forecast"]["run_times"]

BIAS_WINDOW_DAYS = 5          # how many recent days teach the factor
FACTOR_CLIP = (0.55, 1.15)    # sane range: never more than ~2x discount, tiny boost allowed
MIN_HISTORY_DAYS = 5          # first scoreable day needs this much bias history


def load_all_data():

    pre = DataPreprocessor()
    frames = [
        pre.preprocess(file_path=p, required_columns=["TimeStamp"],
                       timestamp_column="TimeStamp")
        for p in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def daily_bias_ratio(real_data, weather, day):
    """
    Forecast/actual sunlight ratio for one finished day, using the
    archived forecast available at 06:45 that morning. None when
    either side is missing.
    """

    group = real_data[real_data["timestamp"].dt.date == day]
    daylight = group[group["ghi_w_m2"] > 30].dropna(subset=["ghi_w_m2"])

    if len(daylight) < 15:
        return None

    as_of = pd.Timestamp(day) + pd.Timedelta(hours=6, minutes=45)
    forecast = weather.forecast_ghi_at(daylight["timestamp"], as_of=as_of)

    if forecast is None:
        return None

    actual = daylight["ghi_w_m2"].to_numpy()
    valid = ~np.isnan(forecast)

    if valid.sum() < 15 or actual[valid].sum() <= 0:
        return None

    return float(forecast[valid].sum() / actual[valid].sum())


def bias_factor(ratios_by_day, day):
    """
    Walk-forward correction factor for `day`: the inverse of the
    median bias over the last BIAS_WINDOW_DAYS finished days.
    Median, not mean - one bizarre day must not own the factor.
    """

    prior = [r for d, r in ratios_by_day.items() if d < day and r is not None]

    if len(prior) < 3:
        return 1.0

    recent = prior[-BIAS_WINDOW_DAYS:]

    return float(np.clip(1.0 / np.median(recent), *FACTOR_CLIP))


def main():

    # This experiment applies the correction ITSELF, to raw signals.
    # The production pipeline now ships the same correction (this
    # experiment's positive verdict is why), so it must be switched
    # off here or the baseline would already be corrected and every
    # re-run would double-correct.
    settings.setdefault("weather", {}).setdefault(
        "bias_correction", {}
    )["enabled"] = False

    data = load_all_data()

    if "is_real_measurement" in data.columns:
        real = data[data["is_real_measurement"].fillna(False)]
    else:
        real = data

    weather = OpenMeteoClient()
    predictor = HybridPredictor()

    days = sorted(set(real["timestamp"].dt.date))

    print("=" * 88)
    print("WEATHER BIAS CORRECTION - walk-forward validation")
    print("=" * 88)

    # Bias ratio for every finished day (archived-forecast based, cached)
    ratios_by_day = {}
    for day in days:
        ratios_by_day[day] = daily_bias_ratio(real, weather, day)

    known = [f"{d}: {r:.2f}" for d, r in ratios_by_day.items() if r is not None]
    print(f"daily bias ratios known for {len(known)} days")

    rows = []

    for day in days[MIN_HISTORY_DAYS:]:

        factor = bias_factor(ratios_by_day, day)

        day_actual = real[real["timestamp"].dt.date == day][
            ["timestamp", "active_power_kw"]
        ]
        if day_actual.empty:
            continue

        for run_str in RUN_TIMES:

            hour, minute = map(int, run_str.split(":"))
            run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            signals = predictor.compute_signals(data, run_time)

            if signals.get("weather_kt") is None:
                continue

            variants = {
                "baseline": signals,
                "bias_corrected": {
                    **signals,
                    "weather_kt": np.clip(
                        signals["weather_kt"] * factor, 0, 1.2
                    ),
                },
            }

            for name, sig in variants.items():

                forecast = predictor.blend_signals(sig)

                merged = forecast.merge(day_actual, on="timestamp", how="inner")
                if len(merged) < 8:
                    continue

                rows.append({
                    "date": str(day),
                    "run_time": run_str,
                    "variant": name,
                    "factor": round(factor, 3),
                    "blocks": len(merged),
                    "deviation_pct": metrics.average_percentage_deviation(
                        merged["final_forecast_kw"],
                        merged["active_power_kw"],
                        CAPACITY_KW,
                    ),
                    "mae_kw": metrics.mean_absolute_error(
                        merged["final_forecast_kw"], merged["active_power_kw"]
                    ),
                })

    table = pd.DataFrame(rows)

    out = Path("outputs/reports/weather_bias_experiment.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(out, index=False)

    per_day = (
        table.groupby(["date", "variant"])["deviation_pct"].mean().unstack()
    )
    per_day["improvement"] = per_day["baseline"] - per_day["bias_corrected"]
    per_day["factor"] = table.groupby("date")["factor"].first()

    print()
    print(per_day.round(3).to_string())

    helped = (per_day["improvement"] > 0).sum()

    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"days scored          : {len(per_day)}")
    print(f"days improved        : {helped}/{len(per_day)}")
    print(f"avg deviation before : {per_day['baseline'].mean():.3f}%")
    print(f"avg deviation after  : {per_day['bias_corrected'].mean():.3f}%")
    print(f"avg improvement      : {per_day['improvement'].mean():+.3f} points")

    recent = per_day[per_day.index >= "2026-07-19"]
    if len(recent):
        print(f"\nmonsoon days (Jul 19+): {recent['baseline'].mean():.3f}% -> "
              f"{recent['bias_corrected'].mean():.3f}%  "
              f"({recent['improvement'].mean():+.3f} pts, "
              f"{(recent['improvement'] > 0).sum()}/{len(recent)} days better)")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
