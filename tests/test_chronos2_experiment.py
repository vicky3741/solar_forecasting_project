"""
=========================================================
Solar Forecasting Project
Chronos-2 Experiment (Vikrant's assigned model study)
=========================================================
Assignment: study Chronos-2 in our project context, use it,
generate outputs, and analyze its cost.

Head start: this pipeline has run Chronos-1 (amazon/
chronos-bolt-small) in production since day one, so the
study is a like-for-like upgrade test on REAL plant data,
not a toy demo. Three contenders, identical inputs:

  1. bolt        - amazon/chronos-bolt-small, the current
                   production model (univariate only).
  2. chronos2    - amazon/chronos-2, same univariate task.
  3. chronos2+wx - amazon/chronos-2 with the Open-Meteo
                   forecast supplied as a covariate. This is
                   THE new capability: Bolt physically cannot
                   accept helper series, so our pipeline has
                   to blend weather in afterwards (65% weight
                   outside the model). Chronos-2 can fuse it
                   inside the model instead.

Method (same honesty rules as every other experiment here):
  * forecast clear-sky index (kt), convert to power via the
    physics curve - identical to production;
  * inputs only up to the run time, no lookahead; weather
    uses the archived model run available at that moment;
  * scored ONLY against real measured meter blocks;
  * per-day, per-run-time, so no overlap double counting.

The covariate variant uses a shorter context (last few days)
because its weather series must align 1:1 with the target
history, and archived weather is fetched per-day. Labeled in
the output.

Run:  python -m tests.test_chronos2_experiment
=========================================================
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.preprocessing.preprocess import DataPreprocessor
from modules.weather.open_meteo import OpenMeteoClient
from modules.evaluation import metrics


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
PERFORMANCE_RATIO = settings["clearsky"]["performance_ratio"]
END_TIME = settings["forecast"]["forecast_end_time"]

RUN_TIMES = ["06:45", "12:45"]        # morning (hard) + midday (easy)
TEST_DAYS = 8                         # most recent days with real data
COVARIATE_CONTEXT_DAYS = 3            # see module docstring

BOLT_ID = settings["forecast_model"]["model"]
CHRONOS2_ID = "amazon/chronos-2"


def load_all_data():
    """Same per-day preprocessing as the orchestrator."""

    pre = DataPreprocessor()
    frames = [
        pre.preprocess(
            file_path=path,
            required_columns=["TimeStamp"],
            timestamp_column="TimeStamp",
        )
        for path in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def build_kt_series(dataframe, clearsky):
    """(timestamp, kt) for every valid daylight reading."""

    with_kt = clearsky.compute_clear_sky_index(
        dataframe, ghi_column="ghi_w_m2", timestamp_column="timestamp"
    )
    return with_kt[["timestamp", "clear_sky_index"]].dropna().reset_index(drop=True)


def horizon_for(run_time):
    """Grid-aligned 15-min blocks from run_time to the day's end."""

    end_hour, end_minute = map(int, END_TIME.split(":"))
    end = run_time.replace(hour=end_hour, minute=end_minute)
    stamps = []
    t = run_time + pd.Timedelta(minutes=15)
    while t <= end:
        stamps.append(t)
        t += pd.Timedelta(minutes=15)
    return pd.DatetimeIndex(stamps)


def kt_to_power(kt_forecast, clearsky_kw):
    return np.clip(np.asarray(kt_forecast) * clearsky_kw, 0, CAPACITY_KW)


def weather_kt_for(weather, clearsky, timestamps, as_of):
    """Open-Meteo forecasted kt at `timestamps`, as known at `as_of`."""

    ghi_forecast = weather.forecast_ghi_at(timestamps, as_of=as_of)
    if ghi_forecast is None:
        return None

    clearsky_ghi = clearsky.get_poa_irradiance(pd.DatetimeIndex(timestamps))["ghi"].to_numpy()
    kt = np.divide(
        ghi_forecast, clearsky_ghi,
        out=np.full_like(clearsky_ghi, np.nan, dtype=float),
        where=clearsky_ghi > 10,
    )
    return np.nan_to_num(np.clip(kt, 0, 1.2), nan=1.0)


def main():

    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline

    print("=" * 88)
    print("CHRONOS-2 vs CHRONOS-BOLT - assigned-model study on real Sirmour data")
    print("=" * 88)

    data = load_all_data()
    clearsky = ClearSkyModel()
    weather = OpenMeteoClient()
    kt_series = build_kt_series(data, clearsky)

    actual_columns = ["timestamp", "active_power_kw"]
    if "is_real_measurement" in data.columns:
        real = data[data["is_real_measurement"].fillna(False)]
    else:
        real = data
    actual = real[actual_columns]

    days = sorted(set(real["timestamp"].dt.date))[-TEST_DAYS:]
    print(f"test days: {days[0]} .. {days[-1]}  |  run times: {', '.join(RUN_TIMES)}")

    print("\nloading models (first Chronos-2 use downloads the weights)...")
    t0 = time.time()
    bolt = BaseChronosPipeline.from_pretrained(BOLT_ID, device_map="cpu", torch_dtype=torch.float32)
    t_bolt_load = time.time() - t0

    t0 = time.time()
    chronos2 = Chronos2Pipeline.from_pretrained(CHRONOS2_ID, device_map="cpu")
    t_c2_load = time.time() - t0

    def param_count(pipeline):
        try:
            return sum(p.numel() for p in pipeline.model.parameters())
        except Exception:
            return None

    print(f"  bolt      : {param_count(bolt) or '?':>12,} params, loaded in {t_bolt_load:.1f}s")
    print(f"  chronos-2 : {param_count(chronos2) or '?':>12,} params, loaded in {t_c2_load:.1f}s")

    rows = []
    inference_seconds = {"bolt": [], "chronos2": [], "chronos2+wx": []}

    for day in days:
        for run_str in RUN_TIMES:

            hour, minute = map(int, run_str.split(":"))
            run_time = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            history = kt_series[kt_series["timestamp"] <= run_time]
            if len(history) < 20:
                continue

            stamps = horizon_for(run_time)
            clearsky_kw = (
                clearsky.get_poa_irradiance(stamps)["poa_global"].to_numpy()
                / 1000 * CAPACITY_KW * PERFORMANCE_RATIO
            )

            target_day = actual[actual["timestamp"].isin(stamps)]
            merged_stamps = pd.DatetimeIndex(target_day["timestamp"])
            if len(merged_stamps) < 8:
                continue
            actual_kw = target_day.set_index("timestamp").loc[merged_stamps, "active_power_kw"].to_numpy()
            keep = np.isin(stamps, merged_stamps)

            context = history["clear_sky_index"].to_numpy(dtype=np.float32)
            forecasts = {}

            # --- 1. production Bolt ---
            t0 = time.time()
            _, mean = bolt.predict_quantiles(
                torch.tensor(context), prediction_length=len(stamps), quantile_levels=[0.5]
            )
            inference_seconds["bolt"].append(time.time() - t0)
            forecasts["bolt"] = np.clip(mean[0].numpy(), 0, 1.2)

            # --- 2. Chronos-2, same univariate task ---
            t0 = time.time()
            _, mean = chronos2.predict_quantiles([context], prediction_length=len(stamps))
            inference_seconds["chronos2"].append(time.time() - t0)
            forecasts["chronos2"] = np.clip(mean[0].numpy().ravel(), 0, 1.2)

            # --- 3. Chronos-2 + weather covariate (the new capability) ---
            try:
                cov_start = run_time - pd.Timedelta(days=COVARIATE_CONTEXT_DAYS)
                cov_hist = history[history["timestamp"] >= cov_start]
                past_wx = weather_kt_for(
                    weather, clearsky, cov_hist["timestamp"], as_of=run_time
                )
                future_wx = weather_kt_for(weather, clearsky, stamps, as_of=run_time)

                if past_wx is not None and future_wx is not None:
                    t0 = time.time()
                    _, mean = chronos2.predict_quantiles(
                        [{
                            "target": cov_hist["clear_sky_index"].to_numpy(dtype=np.float32),
                            "past_covariates": {"weather_kt": past_wx.astype(np.float32)},
                            "future_covariates": {"weather_kt": future_wx.astype(np.float32)},
                        }],
                        prediction_length=len(stamps),
                    )
                    inference_seconds["chronos2+wx"].append(time.time() - t0)
                    forecasts["chronos2+wx"] = np.clip(mean[0].numpy().ravel(), 0, 1.2)
            except Exception as error:
                print(f"  ({day} {run_str}: covariate variant skipped - {error})")

            for name, kt_forecast in forecasts.items():
                power = kt_to_power(kt_forecast, clearsky_kw)[keep]
                rows.append({
                    "date": str(day),
                    "run_time": run_str,
                    "model": name,
                    "blocks": int(keep.sum()),
                    "deviation_pct": round(metrics.average_percentage_deviation(
                        pd.Series(power), pd.Series(actual_kw), CAPACITY_KW), 3),
                    "mae_kw": round(metrics.mean_absolute_error(
                        pd.Series(power), pd.Series(actual_kw)), 1),
                })

    table = pd.DataFrame(rows)
    out = Path("outputs/reports/chronos2_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print()
    print("=" * 88)
    print("RESULTS - pure model vs model (no blending, no corrections)")
    print("=" * 88)
    summary = table.groupby("model").agg(
        runs=("deviation_pct", "size"),
        avg_deviation_pct=("deviation_pct", "mean"),
        avg_mae_kw=("mae_kw", "mean"),
    ).sort_values("avg_deviation_pct").round(3)
    print(summary.to_string())

    print()
    print("inference speed (avg seconds per forecast, CPU):")
    for name, times in inference_seconds.items():
        if times:
            print(f"  {name:<12}: {np.mean(times):6.2f}s over {len(times)} calls")

    print()
    print("per run-time breakdown:")
    print(
        table.groupby(["run_time", "model"])["deviation_pct"].mean().round(3)
        .unstack().to_string()
    )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
