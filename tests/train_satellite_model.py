"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Train and honestly score the satellite -> kt model
=========================================================
LightGBM on the OpenCV satellite features, predicting the
clear-sky index at a horizon. Team 1's architecture, finally
given the actual generation it needs to learn from.

HOW IT IS SCORED, AND WHY IT IS NOT A RANDOM SPLIT
--------------------------------------------------
Walk-forward by DAY: for each test day D, train on every day
strictly before D and predict D. That is what the live system
would actually have known.

A random train/test split would be dishonest here and would
look spectacular. Rows 15 minutes apart on the same afternoon
are almost the same sky, so shuffling puts near-duplicates on
both sides and the model scores its own training data. Any
satellite/generation model reporting a high R2 from a shuffled
split has measured nothing.

THE BASELINES ARE THE POINT
---------------------------
  persistence : hold the last measured kt flat. Free, needs no
                model, no video, no API. This is the number to
                beat - the LLM path already lost to it (7.35%
                vs 3.44% on 2026-07-09).
  clear-sky   : assume kt = 1. Not a forecast, just the scale.

Deviation is mean absolute error as a percentage of plant
CAPACITY, the mentor's metric and what every other signal in
this project was tuned against.

Run:  python -m tests.train_satellite_model
=========================================================
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.config import settings


CAPACITY_MW = settings["plant"]["capacity_mw"]

DATASET = Path("data/windy/satellite_dataset.csv")
MODEL_PATH = Path("models/satellite_kt_lgbm.txt")

# Columns that identify a row rather than describe the sky.
_NOT_FEATURES = {
    "clip", "captured_at", "target_time", "day",
    "actual_mw", "target_kt", "clearsky_mw",
}


def deviation_pct(forecast_mw, actual_mw):

    return float(
        np.mean(np.abs(np.asarray(forecast_mw) - np.asarray(actual_mw)))
        / CAPACITY_MW * 100
    )


def feature_columns(frame):

    return [
        column for column in frame.columns
        if column not in _NOT_FEATURES
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def walk_forward(frame, features, min_train_days=4, params=None):
    """
    Train on every earlier day, predict the next. Returns per-day
    results and the out-of-sample predictions.
    """

    params = params or {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 15,
        # Small data (a few thousand rows from a couple of dozen days),
        # so the model is kept deliberately shallow. Left at LightGBM's
        # defaults it memorises individual afternoons and the
        # walk-forward score collapses.
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 2,
    }

    days = sorted(frame["day"].unique())

    rows = []
    predictions = []

    for index, day in enumerate(days):

        if index < min_train_days:
            continue

        train = frame[frame["day"] < day]
        test = frame[frame["day"] == day]

        if train.empty or test.empty:
            continue

        model = lgb.train(
            params,
            lgb.Dataset(train[features], label=train["target_kt"]),
            num_boost_round=250,
        )

        predicted_kt = np.clip(model.predict(test[features]), 0.0, 1.2)

        clearsky = test["clearsky_mw"].to_numpy()
        actual = test["actual_mw"].to_numpy()

        model_mw = np.clip(predicted_kt * clearsky, 0, CAPACITY_MW)

        # Persistence uses the same kt_now the model was given, so the
        # comparison is like for like - both know what the meter last
        # said, and only the model also sees the sky.
        kt_now = test["kt_now"].to_numpy(dtype=float)
        kt_now = np.where(np.isnan(kt_now), 1.0, kt_now)

        persistence_mw = np.clip(kt_now * clearsky, 0, CAPACITY_MW)

        rows.append({
            "day": day,
            "train_days": index,
            "rows": len(test),
            "model_pct": deviation_pct(model_mw, actual),
            "persistence_pct": deviation_pct(persistence_mw, actual),
            "clearsky_pct": deviation_pct(clearsky, actual),
        })

        result = test[["captured_at", "target_time", "horizon_min"]].copy()
        result["predicted_kt"] = predicted_kt
        result["target_kt"] = test["target_kt"].to_numpy()
        result["model_mw"] = model_mw
        result["actual_mw"] = actual
        predictions.append(result)

    return pd.DataFrame(rows), (
        pd.concat(predictions, ignore_index=True) if predictions
        else pd.DataFrame()
    )


def by_horizon(predictions):
    """
    Error by how far ahead the prediction was. A satellite clip should
    help most in the next hour and fade after that; if it does not,
    the model is leaning on the clock rather than the sky.
    """

    if predictions.empty:
        return pd.DataFrame()

    predictions = predictions.copy()
    predictions["abs_error_mw"] = (
        predictions["model_mw"] - predictions["actual_mw"]
    ).abs()

    return predictions.groupby("horizon_min").agg(
        rows=("abs_error_mw", "size"),
        mean_abs_error_mw=("abs_error_mw", "mean"),
    ).reset_index()


def main():

    parser = argparse.ArgumentParser(
        description="Train and walk-forward score the satellite kt model"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--min-train-days", type=int, default=4)

    args = parser.parse_args()

    path = Path(args.dataset)

    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build it first:\n"
            "  python -m tests.build_satellite_dataset"
        )

    frame = pd.read_csv(path, parse_dates=["captured_at", "target_time"])

    features = feature_columns(frame)

    print("=" * 78)
    print("SATELLITE -> CLEAR-SKY-INDEX MODEL (LightGBM, walk-forward by day)")
    print("=" * 78)
    print(f"rows     : {len(frame)}")
    print(f"days     : {frame['day'].nunique()}")
    print(f"features : {len(features)}")
    print()

    results, predictions = walk_forward(
        frame, features, min_train_days=args.min_train_days
    )

    if results.empty:
        raise SystemExit(
            "Not enough days to walk forward. Collect more clips."
        )

    print(results.to_string(index=False, float_format="%.2f"))
    print("-" * 78)
    print(f"{'MEAN':<12} {'':>12} {'':>6} "
          f"{results['model_pct'].mean():10.2f} "
          f"{results['persistence_pct'].mean():16.2f} "
          f"{results['clearsky_pct'].mean():13.2f}")

    beat = int((results["model_pct"] < results["persistence_pct"]).sum())
    delta = results["persistence_pct"].mean() - results["model_pct"].mean()

    print()
    print(f"Model beat persistence on {beat} of {len(results)} unseen days "
          f"({delta:+.2f} pts on average).")

    horizons = by_horizon(predictions)

    if not horizons.empty:
        print("\nError by horizon:")
        print(horizons.to_string(index=False, float_format="%.3f"))

    if delta > 0 and beat > len(results) / 2:
        print("\nVERDICT: the satellite features are adding something. Retrain "
              "on all days\n         and wire it in behind a config switch, "
              "then re-score.")

        model = lgb.train(
            {
                "objective": "regression", "metric": "l1",
                "learning_rate": 0.05, "num_leaves": 15,
                "min_data_in_leaf": 40, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1,
                "lambda_l2": 1.0, "verbose": -1, "num_threads": 2,
            },
            lgb.Dataset(frame[features], label=frame["target_kt"]),
            num_boost_round=250,
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))

        print(f"         Saved: {MODEL_PATH}")

        importance = pd.DataFrame({
            "feature": features,
            "gain": model.feature_importance("gain"),
        }).sort_values("gain", ascending=False).head(12)

        print("\nWhat the model actually used:")
        print(importance.to_string(index=False, float_format="%.1f"))

    else:
        print("\nVERDICT: not better than holding the last measured "
              "cloudiness flat.\n         Do NOT ship it. The satellite "
              "features are not carrying\n         information the meter "
              "reading does not already have.")

    output = Path("outputs/reports/satellite_model_walkforward.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
