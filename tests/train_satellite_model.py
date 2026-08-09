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


def walk_forward(frame, features, min_train_days=4, params=None,
                 target="residual"):
    """
    Train on every earlier day, predict the next. Returns per-day
    results and the out-of-sample predictions.

    target="kt"       : learn the clear-sky index directly.
    target="residual" : learn (target_kt - kt_now), and add it back to
                        kt_now at predict time.

    Residual is the default and it is not a detail. When the baseline
    to beat is persistence, predicting kt directly makes the model
    spend its capacity re-deriving persistence from pixels - and any
    error in that re-derivation is subtracted straight from the
    baseline it is trying to beat. Predicting the residual hands it
    persistence for free and asks the only question the satellite can
    actually answer: how is the sky about to CHANGE?

    Measured on 2026-08-09 (21 days, walk-forward): direct kt scored
    11.34% against persistence 11.26% - i.e. it gave back everything
    it learned.
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

        # Persistence uses the same kt_now the model was given, so the
        # comparison is like for like - both know what the meter last
        # said, and only the model also sees the sky.
        kt_now = test["kt_now"].to_numpy(dtype=float)
        kt_now = np.where(np.isnan(kt_now), 1.0, kt_now)

        # kt_climate is the recent average sky, taken from the TRAINING
        # days only so it carries no lookahead. Defined here because
        # both the damped baseline and the residual_damped target need
        # it before the model is fitted.
        kt_climate = float(train["target_kt"].mean())

        # The damped curve for the TRAINING rows too, so a model can be
        # asked to improve on it rather than on flat persistence. Both
        # sides use the same kt_climate, which is in-sample for the
        # training rows and strictly out-of-sample for the test day -
        # and the test day is the only thing scored.
        train_weight = np.exp(-train["horizon_min"].to_numpy(dtype=float) / 120.0)
        train_damped = (
            train["kt_now"].to_numpy(dtype=float) * train_weight
            + kt_climate * (1 - train_weight)
        )

        horizon = test["horizon_min"].to_numpy(dtype=float)
        weight = np.exp(-horizon / 120.0)
        damped_kt = kt_now * weight + kt_climate * (1 - weight)

        if target == "residual":
            train_label = train["target_kt"] - train["kt_now"]
        elif target == "residual_damped":
            train_label = train["target_kt"].to_numpy(dtype=float) - train_damped
        else:
            train_label = train["target_kt"]

        model = lgb.train(
            params,
            lgb.Dataset(train[features], label=train_label),
            num_boost_round=250,
        )

        raw = model.predict(test[features])

        if target == "residual":
            predicted_kt = np.clip(kt_now + raw, 0.0, 1.2)
        elif target == "residual_damped":
            predicted_kt = np.clip(damped_kt + raw, 0.0, 1.2)
        else:
            predicted_kt = np.clip(raw, 0.0, 1.2)

        clearsky = test["clearsky_mw"].to_numpy()
        actual = test["actual_mw"].to_numpy()

        model_mw = np.clip(predicted_kt * clearsky, 0, CAPACITY_MW)

        persistence_mw = np.clip(kt_now * clearsky, 0, CAPACITY_MW)

        # DAMPED PERSISTENCE - the baseline that decides whether the
        # satellite is doing anything. Decay kt_now toward the recent
        # average sky as the horizon grows:
        #
        #     kt = kt_now * exp(-h/tau) + kt_climate * (1 - exp(-h/tau))
        #
        # It uses no video, no pixels, no model - only the meter and a
        # clock. Plain persistence degrades badly over hours, so ANY
        # method that drifts toward average will beat it far out. If
        # this baseline matches the LightGBM model, then the model's
        # long-horizon win is mean reversion rather than sky-reading,
        # and the satellite features are decoration.
        #
        # MEASURED 2026-08-09: it does not merely match - it WINS.
        # 9.80% against the best video model's 10.69% (35 km box).
        damped_mw = np.clip(damped_kt * clearsky, 0, CAPACITY_MW)

        rows.append({
            "day": day,
            "train_days": index,
            "rows": len(test),
            "model_pct": deviation_pct(model_mw, actual),
            "persistence_pct": deviation_pct(persistence_mw, actual),
            "damped_pct": deviation_pct(damped_mw, actual),
            "clearsky_pct": deviation_pct(clearsky, actual),
        })

        result = test[["captured_at", "target_time", "horizon_min"]].copy()
        result["predicted_kt"] = predicted_kt
        result["target_kt"] = test["target_kt"].to_numpy()
        result["model_mw"] = model_mw
        result["persistence_mw"] = persistence_mw
        result["damped_mw"] = damped_mw
        result["actual_mw"] = actual
        predictions.append(result)

    return pd.DataFrame(rows), (
        pd.concat(predictions, ignore_index=True) if predictions
        else pd.DataFrame()
    )


def by_horizon(predictions):
    """
    Model AND persistence error by how far ahead the prediction was.

    This is the test that matters for a satellite clip. A picture of
    the sky should beat persistence in the next half hour, where cloud
    that is visibly arriving has not arrived yet, and lose to it hours
    out, where the picture is stale. A flat curve means the features
    are not carrying near-term information at all.

    Both series are shown because "model error rises with horizon" is
    not evidence on its own - EVERY forecast gets worse with horizon.
    The question is whether the GAP to persistence changes.
    """

    if predictions.empty:
        return pd.DataFrame()

    predictions = predictions.copy()

    predictions["model_err"] = (
        predictions["model_mw"] - predictions["actual_mw"]
    ).abs()
    predictions["persist_err"] = (
        predictions["persistence_mw"] - predictions["actual_mw"]
    ).abs()

    grouped = predictions.groupby("horizon_min").agg(
        rows=("model_err", "size"),
        model_mw=("model_err", "mean"),
        persistence_mw=("persist_err", "mean"),
    ).reset_index()

    grouped["gain_mw"] = grouped["persistence_mw"] - grouped["model_mw"]

    return grouped


def by_horizon_band(predictions, capacity_mw):
    """
    The same comparison collapsed into bands, as % of capacity, so it
    can be read against every other number in this project.
    """

    if predictions.empty:
        return pd.DataFrame()

    predictions = predictions.copy()

    bands = [(0, 60, "<= 1h"), (60, 120, "1-2h"),
             (120, 240, "2-4h"), (0, 240, "ALL")]

    rows = []

    for low, high, label in bands:

        window = predictions[
            (predictions["horizon_min"] > (low if label != "<= 1h" else -1))
            & (predictions["horizon_min"] <= high)
        ]

        if window.empty:
            continue

        def error(column):
            return np.mean(
                np.abs(window[column] - window["actual_mw"])
            ) / capacity_mw * 100

        model = error("model_mw")
        damped = error("damped_mw")

        rows.append({
            "band": label,
            "rows": len(window),
            "model_pct": model,
            "persistence_pct": error("persistence_mw"),
            "damped_pct": damped,
            # Against the damped baseline, not plain persistence. This
            # is the number that says whether the VIDEO earned its
            # place: damped uses the meter and a clock and nothing else.
            "video_gain_pts": damped - model,
        })

    return pd.DataFrame(rows)


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

    # Residual training needs kt_now, and a row without it cannot be
    # compared against persistence either - dropping them keeps both
    # sides of the comparison on identical rows.
    before = len(frame)
    frame = frame[frame["kt_now"].notna()].reset_index(drop=True)

    features = feature_columns(frame)

    print("=" * 78)
    print("SATELLITE -> CLEAR-SKY-INDEX MODEL (LightGBM, walk-forward by day)")
    print("=" * 78)
    print(f"rows     : {len(frame)} (dropped {before - len(frame)} without kt_now)")
    print(f"days     : {frame['day'].nunique()}")
    print(f"features : {len(features)}")
    print()

    runs = {}

    for target in ("kt", "residual"):

        results, predictions = walk_forward(
            frame, features,
            min_train_days=args.min_train_days,
            target=target,
        )

        if results.empty:
            raise SystemExit(
                "Not enough days to walk forward. Collect more clips."
            )

        runs[target] = (results, predictions)

    print("Per-day deviation (% of capacity, lower is better)")
    print("-" * 78)

    table = runs["residual"][0][["day", "rows"]].copy()
    table["direct_kt"] = runs["kt"][0]["model_pct"].to_numpy()
    table["residual"] = runs["residual"][0]["model_pct"].to_numpy()
    table["persistence"] = runs["residual"][0]["persistence_pct"].to_numpy()
    table["damped"] = runs["residual"][0]["damped_pct"].to_numpy()

    print(table.to_string(index=False, float_format="%.2f"))
    print("-" * 78)
    print(f"{'MEAN':<22} "
          f"{table['direct_kt'].mean():9.2f} "
          f"{table['residual'].mean():9.2f} "
          f"{table['persistence'].mean():12.2f} "
          f"{table['damped'].mean():7.2f}")

    print()

    for target in ("kt", "residual"):

        results = runs[target][0]

        beat = int((results["model_pct"] < results["persistence_pct"]).sum())
        gain = results["persistence_pct"].mean() - results["model_pct"].mean()

        print(f"  {target:<9} beat persistence on {beat}/{len(results)} days "
              f"({gain:+.2f} pts)")

    results, predictions = runs["residual"]

    beat = int((results["model_pct"] < results["persistence_pct"]).sum())
    delta = results["persistence_pct"].mean() - results["model_pct"].mean()

    horizons = by_horizon(predictions)

    if not horizons.empty:
        print("\nMean absolute error by horizon, MW (residual model):")
        print(horizons.to_string(index=False, float_format="%.3f"))

    bands = by_horizon_band(predictions, CAPACITY_MW)

    if not bands.empty:
        print("\nBy horizon band (% of capacity):")
        print(bands.to_string(index=False, float_format="%.2f"))

    # Always shown, win or lose. When the model does NOT help, which
    # features it leaned on is the diagnosis - if kt_now dominates and
    # the sat_* features contribute almost nothing, the satellite is
    # not carrying signal, rather than the model failing to use it.
    diagnostic = lgb.train(
        {
            "objective": "regression", "metric": "l1",
            "learning_rate": 0.05, "num_leaves": 15,
            "min_data_in_leaf": 40, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "verbose": -1, "num_threads": 2,
        },
        lgb.Dataset(frame[features], label=frame["target_kt"] - frame["kt_now"]),
        num_boost_round=250,
    )

    importance = pd.DataFrame({
        "feature": features,
        "gain": diagnostic.feature_importance("gain"),
    }).sort_values("gain", ascending=False)

    importance["share_pct"] = (
        importance["gain"] / importance["gain"].sum() * 100
    )

    print("\nWhat the residual model leans on (gain share):")
    print(importance.head(12).to_string(index=False, float_format="%.1f"))

    sat_share = importance.loc[
        importance["feature"].str.startswith("sat_"), "share_pct"
    ].sum()

    print(f"\n  satellite features together: {sat_share:.1f}% of gain")

    if delta > 0 and beat > len(results) / 2:
        print("\nVERDICT: the satellite features are adding something. Retrain "
              "on all days\n         and wire it in behind a config switch, "
              "then re-score.")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.save_model(str(MODEL_PATH))

        print(f"\n         Saved: {MODEL_PATH} (residual model, trained on all days)")

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
