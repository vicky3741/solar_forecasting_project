"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Does measuring a SMALLER patch of sky help?
=========================================================
Scores the satellite model at several ground-box sizes and
reports where each one stands against persistence.

THE HYPOTHESIS
--------------
We record Windy at zoom 8. At Sirmour's latitude that is
~556 m per pixel, so a 1280-pixel frame spans ~712 km, and
the 60% centre crop the first features used covered ~427 km.

Cloud 200 km away does not shade this plant in the next 15
minutes. So a 427 km average buries the local sky in weather
that is irrelevant at the horizons we actually schedule -
which is exactly the shape of the failure measured on
2026-08-09: the model LOST to persistence at <= 1h (10.00%
vs 9.18%) and only beat it at 2-4h, the opposite of how a
picture of the sky should behave.

Kushal's Windy-Project-3 records at ZOOM_LEVEL = 11 - eight
times closer, ~69.5 m per pixel, a ~111 km frame - and crops
a box around the plant out of that.

We cannot re-record the 222 clips already in the bucket at a
different zoom. But we CAN crop a smaller box out of them,
which tests the same idea on the data we have: if a tighter
box lifts the <= 1h band, the scale really was the problem
and re-recording at zoom 11 is worth doing. If it does not,
zoom is not the answer and the video route is finished.

Run:  python -m tests.sweep_roi
=========================================================
"""

import argparse
from pathlib import Path

import pandas as pd

from config.config import settings
from tests.train_satellite_model import (
    by_horizon_band, feature_columns, walk_forward
)


CAPACITY_MW = settings["plant"]["capacity_mw"]


def score(path, min_train_days=4):

    frame = pd.read_csv(path, parse_dates=["captured_at", "target_time"])
    frame = frame[frame["kt_now"].notna()].reset_index(drop=True)

    if frame.empty:
        return None

    features = feature_columns(frame)

    results, predictions = walk_forward(
        frame, features, min_train_days=min_train_days, target="residual"
    )

    if results.empty or predictions.empty:
        return None

    bands = by_horizon_band(predictions, CAPACITY_MW).set_index("band")

    row = {
        "rows": len(frame),
        "days": int(frame["day"].nunique()),
        "model_all": results["model_pct"].mean(),
        "persistence_all": results["persistence_pct"].mean(),
        "damped_all": results["damped_pct"].mean(),
    }

    for band in ("<= 1h", "1-2h", "2-4h"):
        if band in bands.index:
            row[f"gain_{band}"] = (
                bands.loc[band, "persistence_pct"]
                - bands.loc[band, "model_pct"]
            )

    return row


def main():

    parser = argparse.ArgumentParser(
        description="Sweep the measured ground-box size"
    )
    parser.add_argument("--min-train-days", type=int, default=4)

    args = parser.parse_args()

    folder = Path("data/windy")

    datasets = []

    default = folder / "satellite_dataset.csv"
    if default.exists():
        datasets.append(("default (~427 km)", default))

    for path in sorted(
        folder.glob("satellite_dataset_*km.csv"),
        key=lambda p: int(p.stem.split("_")[-1].replace("km", "")),
    ):
        km = path.stem.split("_")[-1]
        datasets.append((f"{km} box", path))

    if not datasets:
        raise SystemExit(
            "No datasets found. Build them first:\n"
            "  python -m tests.build_satellite_dataset --roi-km 35,70,140,280"
        )

    rows = []

    for label, path in datasets:

        result = score(path, args.min_train_days)

        if result is None:
            print(f"{label}: no usable rows - skipped")
            continue

        result["roi"] = label
        rows.append(result)

    if not rows:
        raise SystemExit("Nothing could be scored.")

    table = pd.DataFrame(rows)

    columns = ["roi", "rows", "days", "model_all", "persistence_all",
               "damped_all", "gain_<= 1h", "gain_1-2h", "gain_2-4h"]

    columns = [c for c in columns if c in table.columns]

    print("=" * 88)
    print("DOES A SMALLER PATCH OF SKY HELP?")
    print("  *_all      = deviation, % of capacity, lower is better")
    print("  gain_*     = persistence minus model, in points. POSITIVE = the")
    print("               video helped in that horizon band.")
    print("=" * 88)
    print(table[columns].to_string(index=False, float_format="%.2f"))
    print("=" * 88)

    if "gain_<= 1h" in table.columns:

        best = table.loc[table["gain_<= 1h"].idxmax()]

        print(f"\nBest near-term ROI: {best['roi']} "
              f"({best['gain_<= 1h']:+.2f} pts vs persistence at <= 1h)")

        if best["gain_<= 1h"] > 0:
            print(
                "\nThe scale hypothesis HOLDS - a tighter box carries "
                "near-term signal the\nwide crop buried. Re-recording at "
                "zoom 11 should help further, since\ncropping a zoom-8 "
                "frame this small leaves very few pixels to measure."
            )
        else:
            print(
                "\nThe scale hypothesis FAILS - no box size beats "
                "persistence in the next\nhour. Zoom is not what is wrong, "
                "and the satellite-video route is done."
            )

    output = Path("outputs/reports/roi_sweep.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
