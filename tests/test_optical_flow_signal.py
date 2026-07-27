"""
=========================================================
Solar Forecasting Project
Optical-Flow Signal Check
=========================================================
Before wiring the optical-flow numbers into the forecast,
ask the cheaper question first: do they carry any
predictive information at all?

Every Windy capture already writes a sidecar beside the
clip - cloud cover now, how that cover changed across the
clip, motion direction and a confidence figure. They are
free, instant, need no API and never fail on a busy quota,
but nothing in the forecast has ever read them.

The test: at each scheduling time, compare what the sidecar
said with what the sky ACTUALLY did over the following two
hours, measured as the change in clear-sky index (kt) from
the plant's own sensor.

  cloud_cover_trend > 0  means clouding over  -> kt should FALL
  cloud_cover_trend < 0  means clearing       -> kt should RISE

So a useful signal shows a NEGATIVE correlation with the kt
change. Near zero means the numbers are noise and no amount
of clever blending will rescue them.

This is deliberately a measurement, not an integration -
nothing here touches the pipeline.

Run:  python -m tests.test_optical_flow_signal
=========================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.preprocessing.preprocess import DataPreprocessor
from modules.storage.s3_client import S3Storage
from modules.vision.vision_module import VisionModule


RUN_TIMES = settings["forecast"]["run_times"]
LOOKAHEAD_HOURS = 2
TOLERANCE_SECONDS = 300


def load_sidecars():
    """
    {datetime: sidecar dict}, from local folders AND the bucket.

    Reading only the local folders under-samples badly: they hold
    whatever earlier experiments happened to download, while the
    bucket has every sidecar ever captured. The files are a few
    hundred bytes each, so fetching the missing ones is cheap.
    """

    s3 = S3Storage()
    out = {}

    def remember(when, payload):
        if when is not None and "cloud_cover_fraction" in payload:
            out[pd.Timestamp(when)] = payload

    for folder in ("data/windy/new_videos", "data/windy/s3_cache"):
        for path in Path(folder).glob("*.json"):
            when = (VisionModule.parse_video_time(path.name)
                    or s3.parse_video_time(path.name))
            if when is None:
                continue
            try:
                remember(when, json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue

    local_count = len(out)

    cache = Path("data/windy/flow_cache")
    cache.mkdir(parents=True, exist_ok=True)

    try:
        for obj in s3.list_objects_meta(s3.video_prefix):

            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            when = s3.parse_video_time(key) or VisionModule.parse_video_time(key)
            if when is None or pd.Timestamp(when) in out:
                continue

            local = cache / key.split("/")[-1]
            try:
                if not local.exists():
                    s3.download(key, local)
                remember(when, json.loads(local.read_text(encoding="utf-8")))
            except Exception:
                continue

    except Exception as error:
        print(f"  (bucket sidecars unavailable: {error})")

    print(f"  sidecars: {local_count} local + {len(out) - local_count} from the bucket")

    return out


def main():

    pre = DataPreprocessor()
    frames = [
        pre.preprocess(file_path=p, required_columns=["TimeStamp"],
                       timestamp_column="TimeStamp")
        for p in sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))
    ]
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")

    if "is_real_measurement" in data.columns:
        data = data[data["is_real_measurement"].fillna(False)]

    clearsky = ClearSkyModel()
    with_kt = clearsky.compute_clear_sky_index(
        data, ghi_column="ghi_w_m2", timestamp_column="timestamp"
    )
    kt = with_kt[["timestamp", "clear_sky_index"]].dropna()
    kt = kt.set_index("timestamp")["clear_sky_index"]

    sidecars = load_sidecars()

    print("=" * 88)
    print("OPTICAL-FLOW SIGNAL CHECK - does the sidecar predict what the sky did?")
    print("=" * 88)
    print(f"sidecars found: {len(sidecars)}   lookahead: {LOOKAHEAD_HOURS}h")
    print()

    rows = []

    for run_str in RUN_TIMES:
        hour, minute = map(int, run_str.split(":"))

        for day in sorted({t.date() for t in sidecars}):

            target = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

            # nearest sidecar to this scheduling time
            best, best_gap = None, None
            for when, payload in sidecars.items():
                gap = abs((when - target).total_seconds())
                if gap <= TOLERANCE_SECONDS and (best_gap is None or gap < best_gap):
                    best, best_gap = payload, gap

            if best is None:
                continue

            # kt now (mean of the hour before) and kt later
            now_window = kt[(kt.index > target - pd.Timedelta(hours=1))
                            & (kt.index <= target)]
            later_window = kt[(kt.index > target + pd.Timedelta(hours=LOOKAHEAD_HOURS - 1))
                              & (kt.index <= target + pd.Timedelta(hours=LOOKAHEAD_HOURS))]

            if len(now_window) < 2 or len(later_window) < 2:
                continue

            kt_change = float(later_window.mean() - now_window.mean())

            rows.append({
                "date": str(day),
                "run_time": run_str,
                "cover": best.get("cloud_cover_fraction"),
                "trend": best.get("cloud_cover_trend"),
                "trustworthy": best.get("motion_trustworthy"),
                "direction": best.get("motion_direction"),
                "kt_change": round(kt_change, 4),
            })

    if not rows:
        print("No scheduling time had both a sidecar and enough meter data.")
        return

    table = pd.DataFrame(rows).dropna(subset=["trend", "kt_change"])

    out = Path("outputs/reports/optical_flow_signal.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)

    print(table.to_string(index=False))

    print()
    print("=" * 88)
    print("DOES IT PREDICT?")
    print("=" * 88)

    def report(label, subset):
        if len(subset) < 4:
            print(f"  {label:<28} too few samples ({len(subset)})")
            return
        r_trend = subset["trend"].corr(subset["kt_change"])
        r_cover = subset["cover"].corr(subset["kt_change"])
        print(f"  {label:<28} n={len(subset):<3} "
              f"trend vs kt-change r={r_trend:+.3f}   "
              f"cover vs kt-change r={r_cover:+.3f}")

    report("all readings", table)
    report("motion trustworthy only", table[table["trustworthy"] == True])
    report("afternoon runs (>=11:15)", table[table["run_time"] >= "11:15"])

    print()
    print("  A USEFUL trend signal is NEGATIVE (clouding over -> kt falls).")
    print("  Around zero means noise; positive means it points the wrong way.")

    r = table["trend"].corr(table["kt_change"])
    print()
    if r <= -0.3:
        print(f"VERDICT: usable signal (r={r:+.3f}). Worth wiring in and validating.")
    elif r <= -0.1:
        print(f"VERDICT: weak but real (r={r:+.3f}). Might help as a small nudge.")
    else:
        print(f"VERDICT: no usable signal (r={r:+.3f}). Do not build on this as-is.")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
