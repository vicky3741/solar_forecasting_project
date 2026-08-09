"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Build the satellite -> generation training set
=========================================================
Turns every Windy satellite clip in the bucket into training
rows for a model that predicts CLOUD ATTENUATION AHEAD.

This is Team 1's idea (OpenCV features -> gradient boosting)
with the part that was missing: actual generation to learn
from. github.com/Kushal70-51/Windy-Project has the model
wired up, but train_model.py refuses below 20 matched rows
and the shared features log has an entirely empty
"Actual Generation (MW)" column - so it has always run its
untrained fallback, which is why the forecast in that log
reduces exactly to 1.9918 x sin(solar_elevation).

WHAT ONE ROW IS
---------------
For a clip captured at time T and a horizon h:

    features : OpenCV features of the clip (thin/thick cloud,
               entropy, flow divergence/vorticity/energy,
               quadrant cloud fractions - see
               modules/vision/satellite_features.py)
             + h itself, and the solar geometry at T+h
    target   : the clear-sky index actually measured at T+h

So the model learns "given the sky looks like THIS now, how
much of clear sky will actually arrive h minutes from now".

The target is kt, not MW, for the same reason the LLM step
returns kt: kt has no daily shape of its own, so the model
never has to learn about sunrise, and one model transfers
across plants. MW comes back by multiplying the pvlib
clear-sky curve, which is exact.

NO LOOKAHEAD
------------
Every row's features come strictly from a clip captured at or
before T, and the target strictly from meter data at T+h. The
train/test split is BY DAY and forward in time (see
tests/train_satellite_model.py), never a random shuffle -
shuffling 15-minute blocks would put 14:15 in train and 14:30
in test on the same afternoon, which leaks the answer and
produces a wonderful score that means nothing.

Run:  python -m tests.build_satellite_dataset
      python -m tests.build_satellite_dataset --from 2026-07-27 --to 2026-08-07
=========================================================
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.preprocessing.windy_features import load_meter_history
from modules.storage.s3_client import S3Storage
from modules.vision.satellite_features import SatelliteFeatureExtractor
from modules.vision.vision_module import VisionModule
from utils.logger import get_logger


TIMEZONE = settings["plant"]["timezone"]

# Horizons the model is asked about, in minutes. 15 minutes to 4 hours
# covers what a run actually has to schedule: the 06:45 run reaches
# 19:00, but a single satellite clip says nothing useful 12 hours out,
# and padding the training set with rows nobody can predict just
# teaches the model to shrug.
HORIZONS = list(range(15, 241, 15))

# Cache of per-clip OpenCV features, so re-running is cheap and a
# failed run resumes instead of re-decoding every video.
FEATURE_CACHE = Path("data/windy/sat_feature_cache")


class DatasetBuilder:

    def __init__(self, start=None, end=None, roi_km=None, with_k1=False):

        self.logger = get_logger()

        self.storage = S3Storage()
        self.clearsky = ClearSkyModel()

        # Also compute Team 1's method (modules/vision/windy_motion.py),
        # a verbatim port, so their features and ours ride in the same
        # table and get scored by the same walk-forward. Cached under
        # its own key because it is a different measurement.
        self.with_k1 = with_k1
        self.k1_cache = FEATURE_CACHE.parent / "sat_feature_cache_k1"

        if with_k1:
            self.k1_cache.mkdir(parents=True, exist_ok=True)

        # One or several ROI sizes. Several means a sweep, and the
        # sweep shares a SINGLE download pass over the bucket - the
        # clips are the slow part (~20 minutes for 222), the OpenCV is
        # seconds, so re-downloading once per ROI size would waste an
        # hour to compute the same thing four ways.
        self.roi_list = roi_km if isinstance(roi_km, (list, tuple)) else [roi_km]

        self.extractors = {
            km: SatelliteFeatureExtractor(roi_km=km) for km in self.roi_list
        }

        # Cache per ROI size. Features extracted at a 400 km crop are a
        # different measurement from the same clip at 40 km, so they
        # must not share a cache key - reusing one would silently
        # compare a sweep against itself.
        self.cache_dirs = {}

        for km in self.roi_list:

            path = (
                FEATURE_CACHE if km is None
                else FEATURE_CACHE.parent / f"sat_feature_cache_{int(km)}km"
            )
            path.mkdir(parents=True, exist_ok=True)
            self.cache_dirs[km] = path

        self.start = pd.Timestamp(start).date() if start else None
        self.end = pd.Timestamp(end).date() if end else None

    # --------------------------------------------------

    def list_clips(self):
        """
        Every satellite clip key in the bucket within the date range,
        as (captured_at, key) sorted by time.
        """

        paginator = self.storage.client.get_paginator("list_objects_v2")

        clips = []

        for page in paginator.paginate(
            Bucket=self.storage.bucket, Prefix=self.storage.video_prefix
        ):
            for obj in page.get("Contents", []):

                key = obj["Key"]

                if not key.lower().endswith((".webm", ".mp4")):
                    continue

                captured = VisionModule.parse_video_time(Path(key).name)

                if captured is None:
                    continue

                captured = pd.Timestamp(captured)

                if self.start and captured.date() < self.start:
                    continue

                if self.end and captured.date() > self.end:
                    continue

                clips.append((captured, key))

        clips.sort()

        return clips

    # --------------------------------------------------

    def features_for(self, captured, key):
        """
        OpenCV features for one clip at every configured ROI size, as
        {roi_km: features}. Downloads the clip only if at least one ROI
        size is not already cached, and only once for all of them.
        Sizes that fail extraction are simply absent from the result.
        """

        stem = Path(key).stem

        cached = {}
        missing = []

        for km in self.roi_list:

            cache_file = self.cache_dirs[km] / f"{stem}.json"

            if cache_file.exists():
                try:
                    cached[km] = json.loads(
                        cache_file.read_text(encoding="utf-8")
                    )
                    continue
                except Exception:
                    pass   # corrupt cache entry - re-extract below

            missing.append(km)

        k1 = None
        k1_file = self.k1_cache / f"{stem}.json"
        k1_missing = self.with_k1

        if self.with_k1 and k1_file.exists():
            try:
                k1 = json.loads(k1_file.read_text(encoding="utf-8"))
                k1_missing = False
            except Exception:
                pass

        if not missing and not k1_missing:
            return self._merge_k1(cached, k1)

        local = Path(self.storage.video_cache) / Path(key).name
        local.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not local.exists():
                self.storage.client.download_file(
                    self.storage.bucket, key, str(local)
                )

            for km in missing:

                try:
                    features = self.extractors[km].extract(local)
                except Exception as error:
                    self.logger.warning(
                        f"Skipping {Path(key).name} at roi={km}: {error}"
                    )
                    continue

                (self.cache_dirs[km] / f"{stem}.json").write_text(
                    json.dumps(features, indent=2), encoding="utf-8"
                )

                cached[km] = features

            if k1_missing:

                from modules.vision.windy_motion import features_for_model

                try:
                    k1 = features_for_model(local)
                except Exception as error:
                    self.logger.warning(
                        f"Team 1 motion failed on {Path(key).name}: {error}"
                    )
                    k1 = None

                if k1 is not None:
                    k1_file.write_text(
                        json.dumps(k1, indent=2), encoding="utf-8"
                    )

        except Exception as error:
            self.logger.warning(f"Skipping {Path(key).name}: {error}")

        finally:
            # The clips are the bulky part (a few MB each, 236 of them).
            # The extracted numbers are what we need, so the video goes
            # once it has been read - otherwise a full build leaves ~1 GB
            # behind on a box that has 6.7 GB total.
            if local.exists():
                local.unlink(missing_ok=True)

        return self._merge_k1(cached, k1)

    # --------------------------------------------------

    @staticmethod
    def _merge_k1(per_roi, k1):
        """
        Team 1's motion features are computed from their own fixed ROI,
        not ours, so the same k1 block is attached to every ROI variant
        rather than being recomputed per crop. Attaching it lets one
        walk-forward compare "our features", "their features" and both
        together on identical rows and identical days.
        """

        if not k1:
            return per_roi

        return {
            km: {**features, **k1} for km, features in per_roi.items()
        }

    # --------------------------------------------------

    def meter_with_kt(self):
        """
        Meter history with a measured clear-sky index per timestamp.
        """

        meter = load_meter_history()

        if meter is None or meter.empty:
            raise SystemExit(
                "No meter history in data/historical - nothing to learn from."
            )

        meter = meter.copy()

        if meter["timestamp"].dt.tz is None:
            meter["timestamp"] = meter["timestamp"].dt.tz_localize(TIMEZONE)
        else:
            meter["timestamp"] = meter["timestamp"].dt.tz_convert(TIMEZONE)

        expected = self.clearsky.estimate_clearsky_generation(
            meter["timestamp"]
        )

        clearsky_mw = expected["expected_power_kw"].to_numpy() / 1000.0
        actual_mw = meter["active_power_kw"].to_numpy(dtype=float) / 1000.0

        # Only where the clear-sky curve is meaningfully above zero.
        # Near sunrise/sunset the ratio is a tiny number over a tiny
        # number, which is noise, not cloudiness.
        kt = np.divide(
            actual_mw, clearsky_mw,
            out=np.full(len(meter), np.nan),
            where=clearsky_mw > 0.05,
        )

        meter["actual_mw"] = actual_mw
        meter["clearsky_mw"] = clearsky_mw
        meter["actual_kt"] = np.clip(kt, 0, 1.2)

        columns = ["timestamp", "actual_mw", "clearsky_mw", "actual_kt"]

        if "is_real_measurement" in meter.columns:
            meter = meter[meter["is_real_measurement"].fillna(False)]

        return meter[columns].dropna(subset=["actual_kt"])

    # --------------------------------------------------

    def build(self):

        clips = self.list_clips()

        if not clips:
            raise SystemExit("No satellite clips found in the bucket.")

        self.logger.info(f"Found {len(clips)} clip(s) to process")

        meter = self.meter_with_kt().set_index("timestamp")

        rows = {km: [] for km in self.roi_list}
        used = 0

        for index, (captured, key) in enumerate(clips, start=1):

            per_roi = self.features_for(captured, key)

            if not per_roi:
                continue

            # FLOOR TO THE 15-MINUTE GRID. Capture fires at the run time
            # but takes a few seconds, so clips are stamped 14-15-30,
            # 12-45-24, 15-45-14. Meter data sits exactly on
            # :00/:15/:30/:45, so an unfloored capture time puts every
            # target at 14:30:30 and the lookup misses every single one.
            # The first build of this dataset matched 10 of 222 clips
            # for precisely this reason.
            #
            # modules/orchestrator/pipeline.py already floors its run
            # time and carries a comment about the 2026-07-26 run
            # producing zero gradeable blocks without it - same trap.
            captured_local = captured.tz_localize(TIMEZONE).floor("15min")

            # kt measured just BEFORE the clip - persistence, and the
            # single most informative feature there is. Without it the
            # model has to infer the current state from pixels alone,
            # which throws away the meter reading we already have.
            recent = meter.loc[
                (meter.index <= captured_local)
                & (meter.index > captured_local - pd.Timedelta(minutes=60))
            ]

            kt_now = float(recent["actual_kt"].tail(4).mean()) if not recent.empty \
                else np.nan

            matched = 0

            for horizon in HORIZONS:

                target_time = captured_local + pd.Timedelta(minutes=horizon)

                if target_time not in meter.index:
                    continue

                target = meter.loc[target_time]

                if target["clearsky_mw"] <= 0.05:
                    continue      # night; nothing to predict

                position = self.clearsky.location.get_solarposition(
                    pd.DatetimeIndex([target_time])
                )

                shared = {
                    "clip": Path(key).name,
                    "captured_at": captured_local,
                    "target_time": target_time,
                    "day": captured_local.date(),
                    "horizon_min": horizon,
                    "kt_now": kt_now,
                    "solar_elevation_deg": float(
                        position["apparent_elevation"].iloc[0]
                    ),
                    "clearsky_mw": float(target["clearsky_mw"]),
                    "actual_mw": float(target["actual_mw"]),
                    "target_kt": float(target["actual_kt"]),
                }

                for km, features in per_roi.items():

                    row = {
                        k: v for k, v in features.items()
                        if k not in ("sat_video", "sat_captured_at")
                    }
                    row.update(shared)
                    rows[km].append(row)

                matched += 1

            if matched:
                used += 1

            if index % 20 == 0:
                total = sum(len(v) for v in rows.values())
                self.logger.info(
                    f"  {index}/{len(clips)} clips, {total} rows so far"
                )

        if not any(rows.values()):
            raise SystemExit(
                "No clip lined up with meter data. Check that the clip dates "
                "and data/historical overlap."
            )

        print()
        print("=" * 70)
        print("SATELLITE -> GENERATION TRAINING SET")
        print("=" * 70)
        print(f"clips found  : {len(clips)}")
        print(f"clips usable : {used}")

        frames = {}

        for km in self.roi_list:

            if not rows[km]:
                continue

            frame = pd.DataFrame(rows[km])

            suffix = "" if km is None else f"_{int(km)}km"

            output = Path(f"data/windy/satellite_dataset{suffix}.csv")
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)

            label = "default crop" if km is None else f"{int(km)} km box"

            print(f"\n  {label}: {len(frame)} rows, "
                  f"{frame['day'].nunique()} days "
                  f"({frame['day'].min()} to {frame['day'].max()})")
            print(f"    target_kt mean {frame['target_kt'].mean():.3f}, "
                  f"sd {frame['target_kt'].std():.3f}")
            print(f"    saved: {output}")

            frames[km] = frame

        return frames


def main():

    parser = argparse.ArgumentParser(
        description="Build the satellite -> generation training set"
    )
    parser.add_argument("--from", dest="start", default=None)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument(
        "--roi-km", default=None,
        help="ground box measured, in km. Comma-separate for a sweep "
             "(e.g. 35,70,140,280); omitted = the old 60%% centre crop"
    )

    parser.add_argument(
        "--with-k1", action="store_true",
        help="also compute Team 1's motion features (verbatim port) "
             "and attach them to every row"
    )

    args = parser.parse_args()

    roi = None

    if args.roi_km:
        roi = [float(value) for value in args.roi_km.split(",")]

    DatasetBuilder(
        args.start, args.end, roi_km=roi, with_k1=args.with_k1
    ).build()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
