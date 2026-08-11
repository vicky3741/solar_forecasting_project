"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Does capturing Windy closer actually change what we measure?
=========================================================
windy_capture records the satellite layer at ZOOM 8 - one
pixel is ~556 m at Sirmour, so a 1280-px frame spans ~712 km
and the default 0.6 centre crop still covers ~427 km. That
framing was chosen when the consumer was a vision LLM looking
at a picture.

The consumer is now modules/vision/satellite_features.py,
which wants a measurement of the sky OVER THE PLANT. At zoom 8
a 40 km box is 72 pixels across. windy_capture_k1 records the
same layer at ZOOM 11, where that same 40 km box is 575 pixels
- 64x the pixels for the identical patch of ground.

So the question is not "is zoom 11 sharper", which is
arithmetic. It is whether the FEATURES move: whether thick /
thin / clear cloud, entropy and the flow structure come out
differently when the plant's own sky fills the frame instead
of being 10% of it.

WHAT THIS DOES
--------------
Pairs each k1 clip with the embed clip captured nearest in
time on the same day, extracts features from BOTH THROUGH THE
SAME EXTRACTOR AT THE SAME GROUND BOX (--roi-km, passed with
each clip's own zoom so the box means the same thing), and
reports the difference per feature.

Measuring the same ground box is the whole point. Comparing a
0.6 crop of a 712 km frame against a 0.6 crop of a 111 km
frame would compare two different pieces of sky and prove
nothing.

WHAT IT CANNOT DO YET
---------------------
It cannot tell you which forecasts better. That needs k1 clips
on days with finished meter data, and k1 only started
recording on 2026-08-11 - Windy serves no archive, so these
days can only accumulate forwards. This answers the question
that comes first and costs nothing: does the input change at
all? If the features are identical, the capture change is not
worth scheduling and nothing further needs measuring.

Run:  python -m tests.compare_capture_zoom
      python -m tests.compare_capture_zoom --roi-km 25
=========================================================
"""

import argparse
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.vision.satellite_features import extract_for
from modules.vision.vision_module import VisionModule


# Only the features the prompt actually carries. The rest are computed
# and held back (see the cost report, section 2c), so a difference in
# them costs nothing and would just pad this table.
REPORTED = [
    "sat_thick_cloud_pct", "sat_thin_cloud_pct", "sat_clear_pct",
    "sat_cloud_trend_pct", "sat_entropy",
    "sat_north_cloud_pct", "sat_south_cloud_pct",
    "sat_west_cloud_pct", "sat_east_cloud_pct",
    "sat_flow_divergence", "sat_flow_vorticity", "sat_flow_kinetic_energy",
]


def clips_in(folder):
    """{timestamp: path} for every parseable clip in a folder."""

    folder = Path(folder)

    if not folder.exists():
        return {}

    found = {}

    for path in folder.iterdir():

        if path.suffix.lower() not in (".webm", ".mp4"):
            continue

        stamp = VisionModule.parse_video_time(path.name)

        if stamp is not None:
            found[pd.Timestamp(stamp)] = path

    return found


def pair_up(k1_clips, embed_clips, tolerance_minutes):
    """
    Each k1 clip with the embed clip nearest it in time on the same day.

    Nearest rather than "the one before": both captures fire from the
    same scheduler tick seconds apart, so the match is unambiguous, and
    a pair that is minutes apart is comparing two different skies and
    is dropped rather than reported.
    """

    pairs = []

    for stamp, k1_path in sorted(k1_clips.items()):

        same_day = [
            other for other in embed_clips
            if other.date() == stamp.date()
        ]

        if not same_day:
            continue

        nearest = min(same_day, key=lambda other: abs(other - stamp))

        gap = abs((nearest - stamp).total_seconds()) / 60.0

        if gap > tolerance_minutes:
            continue

        pairs.append({
            "run": stamp,
            "gap_minutes": round(gap, 1),
            "k1": k1_path,
            "embed": embed_clips[nearest],
        })

    return pairs


def main():

    parser = argparse.ArgumentParser(
        description="Compare zoom-8 and zoom-11 satellite captures"
    )
    parser.add_argument(
        "--roi-km", type=float, default=40.0,
        help="ground box measured in BOTH clips (default 40). This is "
             "what puts the two zooms on one footing."
    )
    parser.add_argument(
        "--tolerance-minutes", type=float, default=15.0,
        help="how far apart a pair may be captured and still be the "
             "same sky (default 15)"
    )
    parser.add_argument(
        "--out", default="outputs/reports/capture_zoom_comparison.csv"
    )

    args = parser.parse_args()

    plant = settings["plant"]
    latitude = plant["latitude"]

    embed_dir = settings.get("windy_capture", {}).get(
        "video_dir", "data/windy/new_videos"
    )
    k1_dir = settings.get("windy_capture_k1", {}).get(
        "video_dir", "data/windy/k1_videos"
    )

    embed_zoom = settings.get("windy_capture", {}).get("zoom", 8)
    k1_zoom = settings.get("windy_capture_k1", {}).get("zoom", 11)

    pairs = pair_up(
        clips_in(k1_dir), clips_in(embed_dir), args.tolerance_minutes
    )

    print("=" * 78)
    print("CAPTURE ZOOM COMPARISON")
    print(f"  embed : {embed_dir}  (zoom {embed_zoom})")
    print(f"  k1    : {k1_dir}  (zoom {k1_zoom})")
    print(f"  both measured over the same {args.roi_km:.0f} km ground box")
    print("=" * 78)

    if not pairs:
        raise SystemExit(
            f"\nNo paired clips yet.\n"
            f"  k1 clips    : {len(clips_in(k1_dir))}\n"
            f"  embed clips : {len(clips_in(embed_dir))}\n\n"
            "windy_capture_k1 was switched on 2026-08-11 and Windy has "
            "no archive, so pairs can only accumulate forwards. Let the "
            "scheduler run a day and try again."
        )

    rows = []

    for pair in pairs:

        measured = {}

        for source, path, zoom in (
            ("embed", pair["embed"], embed_zoom),
            ("k1", pair["k1"], k1_zoom),
        ):
            measured[source] = extract_for(
                path, roi_km=args.roi_km, zoom=zoom, latitude=latitude
            )

        if measured["embed"] is None or measured["k1"] is None:
            print(f"  {pair['run']:%Y-%m-%d %H:%M}  unreadable clip, skipped")
            continue

        row = {
            "run": pair["run"],
            "gap_minutes": pair["gap_minutes"],
            "embed_clip": Path(pair["embed"]).name,
            "k1_clip": Path(pair["k1"]).name,
        }

        for feature in REPORTED:

            low = measured["embed"].get(feature)
            high = measured["k1"].get(feature)

            row[f"{feature}__embed"] = low
            row[f"{feature}__k1"] = high

            if low is not None and high is not None:
                row[f"{feature}__delta"] = high - low

        rows.append(row)

        print(f"  {pair['run']:%Y-%m-%d %H:%M}  "
              f"thick {measured['embed']['sat_thick_cloud_pct']:5.1f}% -> "
              f"{measured['k1']['sat_thick_cloud_pct']:5.1f}%   "
              f"entropy {measured['embed']['sat_entropy']:.2f} -> "
              f"{measured['k1']['sat_entropy']:.2f}")

    if not rows:
        raise SystemExit("Every pair was unreadable.")

    frame = pd.DataFrame(rows)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print("\n" + "-" * 78)
    print(f"{'feature':<28} {'embed':>10} {'k1':>10} {'mean d':>10} "
          f"{'max |d|':>10}")
    print("-" * 78)

    for feature in REPORTED:

        delta = f"{feature}__delta"

        if delta not in frame.columns:
            continue

        print(f"{feature:<28} "
              f"{frame[f'{feature}__embed'].mean():>10.3f} "
              f"{frame[f'{feature}__k1'].mean():>10.3f} "
              f"{frame[delta].mean():>10.3f} "
              f"{frame[delta].abs().max():>10.3f}")

    print("-" * 78)
    print(f"{len(frame)} pair(s) over "
          f"{frame['run'].dt.date.nunique()} day(s)")

    print(
        "\nHOW TO READ THIS. A mean delta near zero on every row means "
        "the closer capture\nmeasures the same sky the wide one did, and "
        "is not worth scheduling. Large or\none-sided deltas mean the "
        "zoom-8 frame was averaging away the plant's own\nweather - in "
        "which case set satellite.source: k1 and re-score the pipeline "
        "on\ndays that have both clips AND finished meter data."
    )

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
