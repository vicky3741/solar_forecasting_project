"""
=========================================================
Solar Forecasting Project
Windy Video Browser
=========================================================
List and download the Windy clips stored in the team S3
bucket, without needing the AWS console.

  python -m tests.browse_videos                 # what days exist
  python -m tests.browse_videos 2026-07-29      # that day's clips
  python -m tests.browse_videos 2026-07-29 get  # download them

Downloads land in data/windy/downloaded/<date>/ and can be
opened in any video player.
=========================================================
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.storage.s3_client import S3Storage

VIDEO_EXTENSIONS = (".webm", ".mp4")
DOWNLOAD_ROOT = Path("data/windy/downloaded")


def main():

    s3 = S3Storage()

    day = sys.argv[1] if len(sys.argv) > 1 else None
    download = len(sys.argv) > 2 and sys.argv[2].lower() in ("get", "download")

    print(f"bucket : {s3.bucket}")
    print(f"region : {s3.region}")
    print(f"folder : {s3.video_prefix}/")
    print()

    prefix = f"{s3.video_prefix}/{day}" if day else s3.video_prefix
    objects = [
        o for o in s3.list_objects_meta(prefix)
        if o["Key"].lower().endswith(VIDEO_EXTENSIONS)
    ]

    if not objects:
        print("no clips found there")
        return

    if not day:
        # summary by day
        by_day = defaultdict(lambda: {"ours": 0, "other": 0, "bytes": 0})
        for obj in objects:
            name = obj["Key"].split("/")[-1]
            date = obj["Key"].split("/")[-2]
            entry = by_day[date]
            entry["ours" if name.startswith("SIRMOUR_") else "other"] += 1
            entry["bytes"] += obj["Size"]

        print(f"{'day':<12} {'ours':>5} {'others':>7} {'size':>10}")
        print("-" * 38)
        for date in sorted(by_day):
            e = by_day[date]
            print(f"{date:<12} {e['ours']:>5} {e['other']:>7} "
                  f"{e['bytes']/1e6:>8.0f} MB")

        total = sum(o["Size"] for o in objects)
        print()
        print(f"{len(objects)} clips, {total/1e6:.0f} MB total")
        print()
        print("For one day:      python -m tests.browse_videos 2026-07-29")
        print("To download it:   python -m tests.browse_videos 2026-07-29 get")
        return

    print(f"{'time':<10} {'size':>9}  file")
    print("-" * 64)

    for obj in sorted(objects, key=lambda o: o["Key"]):
        name = obj["Key"].split("/")[-1]
        when = s3.parse_video_time(obj["Key"])
        stamp = when.strftime("%H:%M:%S") if when else "?"
        who = "ours " if name.startswith("SIRMOUR_") else "other"
        print(f"{stamp:<10} {obj['Size']/1e6:>7.1f} MB  [{who}] {name}")

    if not download:
        print()
        print(f"To download these: python -m tests.browse_videos {day} get")
        return

    folder = DOWNLOAD_ROOT / day
    folder.mkdir(parents=True, exist_ok=True)

    print()
    for obj in sorted(objects, key=lambda o: o["Key"]):
        name = obj["Key"].split("/")[-1]
        target = folder / name
        if target.exists() and target.stat().st_size == obj["Size"]:
            print(f"  already have {name}")
            continue
        s3.download(obj["Key"], target)

    print()
    print(f"saved to: {folder.resolve()}")
    print("Open that folder and double-click any clip to watch it.")


if __name__ == "__main__":
    main()
