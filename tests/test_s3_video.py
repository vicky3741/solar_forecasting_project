"""
=========================================================
Solar Forecasting Project
S3 Windy Video Fetch + Vision Test
=========================================================
Verifies the live Windy video feed end to end:
  1. Asks S3 for the latest same-day video at/before a
     given run time (the one-video rule).
  2. Downloads it to the local cache.
  3. Runs the Gemini vision analysis on it.
=========================================================
"""

import pandas as pd

from modules.storage.s3_client import S3Storage
from modules.vision.vision_module import VisionModule
from modules.fusion.fusion import FeatureFusion


def main():

    s3 = S3Storage()
    vision = VisionModule()
    fusion = FeatureFusion()

    run_time = pd.Timestamp("2026-07-21 12:45:00")

    print("=" * 60)
    print(f"Latest same-day Windy video at/before {run_time}")
    print("=" * 60)

    key = s3.latest_video_before(run_time)
    print("S3 key:", key)

    if key is None:
        print("No video found for that day.")
        return

    path = s3.fetch_latest_video(run_time)
    print("Downloaded to:", path)

    print("\nRunning Gemini vision on it...")
    result = vision.analyze_video(path, "outputs/extracted_frames")

    features = fusion.prepare_vision_features(result)

    import json
    print("\nVision features:")
    print(json.dumps(features, indent=2))

    # Show the resulting per-block adjustment magnitude
    import numpy as np
    horizons = np.array([15, 60, 120, 240])
    adj = fusion.trend_adjustment_profile(features, horizons)
    print("\nVision adjustment by horizon (min -> factor):")
    for h, a in zip(horizons, adj):
        print(f"  +{h:4d} min : {a:+.4f}")


if __name__ == "__main__":
    main()
