"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Windy video motion - faithful port of Team 1's method
=========================================================
A same-to-same port of video_motion_features.py from
github.com/Kushal70-51/Windy-Project-3, so their technique
can be run on our clips and scored against ours on the same
days with the same yardstick.

Every constant, threshold, formula and output field below
matches theirs. Where ours differed, THEIRS wins here - that
is the point of the file. The differences are worth knowing,
because they are not cosmetic:

  frame sampling   they keep every 3rd frame of the clip;
                   ours sampled uniformly to a cap of 40.
  flow averaging   they take a plain MEAN of the flow field
                   over the whole ROI. Ours takes a MEDIAN
                   over cloud-masked pixels only. Theirs
                   includes static ground in the average,
                   which damps the reading; ours ignores
                   clear areas entirely.
  consistency      theirs is |mean vector| / mean|vector|,
                   trusted at >= 0.35. Ours is the circular
                   resultant length, trusted at >= 0.5.
  coverage trend   theirs compares the FIRST frame to the
                   LAST, with a +/-3 point dead band. Ours
                   averaged the first half against the
                   second.
  threshold        one cut at 150. Ours splits thin (110)
                   from thick (175).

THE ONE DELIBERATE DEVIATION, AND WHY IT CHANGES NO NUMBER
----------------------------------------------------------
Their loop appends every sampled frame FULL COLOUR to a list
before processing. Their clips are 8 seconds
(ANIMATION_RECORD_SECONDS = 8), so that is fine for them.
Ours are 20 seconds, which at 25 fps and every 3rd frame is
~167 frames of 1280x720x3 - about 460 MB held at once. That
is the exact pattern that got this project's capture
OOM-killed on the ~900 MB EC2 box on 2026-07-26.

So frames here are cropped to the ROI and converted to grey
AS THEY ARE READ. Every computation in their code happens
inside the ROI on the greyscale image anyway, so the numbers
are identical - only the peak memory differs.

SCALE NOTE
----------
Their APPROX_KM_ACROSS_FRAME = 100.0 goes with their
ZOOM_LEVEL = 11. We record at zoom 8, where a frame spans
~712 km, so the SAME fractions cut a ~214 km box out of our
clips instead of their ~30 km one. Running this port on our
existing clips therefore tests their ALGORITHM, not their
FRAMING - the framing needs the capture zoom changed. Keep
the two apart when reading any result.
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np

# --- their constants, unchanged ---
APPROX_KM_ACROSS_FRAME = 100.0
MAP_TOP_FRACTION = 0.08
MAP_BOTTOM_FRACTION = 0.24
ROI_WIDTH_FRACTION = 0.30
ROI_HEIGHT_FRACTION = 0.34
PLANT_MAP_Y_FRACTION = 0.64
FRAME_SAMPLE_STEP = 3
CLOUD_BRIGHTNESS_THRESHOLD = 150


def get_roi_box(width, height):
    """Return the plant ROI inside the map, excluding header/timeline UI."""

    map_top = int(height * MAP_TOP_FRACTION)
    map_bottom = int(height * (1.0 - MAP_BOTTOM_FRACTION))
    map_height = max(1, map_bottom - map_top)
    box_w = int(width * ROI_WIDTH_FRACTION)
    box_h = int(map_height * ROI_HEIGHT_FRACTION)
    x1 = (width - box_w) // 2
    plant_y = map_top + int(map_height * PLANT_MAP_Y_FRACTION)
    y1 = max(map_top, min(map_bottom - box_h, plant_y - box_h // 2))

    return x1, y1, x1 + box_w, y1 + box_h


def direction_from_vector(dx, dy):
    """
    Converts an average optical-flow pixel vector (dx, dy) into a
    compass direction string. Image/video y-coordinates increase
    DOWNWARD, so a positive dy means motion toward the bottom of the
    frame (south) - corrected below.
    """

    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return "negligible / stationary"

    angle = np.degrees(np.arctan2(-dy, dx)) % 360

    directions = [
        "East", "Northeast", "North", "Northwest",
        "West", "Southwest", "South", "Southeast",
    ]
    idx = int(((angle + 22.5) % 360) // 45)

    return directions[idx]


def read_roi_frames(video_path, sample_step=FRAME_SAMPLE_STEP):
    """
    Every `sample_step`-th frame, cropped to the plant ROI and
    converted to grey at read time (see the module docstring on why
    this is not the same as buffering full frames).

    Returns (frames, roi_box) or (None, None) when unreadable.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return None, None

    frames = []
    roi_box = None
    frame_index = 0

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        if frame_index % sample_step == 0:

            if roi_box is None:
                height, width = frame.shape[:2]
                roi_box = get_roi_box(width, height)

            x1, y1, x2, y2 = roi_box

            frames.append(
                cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            )

        frame_index += 1

    capture.release()

    return frames, roi_box


def analyze_video(video_path, sample_step=FRAME_SAMPLE_STEP):
    """
    Their analyze_video, field for field.

    Returns None when the clip cannot be read or has fewer than two
    usable frames, matching their behaviour.
    """

    frames, roi_box = read_roi_frames(video_path, sample_step)

    if frames is None:
        return None

    if len(frames) < 2:
        return None

    x1, _, x2, _ = roi_box

    coverage_pcts = []
    flow_vectors = []
    previous = None

    for gray_roi in frames:

        cloud_pixels = int(
            np.count_nonzero(gray_roi > CLOUD_BRIGHTNESS_THRESHOLD)
        )
        total_pixels = int(gray_roi.size)
        coverage_pcts.append(100.0 * cloud_pixels / total_pixels)

        if previous is not None:

            flow = cv2.calcOpticalFlowFarneback(
                previous, gray_roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            # Plain mean over the WHOLE ROI - theirs, not ours. Ours
            # masks to cloud pixels and takes a median.
            flow_vectors.append((
                float(np.mean(flow[..., 0])),
                float(np.mean(flow[..., 1])),
            ))

        previous = gray_roi

    avg_dx = float(np.mean([v[0] for v in flow_vectors])) if flow_vectors else 0.0
    avg_dy = float(np.mean([v[1] for v in flow_vectors])) if flow_vectors else 0.0

    vector_magnitudes = [float(np.hypot(dx, dy)) for dx, dy in flow_vectors]
    mean_magnitude = float(np.mean(vector_magnitudes)) if vector_magnitudes else 0.0

    directional_consistency = (
        float(np.hypot(avg_dx, avg_dy)) / mean_magnitude
        if mean_magnitude > 1e-9 else 0.0
    )

    motion_score = mean_magnitude / max(1, x2 - x1) * 100.0

    avg_direction = (
        direction_from_vector(avg_dx, avg_dy)
        if directional_consistency >= 0.35 else "negligible / stationary"
    )

    coverage_start = coverage_pcts[0]
    coverage_end = coverage_pcts[-1]
    coverage_delta = coverage_end - coverage_start

    if abs(coverage_delta) < 3:
        coverage_trend = "stable"
    elif coverage_delta > 0:
        coverage_trend = "increasing"
    else:
        coverage_trend = "decreasing"

    summary_text = (
        "Cloud motion analysis (computed via optical flow on the recorded "
        "video, NOT LLM-estimated -- treat these as ground-truth numbers):\n"
        f"- Dominant cloud motion direction over the plant's area: {avg_direction}\n"
        f"- Relative cloud-motion score: {motion_score:.3f} (not km/h)\n"
        f"- Cloud coverage directly over the plant: {coverage_start:.1f}% at "
        f"the start of the clip -> {coverage_end:.1f}% at the end "
        f"({coverage_trend})\n"
        f"- Based on {len(frames)} sampled video frames."
    )

    return {
        "avg_direction": avg_direction,
        "avg_motion_score": round(motion_score, 4),
        "directional_consistency": round(directional_consistency, 3),
        "coverage_start_pct": round(coverage_start, 2),
        "coverage_end_pct": round(coverage_end, 2),
        "coverage_trend": coverage_trend,
        "frame_count": len(frames),
        "summary_text": summary_text,
    }


# --------------------------------------------------

# coverage_trend and avg_direction are strings, and a model needs
# numbers. These are the encodings used when the port's output is fed
# into the feature table.
_TREND_CODE = {"decreasing": -1, "stable": 0, "increasing": 1}

_DIRECTION_DEGREES = {
    "East": 90, "Northeast": 45, "North": 0, "Northwest": 315,
    "West": 270, "Southwest": 225, "South": 180, "Southeast": 135,
}


def features_for_model(video_path, prefix="k1_"):
    """
    analyze_video's output as a flat numeric dict, so it can sit beside
    our own features in one table and be scored the same way.

    A "negligible / stationary" direction becomes NaN, not a number -
    encoding it as 0 would tell the model the clouds were heading due
    north, which is a claim the measurement did not make.
    """

    result = analyze_video(video_path)

    if result is None:
        return None

    direction = _DIRECTION_DEGREES.get(result["avg_direction"], np.nan)

    return {
        f"{prefix}motion_score": result["avg_motion_score"],
        f"{prefix}directional_consistency": result["directional_consistency"],
        f"{prefix}direction_deg": direction,
        f"{prefix}coverage_start_pct": result["coverage_start_pct"],
        f"{prefix}coverage_end_pct": result["coverage_end_pct"],
        f"{prefix}coverage_delta_pct": round(
            result["coverage_end_pct"] - result["coverage_start_pct"], 2
        ),
        f"{prefix}coverage_trend_code": _TREND_CODE.get(
            result["coverage_trend"], 0
        ),
        f"{prefix}frame_count": result["frame_count"],
    }


# --------------------------------------------------

def main():

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Team 1's Windy video motion method, ported verbatim"
    )
    parser.add_argument("video")

    args = parser.parse_args()

    result = analyze_video(args.video)

    if result is None:
        print(f"unreadable: {args.video}")
        return 1

    print(json.dumps(
        {k: v for k, v in result.items() if k != "summary_text"}, indent=2
    ))
    print()
    print(result["summary_text"])

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
