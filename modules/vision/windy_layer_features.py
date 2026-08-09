"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Windy layer screenshots - verbatim port of Team 1's method
=========================================================
Same-to-same port of image_feature_extraction.py from
github.com/Kushal70-51/Windy-Project-3.

Five statistics per layer screenshot, taken from a box around
the plant rather than the whole picture:

    avg_brightness      mean grey value in the box
    brightness_std      how varied it is - patchy vs uniform
    avg_saturation      mean HSV saturation
    avg_hue_deg         CIRCULAR mean hue (see below)
    bright_pixel_pct    % of the box brighter than 150

Output names are "{layer}_{stat}", e.g.
satellite_avg_brightness - matching their features log exactly,
so a row from this module and a row from theirs are directly
comparable.

THE HUE FIX IS THEIRS AND IT IS CORRECT
---------------------------------------
Hue is an angle, so it wraps: 359 degrees and 1 degree are two
degrees apart, not 358. An arithmetic mean of those gives 180 -
the opposite colour. The features log we were originally handed
carried four such columns per layer, which is why they measured
nothing.

They fixed it properly, by averaging unit vectors:

    hue_rad  = hue * (2*pi / 180)        # OpenCV stores hue 0-179
    mean_sin = mean(sin(hue_rad))
    mean_cos = mean(cos(hue_rad))
    avg_hue  = degrees(arctan2(mean_sin, mean_cos)) % 360

Note the 2*pi/180 factor: OpenCV packs a 0-360 degree hue into
0-179 to fit one byte, so each unit is two real degrees.

THE ROI BOX
-----------
Strip Windy's UI (top 8%, bottom 24%), then take a box 30% of
the frame wide and 34% of the map tall, centred left-to-right
and sitting 64% down the map - where the plant marker is.

At THEIR zoom 11 that box is about 30 km of sky over the plant.
At OUR zoom 8 the identical fractions cut about 214 km. Same
code, very different measurement - so a clip's zoom has to be
known before any of these numbers mean anything.
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np

# --- their constants, unchanged ---
MAP_TOP_FRACTION = 0.08
MAP_BOTTOM_FRACTION = 0.24
ROI_WIDTH_FRACTION = 0.30
ROI_HEIGHT_FRACTION = 0.34
PLANT_MAP_Y_FRACTION = 0.64
BRIGHT_PIXEL_THRESHOLD = 150

# The layers they capture, in their order.
LAYERS = ["satellite", "wind", "solarpower", "clouds", "rain"]


def get_roi_box(width, height):
    """The plant box inside the map, excluding header/timeline UI."""

    map_top = int(height * MAP_TOP_FRACTION)
    map_bottom = int(height * (1.0 - MAP_BOTTOM_FRACTION))
    map_height = max(1, map_bottom - map_top)
    box_w = int(width * ROI_WIDTH_FRACTION)
    box_h = int(map_height * ROI_HEIGHT_FRACTION)
    x1 = (width - box_w) // 2
    plant_y = map_top + int(map_height * PLANT_MAP_Y_FRACTION)
    y1 = max(map_top, min(map_bottom - box_h, plant_y - box_h // 2))

    return x1, y1, x1 + box_w, y1 + box_h


def extract_single_image_stats(filepath):
    """
    Their five statistics for one screenshot. Returns all-None (never
    zeros) when the image cannot be read - a fabricated 0 brightness is
    indistinguishable from a genuinely black frame downstream.
    """

    img = cv2.imread(str(filepath))

    if img is None:
        return {
            "avg_brightness": None, "brightness_std": None,
            "avg_saturation": None, "avg_hue_deg": None,
            "bright_pixel_pct": None,
        }

    height, width = img.shape[:2]
    x1, y1, x2, y2 = get_roi_box(width, height)
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    avg_brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    avg_saturation = float(np.mean(hsv[..., 1]))

    # Circular mean. OpenCV packs hue into 0-179, so one unit is two
    # real degrees - hence 2*pi/180 rather than pi/180.
    hue_rad = hsv[..., 0].astype(np.float64) * (2 * np.pi / 180.0)
    mean_sin = float(np.mean(np.sin(hue_rad)))
    mean_cos = float(np.mean(np.cos(hue_rad)))
    avg_hue_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)

    bright_pixels = int(np.count_nonzero(gray > BRIGHT_PIXEL_THRESHOLD))
    bright_pixel_pct = 100.0 * bright_pixels / gray.size

    return {
        "avg_brightness": round(avg_brightness, 2),
        "brightness_std": round(brightness_std, 2),
        "avg_saturation": round(avg_saturation, 2),
        "avg_hue_deg": round(avg_hue_deg, 2),
        "bright_pixel_pct": round(bright_pixel_pct, 2),
    }


def extract_layer_features(screenshots):
    """
    {layer: path} -> flat {layer_stat: value} dict, their naming.

    A layer whose screenshot is missing still contributes its five
    keys, set to None, so every row in the features log has the same
    columns whether or not a capture succeeded.
    """

    features = {}

    for layer in LAYERS:

        path = screenshots.get(layer)

        stats = (
            extract_single_image_stats(path) if path
            else {
                "avg_brightness": None, "brightness_std": None,
                "avg_saturation": None, "avg_hue_deg": None,
                "bright_pixel_pct": None,
            }
        )

        for name, value in stats.items():
            features[f"{layer}_{name}"] = value

    return features


def extract_from_folder(folder, stem=None):
    """
    Reads a folder of "<stem>_<layer>.png" screenshots - the naming our
    own capture already uses - and returns their feature dict.
    """

    folder = Path(folder)

    screenshots = {}

    for layer in LAYERS:

        pattern = f"*_{layer}.png" if stem is None else f"{stem}_{layer}.png"

        matches = sorted(folder.glob(pattern))

        if matches:
            screenshots[layer] = matches[-1]

    return extract_layer_features(screenshots)


# --------------------------------------------------

def main():

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Team 1's layer-screenshot features, ported verbatim"
    )
    parser.add_argument("folder", help="folder of <stem>_<layer>.png files")
    parser.add_argument("--stem", default=None)

    args = parser.parse_args()

    features = extract_from_folder(args.folder, args.stem)

    print(json.dumps(features, indent=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
