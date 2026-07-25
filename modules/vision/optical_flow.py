"""
=========================================================
Solar Forecasting Project
Optical Flow Cloud Motion
=========================================================
Deterministic cloud-motion and cloud-cover measurement
from a Windy satellite animation, using dense optical flow
(Farneback) plus brightness thresholding.

This complements - it does NOT replace - the Gemini vision
signal and the physics/kt core. Where Gemini *describes*
the sky in words, this produces hard numbers from pixels:
free, instant, repeatable, and with no API quota. Both are
inputs to the same clear-sky-index pipeline.

Honest limits on the numbers produced here:

  * Direction, relative speed and cover fraction are
    measured directly and are reliable.
  * Real-world km/h is an ESTIMATE. It needs the map scale
    (derived from the Web-Mercator zoom level) AND how much
    real time one video second represents - the Windy
    timeline is time-compressed during playback, and that
    factor is a configured assumption, not a measurement.
    Treat the km/h figure as indicative only; prefer the
    direction and trend values for decisions.
=========================================================
"""

import math
from pathlib import Path

import cv2
import numpy as np


# Web-Mercator ground resolution at zoom 0, in metres per pixel.
_EARTH_METRES_PER_PIXEL_Z0 = 156543.03392

_COMPASS_POINTS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]


def metres_per_pixel(latitude, zoom):
    """
    Ground distance covered by one screen pixel on a
    Web-Mercator tile map at the given latitude and zoom.
    """

    return (
        _EARTH_METRES_PER_PIXEL_Z0
        * math.cos(math.radians(latitude))
        / (2 ** zoom)
    )


def compass_direction(dx, dy):
    """
    Compass bearing clouds are moving TOWARD.

    Image coordinates run y-downward, so a positive dy is
    southward movement - the sign is flipped here so the
    result reads as a normal map bearing.
    """

    if dx == 0 and dy == 0:
        return "none"

    bearing = (math.degrees(math.atan2(dx, -dy)) + 360) % 360

    index = int((bearing + 11.25) % 360 / 22.5)

    return _COMPASS_POINTS[index]


class OpticalFlowAnalyzer:

    def __init__(
        self,
        latitude=24.56,
        zoom=8,
        seconds_per_video_second=360,
        roi_fraction=0.5,
        cloud_brightness_threshold=140,
        max_frames=40
    ):
        """
        seconds_per_video_second : how much real time one
            second of playback represents. Windy's animation
            is time-compressed; this is an assumption used
            only for the indicative km/h figure.
        roi_fraction : fraction of the frame, centred on the
            plant, actually measured - the map edges carry
            UI chrome (timeline, legend, logo) that would
            otherwise pollute the flow field.
        """

        self.latitude = latitude
        self.zoom = zoom
        self.seconds_per_video_second = seconds_per_video_second
        self.roi_fraction = roi_fraction
        self.cloud_brightness_threshold = cloud_brightness_threshold
        self.max_frames = max_frames

    # --------------------------------------------------

    def region_of_interest(self, frame):
        """
        Centre crop of the frame, avoiding Windy's overlaid
        UI around the edges.
        """

        height, width = frame.shape[:2]

        half = self.roi_fraction / 2

        y0 = int(height * (0.5 - half))
        y1 = int(height * (0.5 + half))
        x0 = int(width * (0.5 - half))
        x1 = int(width * (0.5 + half))

        return frame[y0:y1, x0:x1]

    # --------------------------------------------------

    def read_frames(self, video_path):
        """
        Uniformly sampled greyscale ROI frames from the video,
        capped at max_frames.

        Frames are cropped and converted to grey AS THEY ARE READ,
        and the buffer is halved whenever it grows too large, so
        peak memory stays flat no matter how long the clip is.

        An earlier version appended every full-colour frame to a
        list before sampling. At 1280x720x3 bytes that is ~2 MB a
        frame, so a routine 30-second 25 fps clip needed ~2 GB and
        the OOM killer took the process out on the 900 MB EC2 box
        (2026-07-26) - after the clip was recorded but before it
        reached S3. Clip length must not drive memory here.
        """

        video = cv2.VideoCapture(str(video_path))

        if not video.isOpened():
            raise FileNotFoundError(f"Unable to open video:\n{video_path}")

        fps = video.get(cv2.CAP_PROP_FPS) or 25.0

        # Kept frames are `stride` source frames apart. Whenever the
        # buffer exceeds the limit, every other frame is dropped and
        # the stride doubles - so the retained frames stay uniformly
        # spaced and the reported stride stays accurate.
        buffer_limit = max(2, self.max_frames * 2)

        frames = []
        stride = 1
        index = 0

        while True:
            success, frame = video.read()

            if not success:
                break

            if index % stride == 0:
                frames.append(
                    cv2.cvtColor(
                        self.region_of_interest(frame), cv2.COLOR_BGR2GRAY
                    )
                )

                if len(frames) > buffer_limit:
                    frames = frames[::2]
                    stride *= 2

            index += 1

        video.release()

        if not frames:
            raise ValueError(f"No readable frames in {video_path}")

        return frames[:self.max_frames], fps, stride

    # --------------------------------------------------

    def cloud_cover_fraction(self, grey_frame):
        """
        Fraction of the ROI brighter than the cloud
        threshold. On Windy's visible-satellite layer cloud
        reads bright against darker ground.
        """

        return float(
            np.mean(grey_frame > self.cloud_brightness_threshold)
        )

    # --------------------------------------------------

    def analyze(self, video_path):
        """
        Returns cloud motion + cover measurements for one
        Windy animation clip.
        """

        grey_frames, fps, step = self.read_frames(video_path)

        if len(grey_frames) < 2:
            raise ValueError("Need at least two frames for optical flow")

        flows_dx = []
        flows_dy = []

        for previous, current in zip(grey_frames, grey_frames[1:]):

            flow = cv2.calcOpticalFlowFarneback(
                previous, current,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=25,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Measure only where there is actually cloud. The
            # basemap (coastlines, borders, place labels) is
            # static, so averaging over the whole frame drags
            # the estimate towards zero and badly understates
            # cloud speed.
            cloud_mask = previous > self.cloud_brightness_threshold

            if cloud_mask.sum() < 50:      # effectively clear sky
                continue

            # Median over cloud pixels only: robust to the
            # wild vectors dense flow produces on featureless
            # patches, without being diluted by static ground.
            flows_dx.append(float(np.median(flow[..., 0][cloud_mask])))
            flows_dy.append(float(np.median(flow[..., 1][cloud_mask])))

        if not flows_dx:
            flows_dx = [0.0]
            flows_dy = [0.0]

        # Median ACROSS frame pairs, not mean. Windy's timeline
        # loops back to the start part-way through a recording,
        # and that wrap shows up as one or two huge reverse-motion
        # pairs. A mean lets those outliers dominate (and can even
        # flip the reported direction); the median ignores them.
        mean_dx = float(np.median(flows_dx))
        mean_dy = float(np.median(flows_dy))

        pixels_per_step = math.hypot(mean_dx, mean_dy)

        # Agreement between individual pairs, used to say whether
        # the motion reading can be trusted at all (see below).
        pair_directions = [
            math.degrees(math.atan2(dx, -dy)) % 360
            for dx, dy in zip(flows_dx, flows_dy)
            if math.hypot(dx, dy) > 1e-6
        ]

        # px/frame -> km/h, via map scale and the assumed
        # playback time compression (see the module docstring).
        seconds_between_sampled_frames = (step / fps) if fps else 0
        real_seconds = (
            seconds_between_sampled_frames * self.seconds_per_video_second
        )

        metres_per_px = metres_per_pixel(self.latitude, self.zoom)

        # Absolute speed is only reported when the measured
        # displacement is large enough for the calibration to
        # mean anything. On Windy's satellite layer the frames
        # crossfade between hourly images rather than translating
        # cleanly, so sub-pixel motion is indistinguishable from
        # noise - emitting a km/h figure from it would look
        # authoritative while being fabricated.
        _MIN_PIXELS_FOR_SPEED = 0.5

        speed_kmh = None
        if real_seconds > 0 and pixels_per_step >= _MIN_PIXELS_FOR_SPEED:
            metres_per_second = (pixels_per_step * metres_per_px) / real_seconds
            speed_kmh = metres_per_second * 3.6

        covers = [self.cloud_cover_fraction(f) for f in grey_frames]

        half = max(1, len(covers) // 2)
        cover_start = float(np.mean(covers[:half]))
        cover_end = float(np.mean(covers[half:]))

        # Circular consistency of the per-pair directions: the
        # resultant length of the unit vectors. ~1.0 means every
        # pair agreed (trustworthy); ~0.0 means the directions
        # were scattered and the motion reading is noise, which
        # is what happens on a clip with little real cloud
        # translation or a mid-clip timeline wrap.
        if pair_directions:
            radians = np.radians(pair_directions)
            consistency = float(
                math.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians)))
            )
        else:
            consistency = 0.0

        trustworthy = consistency >= 0.5

        return {
            "cloud_cover_fraction": round(float(np.mean(covers)), 4),
            "cloud_cover_start": round(cover_start, 4),
            "cloud_cover_end": round(cover_end, 4),
            # Positive = clouding over, negative = clearing.
            "cloud_cover_trend": round(cover_end - cover_start, 4),
            "motion_direction": (
                compass_direction(mean_dx, mean_dy) if trustworthy else "uncertain"
            ),
            "motion_pixels_per_frame": round(pixels_per_step, 4),
            "motion_speed_kmh_estimate": (
                round(speed_kmh, 1) if speed_kmh is not None else None
            ),
            "motion_consistency": round(consistency, 3),
            "motion_trustworthy": trustworthy,
            "frames_analyzed": len(grey_frames),
            "metres_per_pixel": round(metres_per_px, 1)
        }


# --------------------------------------------------

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Measure cloud motion in a Windy clip"
    )
    parser.add_argument("video", help="path to a Windy .webm/.mp4 clip")
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--lat", type=float, default=24.56)

    args = parser.parse_args()

    analyzer = OpticalFlowAnalyzer(latitude=args.lat, zoom=args.zoom)

    print(json.dumps(analyzer.analyze(Path(args.video)), indent=2))
