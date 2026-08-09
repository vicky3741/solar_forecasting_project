"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Satellite Clip Features
=========================================================
Deterministic OpenCV features from the EUMETSAT satellite
clip, to sit alongside the scraped Windy numbers.

WHY THE SATELLITE CLIP IS STILL WORTH READING
---------------------------------------------
modules/capture/windy_scraper.py replaced the pixel route for
everything Windy renders from a model - solar power, cloud
decks, rain, wind all come back as exact numbers now. The
satellite layer is the exception, for two measured reasons:

  * it is OBSERVATION, not forecast - EUMETSAT imagery of the
    sky that is actually there, which no forecast model gives
    us; and
  * it cannot be scraped as a series. Stepping the embed's
    5-minutely satellite timeline returned one identical
    reading for all 24 steps (241 K / raw 147), because the
    embed holds a single frame in memory.

So for this one layer the pixels really are the only access,
and a clip is the only way to see it move.

WHAT IS COMPUTED, AND WHY THESE AND NOT BRIGHTNESS AVERAGES
-----------------------------------------------------------
Three of these come from Abhijit's feature_extractor.py in
github.com/Abhijitugalmogale/solar_power_gerneration, which
got several things right that the exported features log then
threw away:

  thin vs thick cloud  - one brightness threshold cannot tell
      thin cirrus from a thunderhead, and they attenuate
      completely differently. Two thresholds split the ROI
      into clear / thin / thick.
  entropy              - Shannon entropy of the brightness
      histogram, a patchiness number. settings.yaml already
      records that patchy/broken skies measured 4.2x worse DSM
      penalty than calm ones; today that judgement comes from
      Gemini in words, and this is the deterministic version.
  divergence/vorticity - derivatives of the optical-flow field.
      This matters: Windy dissolves between hourly satellite
      stills rather than sliding them, so MEAN DISPLACEMENT is
      ~0 and direction is unmeasurable (see
      tests/test_why_motion_is_zero.py in ved_solar_project).
      Divergence and vorticity are SPATIAL DERIVATIVES of the
      flow field, so they can be non-zero even when net
      translation is zero - cloud spreading out or spinning up
      shows here when displacement shows nothing.

Explicitly NOT computed: mean hue. Hue is circular, so its
arithmetic mean is meaningless - the average of 350 and 10
degrees comes out 180, the opposite colour. The reference
features log carried four such columns per layer.

ONE CLIP IS ONE MOMENT
----------------------
Everything here describes the sky AT capture time. It is a
nowcast, not a per-block series, and callers must treat it that
way - modules/fusion/llm_scheduler.py puts these in the prompt
header as "current sky observation" rather than as columns
repeated down the block table. Repeating one observation across
96 rows, with nothing saying it was repeated, is exactly what
made the reference log useless.
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np

from modules.vision.optical_flow import OpticalFlowAnalyzer
from utils.logger import get_logger


class SatelliteFeatureExtractor:

    def __init__(
        self,
        roi_fraction=0.6,
        thin_cloud_threshold=110,
        thick_cloud_threshold=175,
        max_frames=24,
    ):
        """
        roi_fraction : centre crop actually measured. 0.6 follows
            Abhijit's extractor; the map edges carry Windy's timeline,
            legend and logo, which are static UI and would otherwise
            be counted as permanently clear sky.
        thin/thick   : brightness cut points separating clear sky from
            thin cloud from thick cloud.
        """

        self.logger = get_logger()

        self.roi_fraction = roi_fraction
        self.thin_threshold = thin_cloud_threshold
        self.thick_threshold = thick_cloud_threshold

        # Frame reading is delegated rather than reimplemented: this
        # analyzer already crops and greys frames AS THEY ARE READ and
        # halves its buffer when it grows, which is what stopped the
        # OOM kill on the ~900 MB EC2 box (2026-07-26, a 30 s clip
        # needed ~2 GB when full-colour frames were buffered first).
        self.reader = OpticalFlowAnalyzer(
            roi_fraction=roi_fraction,
            cloud_brightness_threshold=thin_cloud_threshold,
            max_frames=max_frames,
        )

    # --------------------------------------------------

    def cloud_classes(self, frame):
        """
        Fraction of the ROI that is clear sky, thin cloud and thick
        cloud, by brightness.
        """

        total = frame.size

        thick = float(np.count_nonzero(frame >= self.thick_threshold) / total)
        thin = float(
            np.count_nonzero(
                (frame >= self.thin_threshold) & (frame < self.thick_threshold)
            ) / total
        )

        return {
            "clear_pct": round((1.0 - thin - thick) * 100, 2),
            "thin_cloud_pct": round(thin * 100, 2),
            "thick_cloud_pct": round(thick * 100, 2),
        }

    # --------------------------------------------------

    @staticmethod
    def entropy(frame):
        """
        Shannon entropy of the brightness histogram, in bits.

        Low  (~0-3) = uniform sky, either solidly clear or solidly
                      overcast - either way, predictable.
        High (~6-8) = many different brightnesses in one view, i.e. a
                      broken, patchy field. That is the volatile case
                      where 15-minute blocks swing hardest.
        """

        histogram = np.bincount(frame.ravel(), minlength=256).astype(float)

        total = histogram.sum()

        if total <= 0:
            return 0.0

        probabilities = histogram / total
        non_zero = probabilities[probabilities > 0]

        return float(-np.sum(non_zero * np.log2(non_zero)))

    # --------------------------------------------------

    def quadrants(self, frame):
        """
        Cloud fraction in each half of the ROI.

        The plant sits at the centre of the crop, so these say which
        side weather is sitting on. Combined with the wind bearing
        the scraper reads, an upwind half that is cloudier than the
        downwind half means cloud is on its way in.
        """

        height, width = frame.shape

        clouded = frame >= self.thin_threshold

        return {
            "north_cloud_pct": round(float(clouded[:height // 2, :].mean()) * 100, 2),
            "south_cloud_pct": round(float(clouded[height // 2:, :].mean()) * 100, 2),
            "west_cloud_pct": round(float(clouded[:, :width // 2].mean()) * 100, 2),
            "east_cloud_pct": round(float(clouded[:, width // 2:].mean()) * 100, 2),
        }

    # --------------------------------------------------

    def flow_fields(self, frames):
        """
        Divergence, vorticity and kinetic energy of the optical-flow
        field, averaged over consecutive frame pairs.

        divergence > 0 : the field is spreading out - cloud thinning
                         or dissipating over the area.
        divergence < 0 : converging - cloud piling up, which is what
                         precedes a build-up.
        |vorticity|    : rotation, i.e. organised/rotating weather
                         rather than a flat sheet drifting past.
        kinetic energy : how much movement there is at all, without
                         caring about its direction - the honest
                         measure here, since net direction on Windy's
                         crossfaded stills is not measurable.
        """

        divergences = []
        vorticities = []
        energies = []

        for previous, current in zip(frames, frames[1:]):

            flow = cv2.calcOpticalFlowFarneback(
                previous, current, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            u = flow[..., 0]
            v = flow[..., 1]

            # np.gradient returns d/drow then d/dcol, i.e. y then x.
            du_dy, du_dx = np.gradient(u)
            dv_dy, dv_dx = np.gradient(v)

            divergences.append(float(np.mean(du_dx + dv_dy)))
            vorticities.append(float(np.mean(np.abs(dv_dx - du_dy))))
            energies.append(float(0.5 * np.mean(u ** 2 + v ** 2)))

        if not divergences:
            return {
                "flow_divergence": 0.0,
                "flow_vorticity": 0.0,
                "flow_kinetic_energy": 0.0,
            }

        return {
            # Median across pairs, not mean: Windy's timeline loops
            # back part-way through a recording and that wrap produces
            # one or two wild pairs which a mean would let dominate.
            "flow_divergence": round(float(np.median(divergences)), 6),
            "flow_vorticity": round(float(np.median(vorticities)), 6),
            "flow_kinetic_energy": round(float(np.median(energies)), 6),
        }

    # --------------------------------------------------

    def extract(self, video_path):
        """
        All satellite features for one clip, as a flat dict with
        `sat_` prefixes ready to merge into the feature table.
        """

        frames, _, _ = self.reader.read_frames(video_path)

        if len(frames) < 2:
            raise ValueError(f"need at least two frames: {video_path}")

        features = {"sat_video": Path(video_path).name,
                    "sat_frames_used": len(frames)}

        # --- brightness, over the whole clip ---
        brightness = np.array([float(f.mean()) for f in frames])

        features["sat_brightness_mean"] = round(float(brightness.mean()), 2)
        features["sat_brightness_std"] = round(float(brightness.std()), 2)
        features["sat_contrast"] = round(
            float(np.mean([float(f.std()) for f in frames])), 2
        )

        # --- cloud classes and texture, averaged over frames ---
        classes = [self.cloud_classes(f) for f in frames]

        for key in ("clear_pct", "thin_cloud_pct", "thick_cloud_pct"):
            features[f"sat_{key}"] = round(
                float(np.mean([c[key] for c in classes])), 2
            )

        features["sat_entropy"] = round(
            float(np.mean([self.entropy(f) for f in frames])), 3
        )

        # --- where the cloud is, from the middle of the clip ---
        features.update({
            f"sat_{key}": value
            for key, value in self.quadrants(frames[len(frames) // 2]).items()
        })

        # --- is the sky clouding over or clearing across the clip? ---
        half = max(1, len(classes) // 2)

        start = float(np.mean([
            c["thin_cloud_pct"] + c["thick_cloud_pct"] for c in classes[:half]
        ]))
        end = float(np.mean([
            c["thin_cloud_pct"] + c["thick_cloud_pct"] for c in classes[half:]
        ]))

        features["sat_cloud_trend_pct"] = round(end - start, 2)

        # --- motion structure ---
        features.update({
            f"sat_{key}": value
            for key, value in self.flow_fields(frames).items()
        })

        return features


# --------------------------------------------------

def extract_for(video_path, **kwargs):
    """
    Convenience wrapper that never raises: returns None and logs when
    a clip is unreadable, so a bad video degrades the feature table
    rather than killing the forecast run.
    """

    logger = get_logger()

    try:
        return SatelliteFeatureExtractor(**kwargs).extract(video_path)

    except Exception as error:
        logger.warning(
            f"Satellite features failed for {Path(video_path).name} "
            f"({error}) - continuing without them"
        )
        return None


# --------------------------------------------------

def main():

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Satellite clip features (thin/thick, entropy, flow structure)"
    )
    parser.add_argument("video", help="path to a Windy satellite .webm/.mp4 clip")
    parser.add_argument("--roi", type=float, default=0.6)

    args = parser.parse_args()

    features = SatelliteFeatureExtractor(roi_fraction=args.roi).extract(args.video)

    print(json.dumps(features, indent=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
