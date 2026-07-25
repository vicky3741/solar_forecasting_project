"""
=========================================================
Solar Forecasting Project
Frame Preprocessor
=========================================================
Cleans up a raw Windy frame with OpenCV before it is sent
to an LLM. Added 2026-07-25 in response to the mentor's
question ("are you pre-processing video with OpenCV?") -
previously frames went straight from the video to the LLM
with no enhancement at all.

Two things happen, in order:

  1. Crop out the fixed Windy UI chrome (logo, zoom buttons,
     timeline bar) - see modules/capture/windy_capture.py,
     which always records at a fixed 1280x720 viewport, so
     the chrome sits at the same pixels on every frame. This
     directs the model's attention at the sky/ground content
     instead of interface furniture.
  2. CLAHE contrast enhancement on the luminance channel -
     boosts the visibility of cloud edges and density
     gradations that are subtle in the raw satellite frame,
     without blowing out colour like a naive contrast/gamma
     stretch would.

This is a candidate fix for the "ChatGPT gives an almost
identical answer for every video" finding
(tests/test_llm_comparison.py) - if the model was reading
faint/washed-out frames, sharper contrast may help it
actually discriminate between videos. Validate with
tests/test_llm_comparison.py (preprocess=True) before
trusting it; do not assume it helps without measuring.
=========================================================
"""

import cv2
import numpy as np


# Pixels to crop from each edge of a 1280x720 Windy capture, based on
# where the fixed UI chrome sits (see windy_capture.py embed_url/
# vis_button_position/play_button_position for the same fixed layout).
# Conservative - trims chrome without cutting into the map content
# near the plant marker at the frame centre.
DEFAULT_CROP = {"top": 0, "bottom": 60, "left": 0, "right": 90}


def crop_ui_chrome(frame, crop=None):
    """Removes the fixed-position Windy UI chrome from one frame."""

    crop = crop or DEFAULT_CROP
    height, width = frame.shape[:2]

    top = crop.get("top", 0)
    bottom = height - crop.get("bottom", 0)
    left = crop.get("left", 0)
    right = width - crop.get("right", 0)

    return frame[top:bottom, left:right]


def enhance_contrast(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    L channel of LAB colour space - boosts local contrast (cloud
    edges, density gradients) without shifting colour balance the
    way equalizing RGB channels directly would.
    """

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(frame, strength=5):
    """
    Light denoising - satellite/cloud frames can carry compression
    artefacts from the video codec that read as false texture.
    Fast variant (not Non-Local-Means) to keep per-frame cost low
    across a whole video's worth of frames.
    """

    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def preprocess_frame(frame, crop=True, contrast=True, denoise_frame=False):
    """
    Applies the enabled steps in a fixed, sensible order: crop first
    (so later steps do not spend effort on chrome pixels), then
    denoise (before contrast enhancement, so CLAHE does not amplify
    noise), then contrast enhancement last.
    """

    if crop:
        frame = crop_ui_chrome(frame)

    if denoise_frame:
        frame = denoise(frame)

    if contrast:
        frame = enhance_contrast(frame)

    return frame
