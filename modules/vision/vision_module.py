"""
=========================================================
Solar Forecasting Project
Vision Module
=========================================================
Controls the complete Vision pipeline.
=========================================================
"""

import json
import re
from datetime import datetime
from pathlib import Path

from config.config import settings
from modules.vision.prompt_builder import PromptBuilder, PROMPT_VERSION
from modules.vision.json_parser import JSONParser
from modules.vision.frame_extractor import FrameExtractor


def make_vision_client():
    """
    Builds ONLY the configured provider's client, importing its
    module lazily so the two stay fully independent: choosing
    'openai' never imports the google SDK or needs a GOOGLE_API_KEY,
    and choosing 'gemini' never imports `openai` or needs an
    OPENAI_API_KEY. No cross-dependency, no config conflict.
    """

    provider = settings["vision"].get("provider", "gemini").lower()

    if provider == "openai":
        from modules.vision.openai_client import OpenAIClient
        return OpenAIClient()

    if provider in ("gemini", "google"):
        from modules.vision.gemini_client import GeminiClient
        return GeminiClient()

    raise ValueError(
        f"Unknown vision.provider '{provider}' - use 'gemini' or 'openai'."
    )


class VisionModule:

    # Matches the date/time embedded in Windy export filenames,
    # regardless of whether "Tab" comes before or after it -
    # e.g. "7-9-2026_11_58-Tab.webm" or "Tab-7-9-2026_10_17.webm".
    FILENAME_TIMESTAMP_PATTERN = re.compile(
        r"(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})_(\d{2})"
    )

    # Original auto-capture naming: "sirmour_260723_12_45.webm"
    # (lowercase, YYMMDD_HH_MM). Superseded 2026-07-24 by
    # CURRENT_TIMESTAMP_PATTERN below, but old files with this name
    # may still exist locally, so it is still tried.
    CAPTURE_TIMESTAMP_PATTERN = re.compile(
        r"sirmour_(\d{2})(\d{2})(\d{2})_(\d{2})_(\d{2})"
    )

    # Current auto-capture naming (see modules/capture/windy_capture.py
    # name_stem): "SIRMOUR_2026-07-24_11-15-00.webm". This was added
    # AFTER the rename to the clearer YYYY-MM-DD_HH-MM-SS format - until
    # this pattern existed, parse_video_time returned None for every
    # current-format file, so the local fallback (and any script
    # scanning video_dir by timestamp) silently saw ZERO of them.
    CURRENT_TIMESTAMP_PATTERN = re.compile(
        r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})"
    )

    VIDEO_EXTENSIONS = ("*.webm", "*.mp4")

    @classmethod
    def parse_video_time(cls, filename):
        """
        Recording time encoded in a video filename, trying the
        current auto-capture format first, then the original
        auto-capture format, then the older hand-export format.
        None if the name matches none of them.
        """

        match = cls.CURRENT_TIMESTAMP_PATTERN.search(filename)

        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            try:
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                return None

        match = cls.CAPTURE_TIMESTAMP_PATTERN.search(filename)

        if match:
            year, month, day, hour, minute = map(int, match.groups())
            try:
                return datetime(2000 + year, month, day, hour, minute)
            except ValueError:
                return None

        match = cls.FILENAME_TIMESTAMP_PATTERN.search(filename)

        if match:
            month, day, year, hour, minute = map(int, match.groups())
            try:
                return datetime(year, month, day, hour, minute)
            except ValueError:
                return None

        return None

    def __init__(self):

        self.extractor = FrameExtractor()

        self.prompt_builder = PromptBuilder()

        self.client = make_vision_client()

        self.parser = JSONParser()

    # --------------------------------------------------

    def find_latest_video(self, video_folder, run_time):
        """
        Finds the most recent Windy video at or before
        run_time, on the same calendar day, so a live/
        backtested run never analyzes a video that would not
        have existed yet at that point in time - and never
        treats a stale video from a previous day as if it
        were a fresh read of today's sky. Returns None if no
        qualifying video exists.
        """

        candidates = []

        for pattern in self.VIDEO_EXTENSIONS:

            for video_path in Path(video_folder).glob(pattern):

                video_time = self.parse_video_time(video_path.name)

                if video_time is None:
                    continue

                if (
                    video_time.date() == run_time.date()
                    and video_time <= run_time
                ):
                    candidates.append((video_time, video_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])

        return candidates[-1][1]

    # --------------------------------------------------

    def analyze_video(
        self,
        video_path,
        output_folder,
        target_frames=10,
        start_fraction=0.0,
        end_fraction=1.0,
        preprocess=False
    ):
        """
        Analyzes a Windy video with the configured LLM. Results
        are cached to disk per video, so re-running the pipeline
        (e.g. during backtesting) does not repeatedly burn API
        quota analyzing the same video.

        start_fraction/end_fraction trim the clip before sampling
        frames (see FrameExtractor.extract_frames) - skips the
        map-loading seconds at the start and the timeline's
        loop-back near the end, so the model sees one clean
        forward sequence, as the prompt assumes.

        preprocess=True runs each frame through
        modules/vision/frame_preprocessor.py (UI crop + contrast
        enhancement) before it is sent to the model - added in
        response to the mentor's question about OpenCV
        pre-processing. Off by default so it never silently
        changes existing cached results.
        """

        video_name = Path(video_path).stem

        cache_file = Path(output_folder) / video_name / "vision_result.json"

        if cache_file.exists():

            with open(cache_file, "r", encoding="utf-8") as file:
                cached = json.load(file)

            # Only reuse the cache if it was produced by the current
            # prompt - a bumped PROMPT_VERSION means the fields changed
            # and the video needs re-analyzing.
            if cached.get("prompt_version") == PROMPT_VERSION:
                return cached

        extraction = self.extractor.extract_frames(
            video_path,
            output_folder,
            target_frames,
            start_fraction=start_fraction,
            end_fraction=end_fraction,
            preprocess=preprocess
        )
        prompt = self.prompt_builder.video_prompt()

        response = self.client.analyze_frames(
            extraction,
            prompt
        )

        result = self.parser.parse(response)

        output = {

            "video_name": Path(video_path).name,

            "prompt_version": PROMPT_VERSION,

            "frame_count": extraction["frame_count"],

            "frame_folder": str(extraction["frame_folder"]),

            "metadata_file": str(extraction["metadata_file"]),

            "weather_features": result

        }

        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=4)

        return output
