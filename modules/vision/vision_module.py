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

from modules.vision.prompt_builder import PromptBuilder, PROMPT_VERSION
from modules.vision.gemini_client import GeminiClient
from modules.vision.json_parser import JSONParser
from modules.vision.frame_extractor import FrameExtractor


class VisionModule:

    # Matches the date/time embedded in Windy export filenames,
    # regardless of whether "Tab" comes before or after it -
    # e.g. "7-9-2026_11_58-Tab.webm" or "Tab-7-9-2026_10_17.webm".
    FILENAME_TIMESTAMP_PATTERN = re.compile(
        r"(\d{1,2})-(\d{1,2})-(\d{4})_(\d{1,2})_(\d{2})"
    )

    def __init__(self):

        self.extractor = FrameExtractor()

        self.prompt_builder = PromptBuilder()

        self.client = GeminiClient()

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

        for video_path in Path(video_folder).glob("*.webm"):

            match = self.FILENAME_TIMESTAMP_PATTERN.search(video_path.name)

            if not match:
                continue

            month, day, year, hour, minute = map(int, match.groups())

            try:
                video_time = datetime(year, month, day, hour, minute)
            except ValueError:
                continue

            if video_time.date() == run_time.date() and video_time <= run_time:
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
        target_frames=10
    ):
        """
        Analyzes a Windy video with Gemini. Results are cached
        to disk per video, so re-running the pipeline (e.g.
        during backtesting) does not repeatedly burn API quota
        analyzing the same video.
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
            target_frames
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
