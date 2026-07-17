"""
=========================================================
Solar Forecasting Project
Vision Module
=========================================================
Controls the complete Vision pipeline.
=========================================================
"""

from pathlib import Path

from modules.vision.prompt_builder import PromptBuilder
from modules.vision.gemini_client import GeminiClient
from modules.vision.json_parser import JSONParser
from modules.vision.frame_extractor import FrameExtractor


class VisionModule:

    def __init__(self):

        self.extractor = FrameExtractor()

        self.prompt_builder = PromptBuilder()

        self.client = GeminiClient()

        self.parser = JSONParser()

    # --------------------------------------------------

    def analyze_video(
        self,
        video_path,
        output_folder,
        target_frames=10
    ):

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

        return {

            "video_name": Path(video_path).name,

            "frame_count": extraction["frame_count"],

            "frame_folder": extraction["frame_folder"],

            "metadata_file": extraction["metadata_file"],

            "weather_features": result

        }