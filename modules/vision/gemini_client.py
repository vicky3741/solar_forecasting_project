"""
=========================================================
Solar Forecasting Project
Gemini Client
=========================================================
Handles communication with Gemini Vision API.
=========================================================
"""

import os
from google import genai
from google.genai import types

from config.config import settings


class GeminiClient:

    def __init__(self):

        api_key = settings["vision"]["api_key"]

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please check your .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = settings["vision"]["model"]
    # --------------------------------------------------

    def analyze_frames(
        self,
        extraction,
        prompt
    ):
        frame_paths = extraction["frame_paths"]
        contents = [prompt]

        for frame_path in frame_paths:

            with open(frame_path, "rb") as image:

                image_bytes = image.read()

            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                )
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents
        )

        return response.text
        

    