"""
=========================================================
Solar Forecasting Project
Gemini Client
=========================================================
Handles communication with Gemini Vision API.
=========================================================
"""

import os
import time

from google import genai
from google.genai import types

from config.config import settings
from utils.logger import get_logger


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

        self.logger = get_logger()

        retry = settings["vision"].get("retry", {})
        self.max_attempts = retry.get("max_attempts", 3)
        self.retry_base_seconds = retry.get("base_seconds", 5)
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

        config = types.GenerateContentConfig(
            temperature=settings["vision"]["temperature"],
            max_output_tokens=settings["vision"]["max_output_tokens"]
        )

        # The free tier returns 503 ("model is currently experiencing
        # high demand") in bursts - on 2026-07-23 that cost the vision
        # signal on all three runs of the day, because a single refusal
        # dropped the whole run. Congestion usually clears in seconds,
        # so back off and try again rather than discarding the run.
        # Only transient statuses are retried; a bad request or an auth
        # failure is raised immediately, since repeating it is pointless.
        last_error = None

        for attempt in range(self.max_attempts):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )

                return response.text

            except Exception as error:

                if not self.is_transient(error):
                    raise

                last_error = error

                if attempt < self.max_attempts - 1:
                    delay = self.retry_base_seconds * (2 ** attempt)
                    self.logger.warning(
                        f"Gemini busy (attempt {attempt + 1}"
                        f"/{self.max_attempts}), retrying in {delay}s"
                    )
                    time.sleep(delay)

        raise last_error

    # --------------------------------------------------

    @staticmethod
    def is_transient(error):
        """
        True for errors worth retrying: server overload (503),
        rate limiting (429) and gateway/timeout hiccups.
        """

        text = str(error)

        return any(
            marker in text
            for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                           "500", "502", "504", "deadline")
        )
        

    