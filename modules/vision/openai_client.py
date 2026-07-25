"""
=========================================================
Solar Forecasting Project
OpenAI (ChatGPT) Vision Client
=========================================================
Analyses Windy cloud frames with an OpenAI multimodal model
(e.g. gpt-4o), as an alternative to the Gemini client.

Fully INDEPENDENT of the Gemini client by design: it imports
the `openai` SDK lazily (inside __init__) and reads ONLY its
own OPENAI_API_KEY. Selecting OpenAI therefore needs neither
the google SDK nor a GOOGLE_API_KEY, and selecting Gemini
needs neither `openai` nor an OPENAI_API_KEY - there is no
cross-dependency or config conflict between the two.

Exposes the same analyze_frames(extraction, prompt) -> text
interface as GeminiClient, so it is a drop-in swap: the
frame extraction, prompt, JSON parsing and forecast around
it are byte-for-byte identical, which is what makes an
LLM-vs-LLM comparison a fair test.
=========================================================
"""

import base64
import time

from config.config import settings
from utils.logger import get_logger


class OpenAIClient:

    def __init__(self):

        vision = settings["vision"]

        api_key = vision.get("openai_api_key")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Add it to your .env file "
                "(only needed when vision.provider is 'openai')."
            )

        # Imported here, not at module top, so a Gemini-only setup
        # never needs the openai package installed.
        from openai import OpenAI

        # base_url lets this same client talk to any OpenAI-compatible
        # gateway - notably OpenRouter (https://openrouter.ai/api/v1),
        # which exposes dozens of vision models (many FREE) behind one
        # key. Left unset = OpenAI direct.
        base_url = vision.get("openai_base_url") or None

        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url else OpenAI(api_key=api_key)
        )

        self.model = vision.get("openai_model", "gpt-4o")
        self.temperature = vision.get("temperature", 0.2)
        self.max_tokens = vision.get("max_output_tokens", 8192)

        self.logger = get_logger()

        retry = vision.get("retry", {})
        self.max_attempts = retry.get("max_attempts", 3)
        self.retry_base_seconds = retry.get("base_seconds", 5)

    # --------------------------------------------------

    @staticmethod
    def _encode_image(frame_path):
        """One frame as a base64 data URL for the vision API."""

        with open(frame_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    # --------------------------------------------------

    def analyze_frames(self, extraction, prompt):
        """
        Sends the extracted frames + prompt to the OpenAI model
        and returns the raw text response (parsed downstream by
        the same JSONParser the Gemini path uses).
        """

        content = [{"type": "text", "text": prompt}]

        for frame_path in extraction["frame_paths"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(frame_path)}
            })

        messages = [{"role": "user", "content": content}]

        # Same transient-error backoff as the Gemini client, so a
        # busy-server blip does not drop the run.
        last_error = None

        for attempt in range(self.max_attempts):

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return response.choices[0].message.content

            except Exception as error:

                if not self.is_transient(error):
                    raise

                last_error = error

                if attempt < self.max_attempts - 1:
                    delay = self.retry_base_seconds * (2 ** attempt)
                    self.logger.warning(
                        f"OpenAI busy (attempt {attempt + 1}"
                        f"/{self.max_attempts}), retrying in {delay}s"
                    )
                    time.sleep(delay)

        raise last_error

    # --------------------------------------------------

    @staticmethod
    def is_transient(error):
        """Retry only server-side/rate-limit hiccups, not bad requests."""

        text = str(error)

        return any(
            marker in text
            for marker in ("429", "500", "502", "503", "504",
                           "rate limit", "RateLimit", "timeout",
                           "Timeout", "overloaded")
        )
