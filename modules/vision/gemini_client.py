"""
=========================================================
Solar Forecasting Project
Gemini Client
=========================================================
Handles communication with Gemini Vision API.
=========================================================
"""

import os
import random
import re
import time

from google import genai
from google.genai import types

from config.config import settings
from utils.logger import get_logger


class DailyQuotaExhausted(RuntimeError):
    """
    The free tier's per-day request allowance for this model is gone.

    Distinct from ordinary congestion because no amount of backing off
    inside a run will clear it - the window is 24 hours. Callers should
    stop asking rather than spend the run retrying.
    """


class ModelUnavailable(RuntimeError):
    """
    This model name is retired or not available to this key.

    Google keeps listing such models via models.list() but answers 404
    ("no longer available to new users") when they are called, so a name
    cannot be trusted just because it appears in the catalogue. Treated
    like an exhausted model: step over it and carry on down the chain.
    """


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

        # Each model has its own separate per-day free-tier budget, so a
        # model whose day is spent can be stepped over rather than losing
        # the vision signal for the whole run.
        self.fallback_models = [
            name for name in settings["vision"].get("fallback_models", [])
            if name and name != self.model
        ]

        # Once a model reports its day gone, stop asking it for the rest
        # of this process - seven runs would otherwise each pay the same
        # refusal.
        self.exhausted = set()

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
        #
        # A per-DAY quota 429 is not congestion and must not be retried:
        # on 2026-07-30 the whole day's vision was lost to a 20-requests
        # -per-day cap while the log said only "Gemini busy", which hid
        # the real cause for days. It is now named and raised at once.
        candidates = [
            name for name in [self.model, *self.fallback_models]
            if name not in self.exhausted
        ]

        if not candidates:
            raise DailyQuotaExhausted(
                "every configured Gemini model has spent its daily free-tier "
                "quota; wait for the reset or set vision.provider to openai"
            )

        blocked_error = None

        for name in candidates:

            try:
                return self._generate(name, contents, config)

            except (DailyQuotaExhausted, ModelUnavailable) as error:

                self.exhausted.add(name)
                blocked_error = error

                reason = (
                    "has spent its daily free-tier quota"
                    if isinstance(error, DailyQuotaExhausted)
                    else "is not available to this key"
                )

                remaining = [
                    other for other in candidates
                    if other not in self.exhausted
                ]

                if remaining:
                    self.logger.warning(
                        f"Falling back to {remaining[0]} - {name} {reason}."
                    )
                else:
                    self.logger.error(
                        f"No Gemini model left to try - {name} {reason}."
                    )

        raise blocked_error

    # --------------------------------------------------

    def _generate(self, model, contents, config):
        """
        One model's attempt, with backoff over transient refusals.

        Raises DailyQuotaExhausted when this model's per-day allowance is
        gone, so the caller can step to the next model.
        """

        last_error = None

        for attempt in range(self.max_attempts):

            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                if model != self.model:
                    self.logger.info(f"Vision served by fallback {model}")

                return response.text

            except Exception as error:

                limit = self.daily_quota_limit(error)

                if limit is not None:
                    self.logger.error(
                        f"Daily free-tier quota exhausted for {model} "
                        f"(limit {limit} requests/day). Not retrying - the "
                        f"window is 24 hours."
                    )
                    raise DailyQuotaExhausted(
                        f"{model}: {limit} requests/day exhausted"
                    ) from error

                if self.is_model_unavailable(error):
                    self.logger.warning(
                        f"Model {model} is retired or not available to this "
                        f"key; skipping it."
                    )
                    raise ModelUnavailable(f"{model}: not available") from error

                if not self.is_transient(error):
                    raise

                last_error = error

                if attempt < self.max_attempts - 1:

                    # Prefer the server's own RetryInfo when it sends one;
                    # guessing shorter than it asks just wastes an attempt.
                    delay = self.retry_after(error)

                    if delay is None:
                        delay = self.retry_base_seconds * (2 ** attempt)

                    # Jitter, so seven runs backing off together do not
                    # keep colliding on the same retry instant.
                    delay += random.uniform(0, 0.3 * delay)

                    self.logger.warning(
                        f"{model} busy (attempt {attempt + 1}"
                        f"/{self.max_attempts}), retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)

        raise last_error

    # --------------------------------------------------

    @staticmethod
    def daily_quota_limit(error):
        """
        The per-day request cap when THAT is what refused the call,
        otherwise None.

        A 429 can mean either "too fast, slow down" (worth retrying
        within the run) or "you have used today's allowance" (not worth
        retrying at all). Google distinguishes them by quota id, so key
        off that rather than the status code.
        """

        text = str(error)

        if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
            return None

        if "PerDay" not in text and "per day" not in text.lower():
            return None

        for pattern in (r"'quotaValue':\s*'(\d+)'", r"limit:\s*(\d+)"):
            found = re.search(pattern, text)
            if found:
                return int(found.group(1))

        return "unknown"

    # --------------------------------------------------

    @staticmethod
    def is_model_unavailable(error):
        """
        True when the model name itself is the problem - retired, or not
        released to this key - rather than a temporary condition.
        """

        text = str(error)

        if "404" not in text and "NOT_FOUND" not in text:
            return False

        return any(
            marker in text
            for marker in ("no longer available", "not available",
                           "is not found", "not supported")
        )

    # --------------------------------------------------

    @staticmethod
    def retry_after(error):
        """Seconds the server asked us to wait, when it said so."""

        found = re.search(r"'retryDelay':\s*'(\d+)s'", str(error))

        return int(found.group(1)) if found else None

    # --------------------------------------------------

    @staticmethod
    def is_transient(error):
        """
        True for errors worth retrying: server overload (503),
        per-minute rate limiting (429) and gateway/timeout hiccups.

        Per-day quota exhaustion is deliberately excluded - see
        daily_quota_limit, which is checked first.
        """

        text = str(error)

        return any(
            marker in text
            for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                           "500", "502", "504", "deadline")
        )


    