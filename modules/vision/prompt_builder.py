"""
=========================================================
Solar Forecasting Project
Prompt Builder
=========================================================
Creates prompts for Gemini Vision API.
Supports both image and video analysis.

PROMPT_VERSION is stored alongside cached vision results -
bumping it invalidates old caches so videos get re-analyzed
with the new prompt.
=========================================================
"""

PROMPT_VERSION = 2


class PromptBuilder:

    def __init__(self):
        pass

    # =====================================================
    # IMAGE PROMPT
    # =====================================================

    def image_prompt(self):

        prompt = """
You are an expert meteorologist and solar forecasting specialist.

Analyze the given Windy weather image carefully.
The solar plant is located at the CENTER of the map.

Extract the following weather information.

Return ONLY valid JSON.

Do NOT explain.
Do NOT write markdown.
Do NOT include extra text.

JSON format:

{
    "cloud_coverage_pct": 0,
    "cloud_density": "",
    "cloud_motion_direction": "",
    "cloud_motion_speed": "",
    "weather_condition": "",
    "rain_probability_pct": 0,
    "sun_visibility": 0,
    "irradiance_trend": "",
    "confidence": 0.0
}
"""

        return prompt

    # =====================================================
    # VIDEO PROMPT
    # =====================================================

    def video_prompt(self):

        prompt = """
You are an expert meteorologist supporting a solar power plant generation forecast.

You are given a chronological sequence of frames extracted from a Windy satellite/cloud animation.
The solar plant is located at the CENTER of the map.

IMPORTANT:
These frames belong to the SAME weather animation and are arranged in chronological order.
Analyze the sequence as a whole to judge cloud MOVEMENT over time - do not analyze frames independently.

Report the following, always relative to the CENTER of the map where the plant is:

1. Cloud coverage near the center right now, as a percentage (0-100).
2. Cloud density: one of "none", "thin", "moderate", "thick".
3. Cloud movement direction: the compass direction the clouds are moving TOWARD - one of N, NE, E, SE, S, SW, W, NW.
4. Cloud movement speed: one of "slow", "moderate", "fast".
5. Whether the clouds are moving toward the center (true) or away from / past it (false).
6. The expected change at the center: one of "clearing", "clouding", "stable".
7. Estimated minutes from now until that change meaningfully affects the center (null if stable).
8. Expected irradiance trend at the center over the NEXT 2 HOURS: one of "increasing", "decreasing", "stable".
9. Expected irradiance trend at the center 2 TO 4 HOURS from now: one of "increasing", "decreasing", "stable".
10. Rain probability near the center, as a percentage (0-100).
11. Your confidence in this analysis, from 0.0 to 1.0.

Return ONLY valid JSON.

Do NOT explain.
Do NOT write markdown.
Do NOT include extra text.

JSON format:

{
    "cloud_coverage_pct": 0,
    "cloud_density": "",
    "cloud_motion_direction": "",
    "cloud_motion_speed": "",
    "moving_toward_plant": false,
    "expected_change": "",
    "minutes_until_change": null,
    "trend_next_2h": "",
    "trend_2h_to_4h": "",
    "rain_probability_pct": 0,
    "confidence": 0.0
}
"""

        return prompt
