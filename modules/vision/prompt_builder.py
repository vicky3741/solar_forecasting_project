"""
=========================================================
Solar Forecasting Project
Prompt Builder
=========================================================
Creates prompts for Gemini Vision API.
Supports both image and video analysis.
=========================================================
"""


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

Extract the following weather information.

Return ONLY valid JSON.

Do NOT explain.
Do NOT write markdown.
Do NOT include extra text.

JSON format:

{
    "cloud_coverage": 0,
    "cloud_density": "",
    "cloud_motion_direction": "",
    "cloud_motion_speed": "",
    "weather_condition": "",
    "rain_probability": 0,
    "sun_visibility": 0,
    "irradiance_trend": "",
    "expected_generation_trend": "",
    "confidence": 0.0
}
"""

        return prompt

    # =====================================================
    # VIDEO PROMPT
    # =====================================================

    def video_prompt(self):

        prompt = """
You are an expert meteorologist and solar forecasting specialist.

You are given a chronological sequence of frames extracted from a Windy weather animation.

IMPORTANT:
These frames belong to the SAME weather animation and are arranged in chronological order.

Analyze the complete sequence instead of each frame independently.

Observe the following:

1. Cloud coverage
2. Cloud density
3. Cloud movement direction
4. Cloud movement speed
5. Weather condition
6. Rain probability
7. Sun visibility
8. Expected irradiance trend
9. Expected solar generation trend

Infer the overall weather trend from the sequence.

Return ONLY valid JSON.

Do NOT explain.
Do NOT write markdown.
Do NOT include extra text.

JSON format:

{
    "cloud_coverage": 0,
    "cloud_density": "",
    "cloud_motion_direction": "",
    "cloud_motion_speed": "",
    "weather_condition": "",
    "rain_probability": 0,
    "sun_visibility": 0,
    "irradiance_trend": "",
    "expected_generation_trend": "",
    "confidence": 0.0
}
"""

        return prompt