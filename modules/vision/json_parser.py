"""
=========================================================
Solar Forecasting Project
JSON Parser
=========================================================
Converts Gemini response into validated Python dictionary.
=========================================================
"""

import json


class JSONParser:

    def __init__(self):
        pass

    # --------------------------------------------------

    def clean_response(self, response):

        response = response.replace("```json", "")
        response = response.replace("```", "")

        return response.strip()

    # --------------------------------------------------

    def extract_json_block(self, response):
        """
        Falls back to the outermost {...} block when the model
        wraps the JSON in extra prose despite instructions.
        """

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return response

        return response[start:end + 1]

    # --------------------------------------------------

    def parse(self, response):

        response = self.clean_response(response)

        try:
            return json.loads(response)

        except json.JSONDecodeError:
            pass

        try:
            return json.loads(self.extract_json_block(response))

        except json.JSONDecodeError as error:

            raise ValueError(
                f"Invalid JSON returned by Gemini:\n{error}"
            )