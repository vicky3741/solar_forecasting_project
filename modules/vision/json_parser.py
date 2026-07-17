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

    def parse(self, response):

        response = self.clean_response(response)

        try:

            return json.loads(response)

        except json.JSONDecodeError as error:

            raise ValueError(
                f"Invalid JSON returned by Gemini:\n{error}"
            )