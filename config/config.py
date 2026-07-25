"""
============================================================
Solar Forecasting Project
Configuration Loader
============================================================
Loads the global settings.yaml file and provides
configuration access to the entire project.
============================================================
"""
import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import yaml


class ConfigLoader:
    """
    Loads and stores project configuration.
    """

    def __init__(self):

        # Root directory of the project
        self.project_root = Path(__file__).resolve().parent.parent

        # Path to settings.yaml
        self.config_file = self.project_root / "config" / "settings.yaml"

        # Load configuration
        self.settings = self.load_settings()

    def load_settings(self):
        """
        Read settings.yaml file.
        """

        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found:\n{self.config_file}"
            )

        with open(self.config_file, "r", encoding="utf-8") as file:

            return yaml.safe_load(file)


# Global configuration object
config = ConfigLoader()

# Shortcut used throughout the project
settings = config.settings
# Each provider's key is loaded independently. A missing key is left
# as None and only raised when THAT provider is actually selected and
# used - so a Gemini-only setup needs no OPENAI_API_KEY and vice versa.
settings["vision"]["api_key"] = os.getenv("GOOGLE_API_KEY")          # Gemini
# ChatGPT direct OR any OpenAI-compatible gateway (OpenRouter). Whichever
# key is set works - pair it with the matching openai_base_url in settings.
settings["vision"]["openai_api_key"] = (
    os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
)
settings["windy_capture"]["api_key"] = os.getenv("WINDY_API_KEY")