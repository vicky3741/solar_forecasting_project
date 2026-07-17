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
settings["vision"]["api_key"] = os.getenv("GOOGLE_API_KEY")