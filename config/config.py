"""
============================================================
Solar Forecasting Project
Configuration Loader
============================================================
Loads the global settings.yaml file and provides
configuration access to the entire project.

MULTI-PLANT (2026-08-08)
------------------------
The project now schedules three plants - Sirmour (Madhya
Pradesh, 5.1 MW), Kasipet (Telangana, 15 MW) and Bhupalpally
(Telangana, 10 MW) - from this one codebase, but each plant
runs completely independently: its own data, its own model
state, its own outputs, its own S3 prefixes, its own log and
its own automation service.

How that works, and why it was done this way:

  settings.yaml stays EXACTLY what it always was - Sirmour's
  fully tuned configuration - and is the base layer. A plant
  overlay in config/plants/<key>.yaml is then deep-merged on
  top of it, selected by the SOLAR_PLANT environment
  variable (default "sirmour").

  config/plants/sirmour.yaml is therefore almost empty: the
  base IS Sirmour. With SOLAR_PLANT unset, the resolved
  settings are byte-identical to the pre-multi-plant ones, so
  the live Sirmour automation cannot be disturbed by anything
  done for the new plants. Every new plant-specific knob added
  to settings.yaml carries Sirmour's existing behaviour as its
  default for the same reason.

  Nothing else in the codebase had to learn about plants:
  every module already reads this single `settings` object, so
  pointing that object at a different plant re-points the
  whole pipeline.

Run a different plant by setting the environment variable:

    SOLAR_PLANT=kasipet python -m modules.orchestrator.pipeline

(the per-plant launchers in automation/ do this for you).
============================================================
"""
import copy
import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import yaml


# The plant this process is running for. Every path, prefix and
# tuned parameter below resolves against it.
DEFAULT_PLANT = "sirmour"

PLANT_KEY = (os.getenv("SOLAR_PLANT") or DEFAULT_PLANT).strip().lower()


def deep_merge(base, overlay):
    """
    Returns `base` with `overlay` merged into it, recursing into
    nested dicts so an overlay can change one key of a block
    without restating the rest of it.

    A non-dict value in the overlay replaces the base value
    outright - lists included. That is deliberate: forecast
    run_times differ per plant (Telangana adds a 17:15 run), and
    merging those element-wise would silently produce a list
    belonging to neither plant.
    """

    merged = copy.deepcopy(base)

    for key, value in (overlay or {}).items():

        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


class ConfigLoader:
    """
    Loads and stores project configuration.
    """

    def __init__(self, plant_key=None):

        self.plant_key = (plant_key or PLANT_KEY).strip().lower()

        # Root directory of the project
        self.project_root = Path(__file__).resolve().parent.parent

        # Path to settings.yaml
        self.config_file = self.project_root / "config" / "settings.yaml"

        # Path to this plant's overlay
        self.plant_file = (
            self.project_root / "config" / "plants" / f"{self.plant_key}.yaml"
        )

        # Load configuration
        self.settings = self.load_settings()

    def read_yaml(self, path, required):
        """
        Reads one YAML file. A missing OPTIONAL file resolves to an
        empty overlay rather than an error, so a plant whose
        settings are all inherited needs no file at all.
        """

        if not path.exists():

            if required:
                raise FileNotFoundError(
                    f"Configuration file not found:\n{path}"
                )

            return {}

        with open(path, "r", encoding="utf-8") as file:

            return yaml.safe_load(file) or {}

    def load_settings(self):
        """
        Read settings.yaml (Sirmour's base configuration), then
        merge this plant's overlay over it.
        """

        base = self.read_yaml(self.config_file, required=True)

        # An unknown SOLAR_PLANT must fail loudly. Silently falling
        # back to Sirmour would point a Kasipet automation run at
        # Sirmour's data, outputs and S3 prefixes - the exact
        # cross-plant contamination this whole layer exists to stop.
        known = self.known_plants()

        if self.plant_key not in known:
            raise ValueError(
                f"Unknown plant '{self.plant_key}'. Set SOLAR_PLANT to one of: "
                f"{', '.join(known)}"
            )

        overlay = self.read_yaml(self.plant_file, required=False)

        settings = deep_merge(base, overlay)

        settings.setdefault("plant", {})["key"] = self.plant_key

        return settings

    def known_plants(self):
        """
        Plant keys that have an overlay file, plus the default -
        Sirmour stays valid even though its overlay is nearly empty.
        """

        folder = self.project_root / "config" / "plants"

        found = {DEFAULT_PLANT}

        if folder.exists():
            found.update(path.stem.lower() for path in folder.glob("*.yaml"))

        return sorted(found)


# Global configuration object
config = ConfigLoader()

# Shortcut used throughout the project
settings = config.settings

# Gemini key, PER PLANT. This one has to be split three ways: the free
# tier caps requests per day PER MODEL, and the three plants together now
# want 7 + 8 + 8 = 23 vision calls a day against a cap of 20. On one
# shared key the third plant of the day would quietly lose its cloud
# signal (vision failures degrade silently to no-vision), so each plant
# gets its own:
#     GOOGLE_API_KEY_SIRMOUR / _KASIPET / _BHUPALPALLY
# The plain GOOGLE_API_KEY stays the fallback for any plant without one,
# so an existing single-key setup keeps working untouched.
settings["vision"]["api_key"] = (
    os.getenv(f"GOOGLE_API_KEY_{PLANT_KEY.upper()}")
    or os.getenv("GOOGLE_API_KEY")
)
# ChatGPT direct OR any OpenAI-compatible gateway (OpenRouter). Whichever
# key is set works - pair it with the matching openai_base_url in settings.
# A missing key is left as None and only raised when THAT provider is
# actually selected, so a Gemini-only setup needs no OPENAI_API_KEY.
settings["vision"]["openai_api_key"] = (
    os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
)

# Windy key: ONE key, shared by all three plants, by request 2026-08-08.
# Unlike Gemini this does not need splitting - the key only unlocks the
# premium overlays on the embed page we screen-record, and three captures
# a day-part is nowhere near a rate limit. No key at all still works; the
# capture falls back to Windy's free public embed.
settings["windy_capture"]["api_key"] = os.getenv("WINDY_API_KEY")
