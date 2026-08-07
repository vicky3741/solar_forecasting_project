"""
=========================================================
Solar Forecasting Project
API Key Check
=========================================================
Shows which key each plant would actually use, WITHOUT
printing any key - only its last 4 characters, enough to
tell two keys apart and useless to anyone who sees the
output.

Worth having because the failure mode is silent. A missing
or exhausted Gemini key does not crash a run: vision quietly
degrades to no-vision and the schedule still publishes,
looking normal. This is the cheap way to confirm all three
plants are set up before that happens.

Run:  python -m tests.test_api_keys
=========================================================
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

PLANTS = ["sirmour", "kasipet", "bhupalpally"]


def show(value):
    """Last 4 characters of a secret, or a clear 'not set'."""

    if not value:
        return "NOT SET"

    return f"...{value[-4:]}  ({len(value)} chars)"


def main():

    print("=" * 68)
    print("API KEYS - which key each plant resolves (values are masked)")
    print("=" * 68)

    shared_google = os.getenv("GOOGLE_API_KEY")
    windy = os.getenv("WINDY_API_KEY")

    print()
    print("GEMINI (vision) - one key PER PLANT")
    print("  The free tier caps requests per day per model. Sirmour needs 7 a")
    print("  day, Kasipet and Bhupalpally 8 each = 23 against a cap of 20, so")
    print("  a single shared key runs out and the last plant of the day loses")
    print("  its cloud signal.")
    print()

    problems = []
    seen = {}

    for plant in PLANTS:

        own = os.getenv(f"GOOGLE_API_KEY_{plant.upper()}")
        used = own or shared_google

        if own:
            source = f"GOOGLE_API_KEY_{plant.upper()}"
        elif shared_google:
            source = "GOOGLE_API_KEY (shared fallback)"
        else:
            source = "none"

        print(f"  {plant:12s} {show(used):26s} from {source}")

        if not used:
            problems.append(f"{plant} has no Gemini key at all - it will run "
                            "with no cloud signal, every run")
        else:
            seen.setdefault(used, []).append(plant)

    for _, sharers in seen.items():
        if len(sharers) > 1:
            problems.append(
                f"{' and '.join(sharers)} share one Gemini key - they will "
                "compete for the same 20/day quota"
            )

    print()
    print("WINDY (satellite capture) - ONE key for all three")
    print("  It only unlocks premium overlays on the embed page we record, so")
    print("  there is no quota to split. No key at all is fine too - the")
    print("  capture falls back to the free public embed.")
    print()
    print(f"  WINDY_API_KEY  {show(windy)}")

    print()
    print("AWS (S3 bucket) - one account, all three plants")
    print()
    print(f"  AWS_ACCESS_KEY_ID      {show(os.getenv('AWS_ACCESS_KEY_ID'))}")
    print(f"  AWS_SECRET_ACCESS_KEY  "
          f"{show(os.getenv('AWS_SECRET_ACCESS_KEY'))}")

    if not os.getenv("AWS_ACCESS_KEY_ID"):
        problems.append("no AWS key - no plant can pull meter data or push "
                        "its schedule to the bucket")

    print()
    print("-" * 68)

    if problems:
        print("PROBLEMS")
        for problem in problems:
            print(f"  ! {problem}")
        print()
        print("Keys live in .env in the project root (gitignored - never "
              "commit it).")
        return 1

    print("All three plants have their own Gemini key. Nothing to fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
