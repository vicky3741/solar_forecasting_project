"""
=========================================================
Solar Forecasting Project
Windy Premium Login (one-time, interactive)
=========================================================
Opens a VISIBLE browser at windy.com so you can sign in to
the Windy Premium account by hand, then saves the resulting
browser session to a local file. Every later headless
capture reuses that session and therefore gets premium
layers without ever handling your password.

Run it once:

    python -m modules.capture.windy_login

You type the credentials into the real Windy page yourself
- nothing in this project reads, stores or transmits your
password. Only the resulting session cookies are saved.

SECURITY: the session file is the equivalent of being
logged in. It is gitignored, and it must stay that way -
Team 1 committed theirs (windy_login.json) to a public
repo, which exposed their premium account to anyone who
found it. Never commit this file, and re-run this script
if you ever log out of Windy elsewhere.
=========================================================
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from config.config import settings
from utils.logger import get_logger


LOGIN_URL = "https://www.windy.com/"


def session_path():

    capture = settings.get("windy_capture", {})

    return Path(capture.get("session_file", "windy_session.json"))


def main():

    logger = get_logger()

    target = session_path()

    print("=" * 62)
    print("Windy premium login - one time setup")
    print("=" * 62)
    print()
    print("A browser window will open at windy.com.")
    print()
    print("  1. Sign in to your Windy Premium account in that window.")
    print("     (Type your credentials directly into Windy's own page -")
    print("      this project never sees or stores your password.)")
    print("  2. Once you are logged in, come back here.")
    print("  3. Press ENTER to save the session.")
    print()
    print(f"The session will be written to: {target}")
    print("Keep that file private - it is gitignored and must stay so.")
    print()

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(headless=False)

        context = browser.new_context(
            viewport={"width": 1600, "height": 1000}
        )

        page = context.new_page()
        page.goto(LOGIN_URL, wait_until="load", timeout=90000)

        try:
            input("Press ENTER here once you have finished logging in... ")
        except EOFError:
            print(
                "\nNo interactive console available - run this from a normal "
                "terminal, not from an automated task."
            )
            browser.close()
            return 1

        context.storage_state(path=str(target))

        browser.close()

    logger.info(f"Windy session saved: {target}")

    print()
    print(f"Saved. Captures will now reuse this session from {target}.")
    print("If premium layers stop working, just run this script again.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
