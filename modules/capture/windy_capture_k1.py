"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Windy capture, Team 1's way
=========================================================
A port of the capture in test_multi_image.py from
github.com/Kushal70-51/Windy-Project-3.

SEPARATE FROM modules/capture/windy_capture.py ON PURPOSE.
That one feeds the LIVE Sirmour automation - seven runs a
day, three plants, and a Gemini vision signal tuned against
clips recorded at zoom 8 through the embed. Changing its zoom
or its page would silently change what production sees. This
module writes to its own folder and touches nothing.

WHAT IS DIFFERENT FROM OURS, AND WHY IT MATTERS
-----------------------------------------------
                       ours                theirs
  page                 embed.windy.com     www.windy.com
  viewport             1280 x 720          1600 x 1000
  zoom                 8                   11
  ground across frame  ~712 km             ~111 km
  layers               solarpower, clouds  satellite, wind,
                                           solarpower, clouds, rain
  clip length          20 s                ~10 s of clean footage

The zoom is the substantive one. At zoom 8 a frame spans most
of northern India; at zoom 11 it spans the weather that can
actually reach this plant within a forecast block.

PREMIUM IS REQUIRED HERE, UNLIKE THE EMBED
------------------------------------------
www.windy.com serves the full application, and several of
these overlays are premium-only. Their script loads a saved
session with Playwright's `storage_state`. Ours does the same
when windy_session.json exists - created by
`python -m modules.capture.windy_login`, which is an
interactive login the USER runs.

Without that file this module still runs, but expect the
premium layers to render empty or fall back. It logs which
state it is in rather than producing quietly worthless
screenshots.

HONEST LIMIT ON FIDELITY
------------------------
Their still-screenshot path is reproduced exactly: URL,
viewport, zoom, waits, layer list. Their ANIMATION path also
drives windy.com's own timeline UI - dismiss overlay, seek to
"1h ago", press play, choose slow speed, enable forecast mode -
using selectors internal to that page. Those selectors are not
reproduced here because they are not in the file, and guessing
at them would produce a capture that looks right and records
the wrong thing. The animation here records the layer with the
timeline left at its default, and says so in the log.
=========================================================
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# --- must happen before `playwright` is imported (see windy_capture.py) ---
if os.name == "nt":
    _DEFAULT_BROWSERS_PATH = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "ms-playwright"
    )
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_DEFAULT_BROWSERS_PATH))

from playwright.sync_api import sync_playwright  # noqa: E402

from config.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402


# --- their constants ---
VIEWPORT_WIDTH = 1600
VIEWPORT_HEIGHT = 1000
ZOOM_LEVEL = 11
LAYERS = ["satellite", "wind", "solarpower", "clouds", "rain"]
ANIMATION_LAYER = "satellite"
ANIMATION_RECORD_SECONDS = 8

# their waits, in milliseconds
WAIT_AFTER_NAVIGATION = 6000
WAIT_AFTER_POPUP = 2000
WAIT_AFTER_PICKER = 1500
NAVIGATION_TIMEOUT = 60000
VIDEO_LOAD_WAIT = 15000

# www.windy.com DRAWS ITS MAP WITH WebGL, unlike the embed. The
# low-memory arg set used for the embed capture (--disable-gpu plus
# --disable-software-rasterizer) leaves headless Chromium with no GL
# backend at all, and Windy then renders a flat grey page carrying
# "Failed to initialize WebGL Overlay".
#
# That failure is silent to everything downstream: the screenshot is a
# valid PNG, the capture reports 5/5 layers, and the features come out
# as brightness 146.0 with a standard deviation of exactly 0.0 for
# every layer - a solid grey rectangle measured with great precision.
# Measured 2026-08-09.
#
# SwiftShader is Chromium's software GL implementation, so the map
# renders without a GPU. It costs CPU and memory, which matters on the
# ~900 MB EC2 box - profile before scheduling this there.
WEBGL_BROWSER_ARGS = [
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]

LOW_MEMORY_BROWSER_ARGS = WEBGL_BROWSER_ARGS


class WindyCaptureK1:

    def __init__(self):

        self.logger = get_logger()

        plant = settings["plant"]
        capture = settings["windy_capture"]

        self.lat = plant["latitude"]
        self.lon = plant["longitude"]
        self.plant_tag = plant.get("code", "SIRMOUR")

        self.headless = capture.get("headless", True)
        self.session_file = Path(capture.get("session_file", "windy_session.json"))

        k1 = settings.get("windy_capture_k1", {})

        self.zoom = k1.get("zoom", ZOOM_LEVEL)
        self.layers = k1.get("layers", LAYERS)
        self.record_seconds = k1.get("record_seconds", ANIMATION_RECORD_SECONDS)

        self.screenshot_dir = Path(
            k1.get("screenshot_dir", "data/windy/k1_screenshots")
        )
        self.video_dir = Path(k1.get("video_dir", "data/windy/k1_videos"))

    # --------------------------------------------------

    def page_url(self, overlay):
        """
        Their URL template, unchanged:

            https://www.windy.com/{lat}/{lon}?{overlay},{lat},{lon},{zoom},p:cities

        The path centres the map; the query sets the overlay and zoom.
        """

        return (
            f"https://www.windy.com/{self.lat}/{self.lon}"
            f"?{overlay},{self.lat},{self.lon},{self.zoom},p:cities"
        )

    # --------------------------------------------------

    def context_options(self, **extra):

        options = {
            "viewport": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        }

        if self.session_file.exists():
            options["storage_state"] = str(self.session_file)

        options.update(extra)

        return options

    # --------------------------------------------------

    def dismiss_popups(self, page):
        """
        windy.com opens with a cookie/consent banner and sometimes a
        promo. Both sit over the map and would appear in the
        screenshot.

        Only ever chooses the REJECT / dismiss option, never "accept
        all" - this is an automated visit and there is no reason to
        take non-essential cookies.
        """

        candidates = [
            "button:has-text('Reject')",
            "button:has-text('Decline')",
            "#accept-choices-reject",
            ".cookie-consent button.reject",
            "[aria-label='Close']",
            "button.close",
        ]

        for selector in candidates:

            try:
                element = page.locator(selector).first

                if element.is_visible(timeout=1200):
                    element.click(timeout=1200)
                    page.wait_for_timeout(WAIT_AFTER_POPUP)
                    return True

            except Exception:
                continue

        return False

    # --------------------------------------------------

    def set_picker_point(self, page):
        """
        Right-click the map centre, then CLICK "Show weather picker".

        Both steps are required. The right-click only opens Windy's
        context menu, and that menu then sits directly over the plant -
        which is exactly the area the ROI box measures. Leaving it open
        put a grey menu panel in the middle of every screenshot on
        2026-08-09.

        Clicking the item pins the picker and dismisses the menu, and
        the pinned picker is also what puts the value in the DOM
        (class "picker-change-metric", e.g. "0 W/m²").
        """

        try:
            page.mouse.click(
                VIEWPORT_WIDTH // 2, VIEWPORT_HEIGHT // 2, button="right"
            )
            page.wait_for_timeout(WAIT_AFTER_PICKER)

        except Exception as error:
            self.logger.warning(f"Picker right-click failed ({error})")
            return False

        for selector in ("text=Show weather picker", "text=Weather picker"):

            try:
                page.locator(selector).first.click(timeout=3000)
                page.wait_for_timeout(WAIT_AFTER_PICKER)
                return True

            except Exception:
                continue

        # The menu opened but the item was not found. Close it, or it
        # will cover the plant in the screenshot.
        self.logger.warning(
            "Could not click 'Show weather picker' - dismissing the menu so "
            "it does not cover the plant"
        )

        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
        except Exception:
            pass

        return False

    # --------------------------------------------------

    def read_picker_value(self, page):
        """
        The pinned picker's readout for the current overlay, as text
        (e.g. "0 W/m²"), or None. This is the exact figure at the plant
        - no colour inversion, no pixel statistics.
        """

        try:
            element = page.locator(".picker-change-metric").first

            if element.is_visible(timeout=2000):
                return element.inner_text().strip()

        except Exception:
            pass

        return None

    # --------------------------------------------------

    def capture_layers(self, run_time=None):
        """
        One screenshot per layer, their sequence:
        navigate -> wait 6s -> dismiss popups -> set picker -> shoot.
        """

        run_time = run_time or datetime.now()

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.plant_tag}_{run_time.strftime('%Y-%m-%d_%H-%M-%S')}"

        if not self.session_file.exists():
            self.logger.warning(
                f"{self.session_file} not found - running WITHOUT the Windy "
                "premium session. Premium overlays may render empty. Create "
                "it with: python -m modules.capture.windy_login"
            )

        captured = {}
        readouts = {}

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=self.headless, args=LOW_MEMORY_BROWSER_ARGS
            )
            context = browser.new_context(**self.context_options())
            page = context.new_page()

            for index, layer in enumerate(self.layers):

                try:
                    page.goto(
                        self.page_url(layer),
                        wait_until="load",
                        timeout=NAVIGATION_TIMEOUT,
                    )
                    page.wait_for_timeout(WAIT_AFTER_NAVIGATION)

                    # Banners only appear on the first load of a session.
                    if index == 0:
                        self.dismiss_popups(page)

                    self.set_picker_point(page)

                    value = self.read_picker_value(page)

                    path = self.screenshot_dir / f"{stem}_{layer}.png"
                    page.screenshot(path=str(path))

                    captured[layer] = path
                    readouts[layer] = value

                    self.logger.info(
                        f"k1 layer captured: {path.name}"
                        + (f"  picker={value}" if value else "")
                    )

                except Exception as error:
                    self.logger.warning(
                        f"k1 layer '{layer}' failed ({error}) - skipping"
                    )

            context.close()
            browser.close()

        if readouts:

            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

            import json

            (self.screenshot_dir / f"{stem}_picker.json").write_text(
                json.dumps(readouts, indent=2), encoding="utf-8"
            )

        return captured

    # --------------------------------------------------

    def capture_animation(self, run_time=None):
        """
        The satellite animation, at their zoom and viewport.

        See the module docstring: their timeline choreography (seek to
        "1h ago", slow speed, forecast mode) uses windy.com selectors
        that are not in the file we read, so this records the layer
        with the timeline at its default rather than guessing at them.
        """

        run_time = run_time or datetime.now()

        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "k1 animation: recording with the timeline at its DEFAULT - "
            "their seek/play/speed sequence is not reproduced (see module "
            "docstring)"
        )

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=self.headless, args=LOW_MEMORY_BROWSER_ARGS
            )

            context = browser.new_context(
                **self.context_options(
                    record_video_dir=str(self.video_dir),
                    record_video_size={
                        "width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT
                    },
                )
            )

            page = context.new_page()

            page.goto(
                self.page_url(ANIMATION_LAYER),
                wait_until="load",
                timeout=NAVIGATION_TIMEOUT,
            )
            page.wait_for_timeout(VIDEO_LOAD_WAIT)

            self.dismiss_popups(page)
            self.set_picker_point(page)

            page.wait_for_timeout(self.record_seconds * 1000)

            video_path = page.video.path() if page.video else None

            context.close()
            browser.close()

        if video_path is None:
            raise RuntimeError("Playwright produced no video")

        final = self.video_dir / (
            f"{self.plant_tag}_{run_time.strftime('%Y-%m-%d_%H-%M-%S')}.webm"
        )

        Path(video_path).replace(final)

        size = final.stat().st_size

        if size < 100_000:
            final.unlink(missing_ok=True)
            raise RuntimeError(
                f"Recording came out at {size} bytes - refusing to keep it"
            )

        self.logger.info(f"k1 animation saved: {final}")

        return final


# --------------------------------------------------

def main():

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Windy capture, Team 1's way (zoom 11, 5 layers)"
    )
    parser.add_argument("--no-video", action="store_true")

    args = parser.parse_args()

    capture = WindyCaptureK1()

    run_time = datetime.now()

    screenshots = capture.capture_layers(run_time)

    print(f"layers captured: {len(screenshots)}/{len(capture.layers)}")

    if not args.no_video:
        try:
            video = capture.capture_animation(run_time)
            print(f"animation: {video}")
        except Exception as error:
            print(f"animation failed: {error}")

    if screenshots:

        from modules.vision.windy_layer_features import extract_layer_features

        print("\nfeatures:")
        print(json.dumps(extract_layer_features(screenshots), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
