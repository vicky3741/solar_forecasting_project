"""
=========================================================
Solar Forecasting Project
Windy Value Scraper
=========================================================
Reads Windy's forecast values AT THE PLANT COORDINATES as
NUMBERS, by driving the embed's own JavaScript API - instead
of screen-recording the map and guessing the numbers back
out of coloured pixels.

WHY THIS EXISTS
---------------
The pixel route (record video -> OpenCV brightness/hue/
saturation -> feed to a model) was measured on 2026-08-09 and
it does not work:

  * Windy paints a value as a COLOUR. Averaging the colour of
    a whole 1280x720 frame averages thousands of km of map
    that is not the plant, and the map's own UI chrome too.
  * Averaging hue arithmetically is mathematically wrong -
    hue is circular, so mean(350 deg, 10 deg) comes out 180 deg,
    the opposite colour.
  * One clip = one moment, so every forecast block downstream
    got the SAME feature row. In the reference features log
    every column was identical across blocks 47-54 and the
    resulting forecast collapsed to 1.9918 * sin(solar
    elevation) - i.e. pure clear-sky geometry, with the
    "cloud features" contributing nothing at all.

The embed already holds the exact number. Live checks against
https://embed.windy.com/embed2.html for Sirmour:

    W.store.get('overlay')  -> 'solarpower'
    W.store.get('product')  -> 'ecmwf'
    <big> element           -> '492 W/m2'
    W.store.set('timestamp', +3h)
    <big> element           -> '209 W/m2'   (path 2026080909 -> 2026080912)

So the timeline is programmable and the readout follows it.
That is an exact figure at the plant, free, with no vision
quota and no OpenCV in the path.

WHAT THE FREE TIER ACTUALLY GIVES (measured, not assumed)
---------------------------------------------------------
W.products.ecmwf.calendar.timestamps held 63 entries, EVERY
step exactly 180 minutes. For 2026-08-09 that is 08:30, 11:30,
14:30, 17:30, 20:30, 23:30 IST - only THREE useful daylight
points a day. W.subscription.getTier() returned null and
hasAny() returned false, and the product carries a separate
`intervalPremium`, so finer steps are a paid feature.

Keep that in proportion when weighting this signal: Windy's
solarpower IS ECMWF, the same model modules/weather/open_meteo.py
already delivers HOURLY (about 13 daylight points) and with a
searchable archive. This scraper is therefore a cross-check and
an extra-layer source (clouds/rain/wind/lclouds/...), not a
replacement for the hourly weather feed.

THE STALENESS GUARD
-------------------
Stepping the timeline only changes the readout when Windy
actually loads a new data file for that step. The 5-minutely
EUMETSAT satellite layer did NOT reload frame by frame in the
embed - all 24 steps returned an identical 241 K / raw 147.

Emitting those 24 identical numbers as if they were a
measured time series is exactly the failure that made the
reference features log useless. So `scrape_overlay` refuses to
pretend: if every sampled step comes back identical it marks
the series `stale: True`, and callers must drop it rather than
feed a flat line into a forecast.
=========================================================
"""

import json
import os
import re
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


# Reads "492 W/m²", "241 K", "18 °C", "-3.4 mm" -> (value, unit).
_VALUE_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(.*)")

# The wind layer's readout carries a compass bearing after the unit,
# separated by the glyph Windy uses for its arrow icon: 8 kt"W. Split
# it out so `unit` stays a real unit and the direction is its own
# field, instead of a "kt\"W" string nothing downstream can use.
_WIND_DIRECTION_PATTERN = re.compile(
    r"[\"″”]\s*([NSEW]{1,3})\s*$"
)


def parse_readout(text):
    """
    Splits Windy's picker readout into (value, unit, direction).

    `direction` is None for every layer except wind. Returns
    (None, raw_text, None) when there is no number in the text -
    an unreadable readout must never silently become 0.0, because
    a fabricated zero at midday is indistinguishable from a real
    one downstream.
    """

    if not text:
        return None, "", None

    match = _VALUE_PATTERN.search(text.strip())

    if not match:
        return None, text.strip(), None

    number = match.group(1).replace(",", ".")
    remainder = match.group(2).strip()

    direction = None
    bearing = _WIND_DIRECTION_PATTERN.search(remainder)

    if bearing:
        direction = bearing.group(1)
        remainder = remainder[:bearing.start()].strip()

    try:
        return float(number), remainder, direction
    except ValueError:
        return None, text.strip(), None


class WindyScraper:

    # Layers worth pulling for a solar forecast. solarpower is the
    # headline one (Windy's modelled irradiance); the cloud decks are
    # split low/mid/high because thin high cloud and thick low cloud
    # attenuate very differently, which a single "clouds" number hides.
    DEFAULT_OVERLAYS = [
        "solarpower",
        "clouds",
        "lclouds",
        "mclouds",
        "hclouds",
        "rain",
        "wind",
    ]

    def __init__(self, overlays=None):

        self.logger = get_logger()

        plant = settings["plant"]
        capture = settings["windy_capture"]

        self.lat = plant["latitude"]
        self.lon = plant["longitude"]
        self.plant_tag = plant.get("code", "SIRMOUR")
        self.timezone = plant.get("timezone", "Asia/Kolkata")

        self.api_key = capture.get("api_key")
        self.zoom = capture.get("zoom", 8)
        self.headless = capture.get("headless", True)
        self.session_file = Path(capture.get("session_file", "windy_session.json"))

        scrape = settings.get("windy_scrape", {})

        self.overlays = overlays or scrape.get("overlays", self.DEFAULT_OVERLAYS)
        self.output_dir = Path(scrape.get("output_dir", "data/windy/values"))

        # WHICH PAGE TO READ, AND WHY IT MATTERS
        #
        #   embed  embed.windy.com. No login needed, but ALWAYS free
        #          tier: 3-hourly ECMWF, ~3 usable daylight points a
        #          day. Measured 2026-08-09, and it cannot be improved
        #          - the embed does no premium check at all, and the
        #          session's localStorage holds no auth token to carry
        #          across (all 36 keys are settings_*).
        #   www    www.windy.com, the full application. With the saved
        #          premium session this serves HOURLY ECMWF - 60-minute
        #          steps, 143 of them, about 13 daylight points a day.
        #
        # "auto" picks www when a premium session exists, embed
        # otherwise, so this does the best available thing without
        # needing to be reconfigured when the session appears.
        self.source = scrape.get("source", "auto")

        if self.source == "auto":
            self.source = "www" if self.session_file.exists() else "embed"

        self.zoom_www = scrape.get("zoom_www", 11)

        # How long to let Windy fetch and render a step before reading
        # the value back. Measured at ~600-900 ms on a warm page; the
        # default leaves headroom for the EC2 box.
        self.step_wait_ms = scrape.get("step_wait_ms", 900)

        # How many forecast steps to walk per overlay. The free ECMWF
        # calendar is 3-hourly over 10 days (63 steps); a forecast only
        # ever needs today and a little of tomorrow.
        self.max_steps = scrape.get("max_steps", 16)

        self.page_timeout_ms = scrape.get("page_timeout_ms", 60000)

    # --------------------------------------------------

    LOW_MEMORY_BROWSER_ARGS = [
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-extensions",
        "--no-sandbox",
    ]

    # www.windy.com draws its map with WebGL; the embed does not. With
    # the args above, headless Chromium has no GL backend and the full
    # app renders a flat grey page reading "Failed to initialize WebGL
    # Overlay" - while still producing a valid screenshot and a
    # perfectly precise measurement of nothing. SwiftShader is
    # Chromium's software GL, so the map renders without a GPU, at some
    # CPU and memory cost.
    WEBGL_BROWSER_ARGS = [
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--no-sandbox",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
    ]

    # Where the plant's value appears, per page.
    READOUT_SELECTOR = {
        "embed": "big",                     # the embed's picker
        "www": ".picker-change-metric",     # the full app's pinned picker
    }

    @property
    def browser_args(self):

        return (
            self.WEBGL_BROWSER_ARGS if self.source == "www"
            else self.LOW_MEMORY_BROWSER_ARGS
        )

    @property
    def viewport(self):

        return (
            {"width": 1600, "height": 1000} if self.source == "www"
            else {"width": 1280, "height": 720}
        )

    def www_url(self, overlay):
        """
        Their URL template - the full application, centred on the plant.
        """

        return (
            f"https://www.windy.com/{self.lat}/{self.lon}"
            f"?{overlay},{self.lat},{self.lon},{self.zoom_www},p:cities"
        )

    def layer_url(self, overlay):

        return (
            self.www_url(overlay) if self.source == "www"
            else self.embed_url(overlay)
        )

    def embed_url(self, overlay):
        """
        The same embed windy_capture.py records, pinned to this
        plant. Only the overlay differs per call.
        """

        params = (
            f"lat={self.lat}&lon={self.lon}"
            f"&detailLat={self.lat}&detailLon={self.lon}"
            f"&width=1280&height=720&zoom={self.zoom}"
            f"&level=surface&overlay={overlay}"
            "&menu=&message=&marker=true&calendar=&pressure="
            "&type=map&location=coordinates&detail=&metricWind=default"
            "&metricTemp=default&radarRange=-1"
        )

        if self.api_key:
            params += f"&key={self.api_key}"

        return f"https://embed.windy.com/embed2.html?{params}"

    # --------------------------------------------------

    # Runs inside the page. Walks the CURRENT overlay's own calendar,
    # setting each timestamp and reading the picker back, and returns
    # the raw rows for Python to interpret.
    #
    # TIMING - this is the whole correctness problem, measured on
    # 2026-08-09 against values verified by hand:
    #
    #   * A plain sleep after set('timestamp') reads the PREVIOUS
    #     step's value whenever the new data file is still loading.
    #     A 900 ms sleep got 06Z wrong (reported 173, actual 765).
    #   * W.broadcast's `redrawFinished` fires when the MAP finishes
    #     redrawing, but the picker label updates on a later tick - so
    #     waiting only for that reads consistently one step behind
    #     (it put 765 against 09Z instead of 06Z).
    #
    # Both together are correct: wait for the redraw, THEN poll the
    # label until it reads the same twice running. That settles in
    # ~750 ms per step and reproduces the hand-verified series
    # (03Z=173, 06Z=765, 09Z=492, 12Z=209, 15Z=0 W/m2).
    #
    # `raw` (W.interpolator's underlying tile number) is recorded
    # alongside so a label that moves while the data behind it does
    # not is visible rather than silently trusted.
    _SWEEP_JS = """
    async ({lat, lon, waitMs, maxSteps, selector, useInterpolator}) => {
      const W = window.W;
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      const read = () => {
        const el = document.querySelector(selector);
        return el ? el.innerText.trim() : null;
      };

      const product = W.store.get('product');
      const calendar = W.products[product] && W.products[product].calendar;
      if (!calendar || !calendar.timestamps || !calendar.timestamps.length) {
        return {error: 'no calendar for product ' + product};
      }

      const waitRedraw = () => new Promise(res => {
        let done = false;
        const timer = setTimeout(() => { if (!done) { done = true; res('timeout'); } }, 8000);
        try {
          W.broadcast.once('redrawFinished', () => {
            if (!done) { done = true; clearTimeout(timer); res('ok'); }
          });
        } catch (e) { done = true; clearTimeout(timer); res('no-event'); }
      });

      // Poll until the label reads identically twice in a row, so a
      // half-loaded step can never be recorded as a measurement.
      //
      // A null reading is NOT a stable reading. The full app's picker
      // can be empty for a second or two after a step while it fetches
      // the point value, and treating "null twice" as settled recorded
      // gaps at 10:00Z and 11:00Z and lost every cloud layer entirely
      // on 2026-08-09. Null therefore resets the counter and keeps
      // waiting, up to the same overall limit.
      const settle = async () => {
        let previous = null, stable = 0;
        const limit = Math.max(12, Math.ceil(waitMs / 250) * 6);
        for (let i = 0; i < limit; i++) {
          await sleep(250);
          const now = read();
          if (now !== null && now !== '' && now === previous) {
            stable++;
            if (stable >= 2) return {value: now, settledMs: (i + 1) * 250};
          } else {
            stable = 0; previous = now;
          }
        }
        return {value: read(), settledMs: limit * 250, settled: false};
      };

      // W.interpolator is EMBED-ONLY. On www.windy.com it throws
      // "Cannot destructure property 'premiumOnly' of 'j[e]'" from deep
      // inside Windy's own bundle, which aborts the whole sweep and
      // loses every layer. The full app does not need it - the pinned
      // picker already puts the exact figure in the DOM - so it is
      // simply not called there.
      const interpolate = () => new Promise((res) => {
        if (!useInterpolator) { res(null); return; }
        let done = false;
        const timer = setTimeout(() => { if (!done) { done = true; res(null); } }, 6000);
        try {
          W.interpolator(fn => {
            if (done) return;
            done = true; clearTimeout(timer);
            try { res(fn({lat: lat, lon: lon})); } catch (e) { res(null); }
          });
        } catch (e) { done = true; clearTimeout(timer); res(null); }
      });

      // W.store.get('path') THROWS on www.windy.com - the same
      // "Cannot destructure property 'premiumOnly'" from inside Windy's
      // bundle. It is only a diagnostic (which data file served this
      // step), so it must never be allowed to abort a sweep and lose
      // every layer. An earlier survey had this read inside a
      // try/catch and reported path as null, which hid the throw and
      // sent the whole investigation after the timestamp setter and
      // then the interpolator, neither of which was at fault.
      const safePath = () => {
        try { return W.store.get('path'); } catch (e) { return null; }
      };

      const stamps = calendar.timestamps.slice(0, maxSteps);
      const rows = [];

      for (const ts of stamps) {
        W.store.set('timestamp', ts);
        await waitRedraw();
        const settled = await settle();
        rows.push({
          ts: ts,
          path: safePath(),
          big: settled.value,
          settledMs: settled.settledMs,
          settled: settled.settled !== false,
          raw: await interpolate()
        });
      }

      return {
        product: product,
        overlay: W.store.get('overlay'),
        refTime: calendar.refTimeTxt || null,
        updated: calendar.updateTxt || null,
        stepMinutes: calendar.timestamps.length > 1
          ? (calendar.timestamps[1] - calendar.timestamps[0]) / 60000 : null,
        totalSteps: calendar.timestamps.length,
        premium: (() => { try { return W.subscription.hasAny(); } catch (e) { return null; } })(),
        rows: rows
      };
    }
    """

    # --------------------------------------------------

    def pin_picker(self, page):
        """
        Pin the full app's weather picker on the plant, so its value
        appears in the DOM.

        Right-click alone is NOT enough: it opens Windy's context menu
        and leaves it sitting over the plant. The menu item has to be
        clicked, which both pins the picker and clears the menu. If the
        item cannot be found, Escape closes the menu so at least it is
        not covering the map.
        """

        width = self.viewport["width"]
        height = self.viewport["height"]

        try:
            page.mouse.click(width // 2, height // 2, button="right")
            page.wait_for_timeout(1500)
        except Exception as error:
            self.logger.warning(f"Picker right-click failed ({error})")
            return False

        for selector in ("text=Show weather picker", "text=Weather picker"):

            try:
                page.locator(selector).first.click(timeout=3000)
                page.wait_for_timeout(2000)
                return True
            except Exception:
                continue

        self.logger.warning(
            "Could not pin the weather picker - readings for this layer will "
            "be empty"
        )

        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

        return False

    # --------------------------------------------------

    def scrape_overlay(self, page, overlay):
        """
        Walks one overlay's forecast timeline and returns its
        readings, or None when the page never produced a usable
        calendar.

        The returned dict carries `stale`: True when every step
        came back with an identical reading. That is not a
        forecast - it means Windy served one data file for the
        whole sweep (the 5-minutely satellite layer behaves this
        way inside the embed), and the caller must discard it
        rather than emit a flat line as if it were measured.
        """

        page.goto(
            self.layer_url(overlay),
            wait_until="load",
            timeout=self.page_timeout_ms
        )

        # Windy loads its bundle, then the product's calendar. Reading
        # before both exist returns an empty sweep, so wait for the
        # object itself rather than a fixed sleep.
        page.wait_for_function(
            "() => window.W && window.W.store && window.W.products "
            "&& window.W.products[window.W.store.get('product')] "
            "&& window.W.products[window.W.store.get('product')].calendar",
            timeout=self.page_timeout_ms
        )
        page.wait_for_timeout(2000)   # let the first tiles paint

        if self.source == "www":
            self.pin_picker(page)

        result = page.evaluate(
            self._SWEEP_JS,
            {
                "lat": self.lat,
                "lon": self.lon,
                "waitMs": self.step_wait_ms,
                "maxSteps": self.max_steps,
                "selector": self.READOUT_SELECTOR[self.source],
                "useInterpolator": self.source == "embed",
            }
        )

        if not result or result.get("error"):
            self.logger.warning(
                f"Windy scrape for '{overlay}' returned nothing "
                f"({(result or {}).get('error')})"
            )
            return None

        readings = []

        unsettled = 0

        for row in result["rows"]:

            value, unit, direction = parse_readout(row.get("big"))

            if not row.get("settled", True):
                unsettled += 1

            readings.append({
                # Windy timestamps are ms UTC. Kept in UTC here, with an
                # explicit Z, so the caller converts once against the
                # plant timezone rather than guessing at a naive string.
                "timestamp": (
                    datetime.utcfromtimestamp(row["ts"] / 1000).isoformat() + "Z"
                ),
                "value": value,
                "unit": unit,
                "direction": direction,
                "raw": row.get("raw"),
                "path": row.get("path"),
                "settled": row.get("settled", True),
            })

        if unsettled:
            self.logger.warning(
                f"Windy '{overlay}': {unsettled} of {len(readings)} step(s) "
                "never settled to a stable reading - those values may be the "
                "previous step's and should be treated as suspect"
            )

        distinct = {
            (r["value"], json.dumps(r["raw"])) for r in readings
        }

        stale = len(readings) > 1 and len(distinct) == 1

        if stale:
            self.logger.warning(
                f"Windy '{overlay}': all {len(readings)} steps returned the "
                f"same reading ({readings[0]['value']} {readings[0]['unit']}) "
                "- Windy served one data file for the whole sweep. Marked "
                "stale; do NOT use this as a time series."
            )

        return {
            "overlay": result.get("overlay", overlay),
            "product": result.get("product"),
            "provider_ref_time": result.get("refTime"),
            "provider_updated": result.get("updated"),
            "step_minutes": result.get("stepMinutes"),
            "total_steps_available": result.get("totalSteps"),
            "premium": result.get("premium"),
            "stale": stale,
            "readings": readings,
        }

    # --------------------------------------------------

    def scrape(self, run_time=None):
        """
        Scrapes every configured overlay in ONE browser session and
        writes the result beside the captures as
        <PLANT>_<YYYY-MM-DD_HH-MM-SS>_values.json.

        One browser for all overlays is deliberate: launching Chromium
        is the expensive part on the ~900 MB EC2 box, and the embed
        only needs its overlay swapped between sweeps.
        """

        run_time = run_time or datetime.now()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "plant": self.plant_tag,
            "latitude": self.lat,
            "longitude": self.lon,
            "scraped_at": run_time.isoformat(),
            "layers": {},
        }

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=self.headless,
                args=self.browser_args
            )

            options = {"viewport": self.viewport}
            if self.session_file.exists():
                options["storage_state"] = str(self.session_file)

            self.logger.info(
                f"Windy scrape source: {self.source}"
                + (" (premium session present)" if self.session_file.exists()
                   else " (no session - free tier)")
            )

            context = browser.new_context(**options)
            page = context.new_page()

            for overlay in self.overlays:

                try:
                    layer = self.scrape_overlay(page, overlay)

                except Exception as error:
                    # One bad layer must not cost the whole sweep - the
                    # others are still a usable forecast input.
                    self.logger.warning(
                        f"Windy scrape failed for '{overlay}' ({error}) - skipping"
                    )
                    continue

                if layer is None:
                    continue

                payload["layers"][overlay] = layer

                usable = [
                    r for r in layer["readings"] if r["value"] is not None
                ]

                self.logger.info(
                    f"Windy '{overlay}': {len(usable)} value(s), "
                    f"{layer['step_minutes']} min steps, "
                    f"product {layer['product']}"
                    + (" [STALE - unusable as a series]" if layer["stale"] else "")
                )

            context.close()
            browser.close()

        stem = f"{self.plant_tag}_{run_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        output_path = self.output_dir / f"{stem}_values.json"

        output_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        self.logger.info(f"Windy values saved: {output_path}")

        return payload, output_path


# --------------------------------------------------

def main():
    """
    Runnable on its own so a scrape can be inspected without the
    scheduler: `python -m modules.capture.windy_scraper`.
    """

    scraper = WindyScraper()

    payload, path = scraper.scrape()

    print(f"saved: {path}\n")

    for overlay, layer in payload["layers"].items():

        flag = "  [STALE]" if layer["stale"] else ""

        print(f"{overlay:12s} product={layer['product']:10s} "
              f"step={layer['step_minutes']} min{flag}")

        for reading in layer["readings"][:8]:
            print(f"    {reading['timestamp']}  "
                  f"{reading['value']} {reading['unit']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
