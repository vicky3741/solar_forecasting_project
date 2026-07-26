"""
=========================================================
Solar Forecasting Project
Windy Video Capture
=========================================================
Records the Windy satellite-view animation for the plant
location using a headless browser (Playwright), saves it
locally under the SIRMOUR_YYYY-MM-DD_HH-MM-SS convention
(see name_stem) and uploads it to the team S3 bucket so the
forecast can pull it back down.

RELIABILITY NOTES - this module previously failed when run
by Windows Task Scheduler ("BrowserType.launch: Executable
doesn't exist ...") while working fine from an interactive
shell. Two defences are in place:

  1. On Windows only, PLAYWRIGHT_BROWSERS_PATH is resolved
     and set explicitly BEFORE playwright is imported. Task
     Scheduler hands the process a trimmed environment, so
     leaving Playwright to infer the browser location from
     LOCALAPPDATA made it look in the wrong place. This must
     stay Windows-only: on 2026-07-25, deploying to an Ubuntu
     EC2 server, an earlier unconditional version of this fix
     built a literal "AppData/Local" path on Linux (there is
     no LOCALAPPDATA there, so it fell back to a Windows-
     shaped default), which broke a previously-working
     Linux install. Off-Windows, Playwright's own default
     resolution (~/.cache/ms-playwright and equivalents) is
     correct and must be left alone.
  2. This file is runnable on its own
     (`python -m modules.capture.windy_capture`) so the
     scheduler can invoke each capture as a FRESH short-lived
     subprocess rather than inside its own long-running
     process - isolating browser state and letting a hung
     capture be killed by timeout without taking the
     scheduler down with it.
=========================================================
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# --- must happen before `playwright` is imported (see note above) ---
if os.name == "nt":
    _DEFAULT_BROWSERS_PATH = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "ms-playwright"
    )
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_DEFAULT_BROWSERS_PATH))

from playwright.sync_api import sync_playwright  # noqa: E402

from config.config import settings  # noqa: E402
from modules.storage.s3_client import S3Storage  # noqa: E402
from modules.vision.optical_flow import OpticalFlowAnalyzer  # noqa: E402
from utils.logger import get_logger  # noqa: E402


class WindyCapture:

    def __init__(self):

        self.logger = get_logger()

        plant = settings["plant"]
        capture = settings["windy_capture"]

        self.lat = plant["latitude"]
        self.lon = plant["longitude"]

        self.api_key = capture.get("api_key")
        self.zoom = capture.get("zoom", 8)
        self.overlay = capture.get("overlay", "satellite")
        self.capture_seconds = capture.get("capture_seconds", 20)
        self.headless = capture.get("headless", True)
        self.video_dir = Path(capture.get("video_dir", "data/windy/videos"))
        self.upload_to_s3 = capture.get("upload_to_s3", True)

        # Premium session saved by modules/capture/windy_login.py.
        # Absent = we fall back to the free public embed, which
        # still works; premium simply unlocks extra overlays.
        self.session_file = Path(capture.get("session_file", "windy_session.json"))

        # Extra still layers grabbed alongside the satellite clip.
        # solarpower is the valuable one - it is Windy's modelled
        # irradiance, far closer to what we actually forecast than
        # a picture of cloud is.
        self.layers = capture.get("layers", [])
        self.screenshot_dir = Path(
            capture.get("screenshot_dir", "data/windy/screenshots")
        )

        self.vis_button_position = tuple(
            capture.get("vis_button_position", [117, 688])
        )
        self.play_button_position = tuple(
            capture.get("play_button_position", [31, 665])
        )

        # Playwright's raw WebM is ~56 MB for a 20s 720p clip. Seven of
        # those a day fills the EC2 box's 6.7 GB disk in under two days
        # and takes the whole automation down with it, so each clip is
        # re-encoded to H.264 (same resolution, ~20x smaller) and old
        # local files are pruned once they are safely in S3.
        self.compress = capture.get("compress", True)
        self.compress_crf = capture.get("compress_crf", 30)
        self.local_retention_days = capture.get("local_retention_days", 3)

        self.storage = S3Storage()

    # --------------------------------------------------

    def embed_url(self, overlay=None):
        """
        Windy's embed widget, centered on the plant and set to
        the given overlay layer (defaults to the configured
        one). `key` is only appended when a premium API key is
        configured.
        """

        overlay = overlay or self.overlay

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

    def context_options(self, **extra):
        """
        Browser-context options, carrying the saved premium
        session when one exists so premium-only overlays
        render instead of falling back to the free set.
        """

        options = {"viewport": {"width": 1280, "height": 720}}

        if self.session_file.exists():
            options["storage_state"] = str(self.session_file)

        options.update(extra)

        return options

    # --------------------------------------------------

    # Chromium defaults assume a desktop's worth of memory. The EC2
    # box has ~900 MB of RAM, and on 2026-07-26 a capture there died
    # with "Mouse.click: Target crashed" - the renderer being OOM-killed
    # mid-recording. That left a 14-minute, 55 MB clip behind (Playwright
    # keeps recording while the retry runs), which is what nearly filled
    # the disk. These flags cut the renderer's footprint; the big one is
    # --disable-dev-shm-usage, since /dev/shm here is RAM-backed and
    # small, so Chromium exhausting it takes the tab down.
    LOW_MEMORY_BROWSER_ARGS = [
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-extensions",
        "--no-sandbox",
    ]

    # Plant tag matching the S3 site convention (inputs/.../SIRMOUR).
    PLANT_TAG = "SIRMOUR"

    def name_stem(self, run_time):
        """
        Shared, human-readable basename for every artifact of one
        capture: PLANT_YYYY-MM-DD_HH-MM-SS. Unambiguous about the
        exact date and time, and still parseable by S3Storage's
        date-time regex so the forecast can locate the clip.
        """

        return f"{self.PLANT_TAG}_{run_time.strftime('%Y-%m-%d_%H-%M-%S')}"

    def output_filename(self, run_time):

        return f"{self.name_stem(run_time)}.webm"

    # --------------------------------------------------

    def compress_video(self, video_path):
        """
        Re-encodes a recorded clip to H.264 MP4 at the same
        resolution, which keeps the cloud detail the vision models
        read while cutting the file from ~56 MB to a few MB.

        Returns the new path, or the original untouched path if
        ffmpeg is missing or fails - a bulky clip is always better
        than no clip.
        """

        video_path = Path(video_path)

        if not self.compress:
            return video_path

        if shutil.which("ffmpeg") is None:
            self.logger.warning(
                "ffmpeg not found - keeping the raw clip uncompressed"
            )
            return video_path

        compressed = video_path.with_suffix(".mp4")

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(video_path),
                    "-c:v", "libx264",
                    "-crf", str(self.compress_crf),
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    str(compressed),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode != 0 or not compressed.exists():
                self.logger.warning(
                    f"ffmpeg compression failed ({result.stderr.strip()[-200:]}) "
                    "- keeping the raw clip"
                )
                compressed.unlink(missing_ok=True)
                return video_path

            before_mb = video_path.stat().st_size / 1e6
            after_mb = compressed.stat().st_size / 1e6

            video_path.unlink(missing_ok=True)

            self.logger.info(
                f"Compressed {before_mb:.0f} MB -> {after_mb:.1f} MB: "
                f"{compressed.name}"
            )

            return compressed

        except Exception as error:
            self.logger.warning(
                f"ffmpeg compression skipped ({error}) - keeping the raw clip"
            )
            compressed.unlink(missing_ok=True)
            return video_path

    # --------------------------------------------------

    def prune_local_files(self):
        """
        Deletes local clips, screenshots and cached S3 videos older
        than `local_retention_days`. Everything is already in S3, so
        the local copies exist only as a same-day fallback for when
        the bucket is unreachable - keeping them forever is what
        would eventually fill the disk and stop the automation.
        """

        if not self.local_retention_days:
            return 0

        cutoff = datetime.now() - timedelta(days=self.local_retention_days)

        folders = [self.video_dir, self.screenshot_dir]

        cache_dir = Path(self.storage.video_cache)
        if cache_dir.exists():
            folders.append(cache_dir)

        removed = 0
        freed = 0

        for folder in folders:

            if not folder.exists():
                continue

            for path in folder.iterdir():

                if not path.is_file():
                    continue

                try:
                    if datetime.fromtimestamp(path.stat().st_mtime) >= cutoff:
                        continue

                    freed += path.stat().st_size
                    path.unlink()
                    removed += 1

                except OSError as error:
                    self.logger.warning(f"Could not remove {path.name}: {error}")

        if removed:
            self.logger.info(
                f"Pruned {removed} local file(s) older than "
                f"{self.local_retention_days} days, freeing {freed/1e6:.0f} MB"
            )

        return removed

    # --------------------------------------------------

    def capture(self, run_time=None):
        """
        Records one clip, retrying once on failure - a
        scheduled run only gets one shot at a given time, so a
        single retry meaningfully cuts the chance of losing a
        run's video to a transient glitch.
        """

        try:
            return self._capture_once(run_time)

        except Exception as error:
            self.logger.warning(f"Windy capture failed, retrying once: {error}")
            time.sleep(5)
            return self._capture_once(run_time)

    # --------------------------------------------------

    def _capture_once(self, run_time=None):
        """
        Records one Windy animation clip and returns the local
        file path, renamed via name_stem (SIRMOUR_YYYY-MM-DD_HH-MM-SS).
        `run_time` defaults to now and only affects the output
        filename.
        """

        run_time = run_time or datetime.now()

        self.video_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=self.headless,
                args=self.LOW_MEMORY_BROWSER_ARGS
            )

            context = browser.new_context(
                **self.context_options(
                    record_video_dir=str(self.video_dir),
                    record_video_size={"width": 1280, "height": 720}
                )
            )

            page = context.new_page()
            page.goto(self.embed_url(), wait_until="load", timeout=60000)
            page.wait_for_timeout(5000)  # let map tiles finish loading

            # The embed opens on the infrared sub-layer; switch to
            # visible-light and start the animation so the clip shows
            # actual cloud movement rather than one static frame.
            # These are fixed pixel positions in the embed's control
            # bar - stable only because the viewport is pinned to
            # 1280x720 via the URL params above.
            page.mouse.click(*self.vis_button_position)
            page.wait_for_timeout(1500)
            page.mouse.click(*self.play_button_position)

            page.wait_for_timeout(self.capture_seconds * 1000)

            video_path = page.video.path() if page.video else None

            context.close()
            browser.close()

        if video_path is None:
            raise RuntimeError("Playwright did not produce a video file")

        final_path = self.video_dir / self.output_filename(run_time)

        Path(video_path).replace(final_path)

        # A crashed first attempt leaves its own part-recorded
        # "page@<hash>.webm" behind. Those are never named by
        # output_filename, so nothing else ever cleans them up.
        for orphan in self.video_dir.glob("page@*.webm"):
            orphan.unlink(missing_ok=True)
            self.logger.info(f"Removed orphaned recording: {orphan.name}")

        # A healthy 20-30s clip is a few MB. When the disk filled on
        # 2026-07-26, Playwright "recorded" 0-byte files and they were
        # uploaded to S3 as if they were real - five scheduling slots
        # of junk. Fail loudly instead, so the retry (or the log) makes
        # the problem visible and nothing empty ever reaches the bucket.
        size = final_path.stat().st_size
        if size < 100_000:
            final_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Recording came out at {size} bytes - disk full or "
                "recording failure; refusing to keep/upload it"
            )

        final_path = self.compress_video(final_path)

        self.logger.info(f"Windy capture saved: {final_path}")

        return final_path

    # --------------------------------------------------

    def analyze_motion(self, video_path):
        """
        Runs the deterministic optical-flow measurement over a
        captured clip and writes the numbers beside it as a
        .json sidecar.

        This is recorded ALONGSIDE the existing signals, not
        substituted for any of them - the forecast still runs
        on the physics/kt core. Logging the numbers now means
        that when there is enough history we can backtest
        whether they add anything before letting them touch a
        forecast.
        """

        try:
            analyzer = OpticalFlowAnalyzer(
                latitude=self.lat,
                zoom=self.zoom
            )

            features = analyzer.analyze(video_path)

            sidecar = Path(video_path).with_suffix(".json")
            sidecar.write_text(json.dumps(features, indent=2), encoding="utf-8")

            self.logger.info(
                "Cloud motion: cover "
                f"{features['cloud_cover_fraction']:.0%}, "
                f"trend {features['cloud_cover_trend']:+.3f}, "
                f"direction {features['motion_direction']}"
            )

            return features, sidecar

        except Exception as error:
            self.logger.warning(
                f"Optical-flow analysis failed ({error}) - clip still saved"
            )
            return None, None

    # --------------------------------------------------

    def capture_layers(self, run_time=None):
        """
        Grabs one still screenshot per configured extra layer
        (e.g. solarpower, clouds, rain) at the same moment as
        the satellite clip.

        These are stills, not animations - what matters for a
        layer like solarpower is its value over the plant now,
        not how it moves. Returns {layer: path}; layers that
        fail are skipped rather than aborting the run.
        """

        run_time = run_time or datetime.now()

        if not self.layers:
            return {}

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        stem = self.name_stem(run_time)
        captured = {}

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=self.headless,
                args=self.LOW_MEMORY_BROWSER_ARGS
            )
            context = browser.new_context(**self.context_options())
            page = context.new_page()

            for layer in self.layers:

                try:
                    page.goto(
                        self.embed_url(overlay=layer),
                        wait_until="load",
                        timeout=60000
                    )
                    page.wait_for_timeout(5000)  # let tiles render

                    path = self.screenshot_dir / f"{stem}_{layer}.png"
                    page.screenshot(path=str(path))

                    captured[layer] = path
                    self.logger.info(f"Windy layer captured: {path}")

                except Exception as error:
                    self.logger.warning(
                        f"Layer '{layer}' capture failed ({error}) - skipping"
                    )

            context.close()
            browser.close()

        return captured

    # --------------------------------------------------

    def capture_and_upload(self, run_time=None):
        """
        Captures one clip and pushes it to the same S3 prefix
        the forecast's auto-pull already reads from
        (videos/<state>/<site>/<date>/<file>).

        The clip is uploaded FIRST, before motion analysis and the
        extra layer screenshots. Those steps are the memory-hungry
        ones, and on 2026-07-26 optical flow was OOM-killed on the
        server after a clip had been recorded but before it reached
        the bucket - losing that scheduling slot's video entirely.
        A process killed outright cannot run an exception handler,
        so the only real protection is to bank the clip first.

        Returns (local_path, s3_key); s3_key is None when the
        upload was skipped or failed, so a capture is never
        lost just because the network was down.
        """

        run_time = run_time or datetime.now()

        local_path = self.capture(run_time)

        date_str = run_time.strftime("%Y-%m-%d")
        key = None

        if self.upload_to_s3:
            try:
                if self.storage.is_available():
                    key = f"{self.storage.video_prefix}/{date_str}/{local_path.name}"
                    self.storage.upload(local_path, key)
                else:
                    self.logger.warning(
                        "S3 not available - clip kept local only"
                    )
            except Exception as error:
                key = None
                self.logger.error(
                    f"S3 upload failed ({error}) - clip kept locally at {local_path}"
                )

        # --- everything below is a bonus signal; the clip is safe ---

        _, motion_sidecar = self.analyze_motion(local_path)

        try:
            layer_paths = self.capture_layers(run_time)
        except Exception as error:
            self.logger.warning(f"Layer capture failed ({error}) - continuing")
            layer_paths = {}

        if motion_sidecar is not None:
            layer_paths["_motion"] = motion_sidecar

        if self.upload_to_s3 and key is not None:
            for layer_path in layer_paths.values():
                try:
                    self.storage.upload(
                        layer_path,
                        f"{self.storage.video_prefix}/{date_str}/{layer_path.name}"
                    )
                except Exception as error:
                    self.logger.warning(
                        f"Layer upload failed for {layer_path.name}: {error}"
                    )

        # Prune only what is already safely in the bucket.
        if key is not None or not self.upload_to_s3:
            self.prune_local_files()

        return local_path, key


# --------------------------------------------------

def main():
    """
    Entry point so a capture can run as its own short-lived
    process (see the reliability notes at the top).
    Exit code 0 = video recorded, 1 = capture failed.
    """

    capture = WindyCapture()

    try:
        local_path, key = capture.capture_and_upload()

    except Exception as error:
        capture.logger.error(f"Windy capture failed: {error}")
        return 1

    print(f"saved: {local_path}")
    print(f"s3   : {key or '(not uploaded)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
