"""
=========================================================
Solar Forecasting Project
S3 Storage Client
=========================================================
Read/write access to the team-shared AWS S3 bucket that
Team 3 maintains (the live data lake for meter data,
Enercast, Windy videos and weather reports).

Credentials are read from the environment only
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, loaded from the
gitignored .env by config.py) - boto3 picks them up
automatically, so no secret ever appears in code or config.

Bucket data-lake layout (Team 3's convention):
    inputs/<state>/<site>/<YYYY-MM-DD>/<DataType>/<file>
    e.g. inputs/MadhyaPradesh/SIRMOUR/2026-07-15/Metered_Data/2026_07_15_SOLAR_INV.csv

This team writes its forecast outputs back under
    outputs/team2/SIRMOUR/...
=========================================================
"""

import re
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config.config import settings
from utils.logger import get_logger


# Windy video filenames come in two formats:
#   current  : sirmour_YYMMDD_HH_MM.<ext>       e.g. sirmour_260721_12_57.mp4
#   older     : ..._SIRMOUR_satellite_YYYY-MM-DD_HH-MM-SS_clean.mp4 (Jul 14-15)
_VIDEO_TIME_PATTERN = re.compile(r"sirmour_(\d{2})(\d{2})(\d{2})_(\d{2})_(\d{2})")
_VIDEO_TIME_PATTERN_OLD = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})"
)

_VIDEO_EXTENSIONS = (".mp4", ".webm")


class S3Storage:

    def __init__(self):

        self.logger = get_logger()

        store = settings.get("storage", {})

        self.enabled = store.get("enabled", False)
        self.bucket = store.get("bucket")
        self.region = store.get("region")
        self.site_prefix = store.get("site_prefix", "").rstrip("/")
        self.video_prefix = store.get("video_prefix", "").rstrip("/")
        self.video_cache = store.get("video_cache", "data/windy/s3_cache")
        self.output_prefix = store.get("output_prefix", "").rstrip("/")

        self._client = None

    # --------------------------------------------------

    @property
    def client(self):

        if self._client is None:
            self._client = boto3.client("s3", region_name=self.region)

        return self._client

    # --------------------------------------------------

    def is_available(self):
        """
        True only if storage is enabled and the credentials
        actually authenticate - so callers can fall back to
        local files when the cloud is unreachable instead of
        crashing.
        """

        if not self.enabled:
            return False

        try:
            boto3.client("sts", region_name=self.region).get_caller_identity()
            return True

        except (BotoCoreError, ClientError, NoCredentialsError) as error:
            self.logger.warning(f"S3 not available: {error}")
            return False

    # --------------------------------------------------

    def list_objects(self, prefix=""):
        """
        Returns the keys of every object under `prefix`.
        """

        return [obj["Key"] for obj in self.list_objects_meta(prefix)]

    # --------------------------------------------------

    def list_objects_meta(self, prefix=""):
        """
        Like list_objects but returns each object's key, size
        and last-modified time - needed to dedupe and skip
        redundant downloads during a sync.
        """

        objects = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append({
                    "Key": obj["Key"],
                    "Size": obj["Size"],
                    "LastModified": obj["LastModified"]
                })

        return objects

    # --------------------------------------------------

    def download(self, key, local_path):
        """
        Downloads a single object to `local_path`.
        """

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        self.client.download_file(self.bucket, key, str(local_path))

        self.logger.info(f"S3 downloaded: {key} -> {local_path}")

        return local_path

    # --------------------------------------------------

    def upload(self, local_path, key):
        """
        Uploads a single local file to `key` in the bucket.
        """

        self.client.upload_file(str(local_path), self.bucket, key)

        self.logger.info(f"S3 uploaded: {local_path} -> {key}")

        return key

    # --------------------------------------------------

    def site_key(self, date, data_type, filename):
        """
        Builds a bucket key for a site input file, e.g.
        (date="2026-07-15", data_type="Metered_Data",
         filename="2026_07_15_SOLAR_INV.csv").
        """

        return f"{self.site_prefix}/{date}/{data_type}/{filename}"

    # --------------------------------------------------

    def output_key(self, *parts):
        """
        Builds a bucket key under this team's output area.
        """

        return "/".join([self.output_prefix, *parts])

    # --------------------------------------------------

    def download_site_data(self, data_type, local_folder, extension=".csv"):
        """
        Pulls every file of a given data type (e.g.
        "Metered_Data") across all dates in the bucket into a
        local folder. Returns the list of downloaded paths.
        """

        local_folder = Path(local_folder)
        downloaded = []

        for key in self.list_objects(self.site_prefix):

            if f"/{data_type}/" not in key or not key.endswith(extension):
                continue

            filename = key.split("/")[-1]
            path = self.download(key, local_folder / filename)
            downloaded.append(path)

        return downloaded

    # --------------------------------------------------

    def sync_meter_data(self, local_folder, min_size_bytes=100):
        """
        Merges the bucket's meter CSVs into `local_folder`.

        The same meter file can appear under several date
        folders in the bucket (Team 3 re-dumps batches), so
        files are keyed by their real filename (which encodes
        the data date) and the most recently modified copy
        wins. Placeholder/empty files below min_size_bytes are
        skipped, and a file whose local copy already matches
        the bucket size is not re-downloaded. Existing local
        files for dates the bucket does not have are left
        untouched (merge, never delete).
        """

        local_folder = Path(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)

        # filename -> (last_modified, key, size), newest kept
        best = {}

        for obj in self.list_objects_meta(self.site_prefix):

            key = obj["Key"]

            if "/Metered_Data/" not in key or not key.endswith(".csv"):
                continue

            if obj["Size"] < min_size_bytes:
                continue

            filename = key.split("/")[-1]

            if filename not in best or obj["LastModified"] > best[filename][0]:
                best[filename] = (obj["LastModified"], key, obj["Size"])

        synced = []

        for filename, (_, key, size) in best.items():

            local_path = local_folder / filename

            if local_path.exists() and local_path.stat().st_size == size:
                continue  # already up to date

            self.download(key, local_path)
            synced.append(local_path)

        return synced

    # --------------------------------------------------

    def push_output(self, local_path, *key_parts):
        """
        Uploads a local file to this team's output area in the
        bucket, e.g. push_output(path, "forecasts", "run.csv")
        -> outputs/team2/SIRMOUR/forecasts/run.csv
        """

        return self.upload(local_path, self.output_key(*key_parts))

    # --------------------------------------------------

    @staticmethod
    def parse_video_time(key):
        """
        Recording datetime encoded in a Windy video filename,
        or None if it matches neither known pattern.
        """

        match = _VIDEO_TIME_PATTERN.search(key)
        if match:
            yy, mm, dd, hh, minute = map(int, match.groups())
            try:
                return datetime(2000 + yy, mm, dd, hh, minute)
            except ValueError:
                return None

        match = _VIDEO_TIME_PATTERN_OLD.search(key)
        if match:
            yyyy, mm, dd, hh, minute, ss = map(int, match.groups())
            try:
                return datetime(yyyy, mm, dd, hh, minute, ss)
            except ValueError:
                return None

        return None

    # --------------------------------------------------

    def latest_video_before(self, run_time):
        """
        Key of the most recent Windy video recorded on the
        same calendar day at or before run_time - the single
        freshest same-day video, matching the forecast's
        "one latest same-day video" rule. Returns None when no
        qualifying video exists.
        """

        run_time = run_time.to_pydatetime() if hasattr(run_time, "to_pydatetime") else run_time

        date_prefix = f"{self.video_prefix}/{run_time.date()}"

        candidates = []

        for obj in self.list_objects_meta(date_prefix):

            key = obj["Key"]

            if not key.lower().endswith(_VIDEO_EXTENSIONS):
                continue

            video_time = self.parse_video_time(key)

            if video_time is None or video_time > run_time:
                continue

            candidates.append((video_time, key))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])

        return candidates[-1][1]

    # --------------------------------------------------

    def fetch_latest_video(self, run_time):
        """
        Downloads the latest same-day Windy video at/before
        run_time into the local video cache and returns its
        path, or None if there is no such video. Skips the
        download when the cached file already matches.
        """

        key = self.latest_video_before(run_time)

        if key is None:
            return None

        filename = key.split("/")[-1]
        local_path = Path(self.video_cache) / filename

        if local_path.exists():
            return local_path

        return self.download(key, local_path)
