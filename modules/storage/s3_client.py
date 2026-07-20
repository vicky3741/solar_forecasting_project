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

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config.config import settings
from utils.logger import get_logger


class S3Storage:

    def __init__(self):

        self.logger = get_logger()

        store = settings.get("storage", {})

        self.enabled = store.get("enabled", False)
        self.bucket = store.get("bucket")
        self.region = store.get("region")
        self.site_prefix = store.get("site_prefix", "").rstrip("/")
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

        keys = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return keys

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
