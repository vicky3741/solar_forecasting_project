"""
=========================================================
Solar Forecasting Project
S3 Storage Smoke Test
=========================================================
Verifies read + write against the team-shared S3 bucket:
  1. Confirms credentials authenticate.
  2. Lists what is in the site input area.
  3. Downloads one real meter CSV to a temp path.
  4. Uploads a tiny test file to this team's output area
     and confirms it lands, then cleans it up.
=========================================================
"""

import tempfile
from pathlib import Path

from modules.storage.s3_client import S3Storage


def main():

    s3 = S3Storage()

    print("=" * 60)
    print("S3 Storage Test")
    print("=" * 60)
    print(f"Bucket : {s3.bucket}")
    print(f"Region : {s3.region}")

    if not s3.is_available():
        print("\nS3 is not available (check .env credentials). Stopping.")
        return

    print("Credentials OK.\n")

    # 1. List site inputs
    keys = s3.list_objects(s3.site_prefix)
    print(f"Objects under {s3.site_prefix}/ : {len(keys)}")

    meter_keys = [k for k in keys if "/Metered_Data/" in k and k.endswith(".csv")]
    print(f"Meter CSVs found : {len(meter_keys)}")
    for k in meter_keys[:5]:
        print("   ", k)

    # 2. Download one meter file
    if meter_keys:
        target = Path(tempfile.gettempdir()) / "s3_meter_sample.csv"
        s3.download(meter_keys[0], target)
        size = target.stat().st_size
        print(f"\nDownloaded sample: {target.name} ({size} bytes) - read OK")

    # 3. Upload a small test file to our own output area
    test_local = Path(tempfile.gettempdir()) / "team2_s3_write_test.txt"
    test_local.write_text("team2 s3 write test\n", encoding="utf-8")

    test_key = s3.output_key("_connectivity_test.txt")
    s3.upload(test_local, test_key)

    # Confirm it is there
    written = s3.list_objects(s3.output_prefix)
    ok = test_key in written
    print(f"\nUploaded test file to: {test_key}")
    print(f"Write confirmed      : {ok}")

    # Clean up the test object
    if ok:
        s3.client.delete_object(Bucket=s3.bucket, Key=test_key)
        print("Test file cleaned up.")

    print("\nS3 read + write both working.")


if __name__ == "__main__":
    main()
