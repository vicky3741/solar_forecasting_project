"""
=========================================================
Solar Forecasting Project
Plant Meter-History Ingest
=========================================================
Loads a plant's daily meter CSVs from wherever the site sent
them into that plant's own historical folder, and (optionally)
into the team S3 bucket under the same layout the live
auto-pull already reads:

    inputs/<state>/<SITE>/<YYYY-MM-DD>/Metered_Data/<file>

Written for the 2026-08-08 hand-off of Kasipet and
Bhupalpally, whose 37 days of history (2026-07-01 .. 08-06)
arrived as one folder per day inside a zip. It is not
one-off: the same command re-runs safely whenever a new batch
of days is dropped in, and it is how a fourth plant would be
onboarded.

WHAT IT CHECKS
--------------
The files are validated through the plant's OWN configured
schema before being accepted, so a vendor changing its export
(a renamed column, a different date order) is caught here -
at ingest, against a file you can go and look at - rather
than surfacing later as a quietly wrong forecast. Anything
that fails is reported and skipped; the good days still land.

The date in the filename must also agree with the timestamps
inside it. Telangana's export is dd-mm-yyyy, which pandas
reads as mm-dd-yyyy unless told otherwise, so this mismatch
is exactly the kind that would otherwise pass silently.

Run:
    SOLAR_PLANT=kasipet python -m tests.ingest_plant_history \
        "C:/.../New Plants/KASIPET" --upload

    (--upload also pushes each day to S3; omit it to stay local)
=========================================================
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.preprocessing.preprocess import DataPreprocessor
from modules.storage.s3_client import S3Storage
from utils.file_manager import date_in_name


def parse_args():

    parser = argparse.ArgumentParser(
        description="Ingest a plant's daily meter CSVs into its historical folder."
    )
    parser.add_argument(
        "source",
        help="folder to search recursively for the plant's daily CSVs",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="filename glob to pick up (default: *.csv)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="also upload each accepted day to the team S3 bucket",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-copy days that are already present locally",
    )
    return parser.parse_args()


def check_file(preprocessor, path, expected_date):
    """
    Runs the plant's configured schema over one file. Returns
    (ok, message) - the message describes the day on success and
    the reason on failure.
    """

    try:
        frame = preprocessor.preprocess(file_path=path)

    except Exception as error:
        return False, f"unreadable under this plant's schema: {error}"

    if "active_power_kw" not in frame.columns:
        return False, (
            "no active_power_kw after the schema was applied - check "
            "data_schema.column_map for this plant"
        )

    dates = sorted(frame["timestamp"].dt.date.unique())

    if expected_date is not None and expected_date not in dates:
        return False, (
            f"filename says {expected_date} but the timestamps inside cover "
            f"{dates[0]}..{dates[-1]} - check data_schema.dayfirst"
        )

    real = int(frame.get("is_real_measurement", pd.Series(dtype=bool)).sum())
    peak = float(frame["active_power_kw"].max()) / 1000

    return True, f"{len(frame):3d} blocks, {real:3d} measured, peak {peak:6.3f} MW"


def main():

    args = parse_args()

    plant = settings["plant"]
    code = plant["code"]
    state = plant["state"]

    source = Path(args.source)

    if not source.exists():
        print(f"Source folder not found: {source}")
        return 1

    destination = Path(settings["paths"]["historical_data"])
    destination.mkdir(parents=True, exist_ok=True)

    files = sorted(set(source.rglob(args.pattern)))

    if not files:
        print(f"No files matching {args.pattern} under {source}")
        return 1

    print("=" * 78)
    print(f"INGEST - {plant['name']}  ({code}, {plant['capacity_mw']} MW)")
    print("=" * 78)
    print(f"source      : {source}")
    print(f"destination : {destination}")
    print(f"files found : {len(files)}")
    print()

    preprocessor = DataPreprocessor()
    storage = S3Storage() if args.upload else None
    can_upload = storage.is_available() if storage else False

    if args.upload and not can_upload:
        print("S3 is not reachable - files will be copied locally only.")
        print()

    accepted, skipped, uploaded = [], [], 0

    for path in files:

        day = date_in_name(path.name)

        target = destination / path.name

        already_local = target.exists() and not args.force

        if already_local:
            print(f"  = {path.name:32s} already present")

        else:
            ok, message = check_file(preprocessor, path, day)

            if not ok:
                print(f"  ! {path.name:32s} SKIPPED - {message}")
                skipped.append((path.name, message))
                continue

            shutil.copy2(path, target)
            print(f"  + {path.name:32s} {message}")

        accepted.append(day)

        # Uploading is decided independently of the local copy: a day
        # can be in place locally and still missing from the bucket
        # (that is exactly the state after a first, local-only run),
        # and the live auto-pull reads the bucket, not this folder.
        if can_upload and day is not None:
            key = f"inputs/{state}/{code}/{day}/Metered_Data/{path.name}"
            try:
                storage.upload(target, key)
                uploaded += 1
            except Exception as error:
                print(f"      (upload failed: {error})")

    print()
    print("-" * 78)
    known = sorted(d for d in accepted if d is not None)
    if known:
        print(f"days in place : {len(known)}  ({known[0]} .. {known[-1]})")
    print(f"skipped       : {len(skipped)}")
    print(f"uploaded to S3: {uploaded}")

    for name, message in skipped:
        print(f"   ! {name}: {message}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
