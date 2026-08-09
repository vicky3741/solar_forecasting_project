"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Meter data inbox watcher
=========================================================
Copies new meter CSVs from wherever the team drops them into
data/historical, so the feedback loop always has yesterday's
generation without anyone remembering to move a file.

Confirmed with the user on 2026-08-09: fresh meter exports
arrive by hand in

    Desktop/Vedanjay/SIRMOUR_DATA/SIRMOUR_DATA/meter_data/

as YYYY_MM_DD_SOLAR_INV.csv, and are then hand-copied into
data/historical. That hand-copy is the step this replaces.

WHY THIS IS COPY-ONLY, AND NEVER DELETES
----------------------------------------
The inbox is a shared folder that other people put things in
and take things out of. Deleting from it - even "after
successful import" - means a teammate's file can vanish
between them saving it and them looking for it again. So the
inbox is treated as strictly read-only: files are copied,
never moved, never removed.

An existing file in data/historical is only overwritten when
the inbox copy is NEWER and DIFFERENT in size, and the
overwrite is logged by name. Vendors do reissue a day's
export after correcting it - "(1)" suffixed duplicates are
already visible in the inbox - so silently ignoring newer
copies would pin the model to a superseded file.
=========================================================
"""

import argparse
import re
import shutil
from pathlib import Path

from config.config import settings
from utils.logger import get_logger


# The team's drop folder. Configurable, because the Telangana plants'
# exports arrive somewhere else entirely.
DEFAULT_INBOX = (
    Path.home() / "OneDrive" / "Desktop" / "Vedanjay"
    / "SIRMOUR_DATA" / "SIRMOUR_DATA" / "meter_data"
)

# "2026_08_07_SOLAR_INV.csv", and the reissued "..._SOLAR_INV (1).csv".
_METER_FILENAME = re.compile(
    r"^(\d{4})_(\d{2})_(\d{2})_.*\.csv$", re.IGNORECASE
)


class MeterInbox:

    def __init__(self, inbox=None, destination=None):

        self.logger = get_logger()

        watch = settings.get("meter_inbox", {})

        self.inbox = Path(
            inbox or watch.get("path") or DEFAULT_INBOX
        ).expanduser()

        self.destination = Path(
            destination or settings["paths"]["historical_data"]
        )

    # --------------------------------------------------

    def candidates(self):
        """
        Inbox files that look like a dated meter export, newest first.
        """

        if not self.inbox.exists():
            return []

        found = []

        for path in self.inbox.glob("*.csv"):

            if _METER_FILENAME.match(path.name):
                found.append(path)

        return sorted(found, key=lambda p: p.name)

    # --------------------------------------------------

    def canonical_name(self, path):
        """
        The name a file should have in data/historical.

        Strips a vendor's "(1)" reissue suffix so a corrected export
        REPLACES the original rather than sitting beside it - two files
        for one day would be concatenated by the preprocessor and that
        day would be counted twice.
        """

        return re.sub(r"\s*\(\d+\)(?=\.csv$)", "", path.name)

    # --------------------------------------------------

    def sync(self, dry_run=False):

        self.destination.mkdir(parents=True, exist_ok=True)

        copied = []
        replaced = []
        skipped = 0

        for source in self.candidates():

            target = self.destination / self.canonical_name(source)

            if target.exists():

                same_size = target.stat().st_size == source.stat().st_size
                newer = source.stat().st_mtime > target.stat().st_mtime

                if same_size or not newer:
                    skipped += 1
                    continue

                if not dry_run:
                    shutil.copy2(source, target)

                replaced.append(target.name)
                continue

            if not dry_run:
                shutil.copy2(source, target)

            copied.append(target.name)

        for name in copied:
            self.logger.info(f"Meter inbox: imported {name}")

        for name in replaced:
            self.logger.warning(
                f"Meter inbox: REPLACED {name} with a newer export from the "
                "inbox - that day's numbers have changed"
            )

        return {
            "inbox": str(self.inbox),
            "exists": self.inbox.exists(),
            "copied": copied,
            "replaced": replaced,
            "already_present": skipped,
        }


# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Copy new meter exports from the team inbox into "
                    "data/historical"
    )
    parser.add_argument("--inbox", default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    inbox = MeterInbox(args.inbox)

    result = inbox.sync(dry_run=args.dry_run)

    print("=" * 68)
    print("METER INBOX" + ("  (dry run - nothing copied)" if args.dry_run else ""))
    print("=" * 68)
    print(f"inbox           : {result['inbox']}")

    if not result["exists"]:
        print("\nThat folder does not exist. Pass --inbox <path>, or set "
              "meter_inbox.path in config/settings.yaml.")
        return 1

    print(f"already present : {result['already_present']}")
    print(f"imported        : {len(result['copied'])}")

    for name in result["copied"]:
        print(f"    + {name}")

    if result["replaced"]:
        print(f"replaced        : {len(result['replaced'])}")
        for name in result["replaced"]:
            print(f"    ~ {name}  (newer export - that day's numbers changed)")

    existing = sorted(Path(settings["paths"]["historical_data"]).glob("*.csv"))

    if existing:
        print(f"\ndata/historical : {len(existing)} day(s), "
              f"{existing[0].name} to {existing[-1].name}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
