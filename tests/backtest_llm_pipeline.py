"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Backtest the whole LLM pipeline against real generation
=========================================================
Runs the new pipeline as it would have run on past days -
features, precedent, LLM, validator, freeze - and saves each
schedule so tests/test_llm_approach_score.py can grade it
against actual meter data.

WHAT THIS TEST CAN AND CANNOT INCLUDE
-------------------------------------
Included, exactly as a live run would have them:
  * pvlib clear-sky physics
  * meter history up to the run time (never past it)
  * satellite clip features from that day's clips, pulled
    back out of S3
  * retrieved precedent from days strictly BEFORE the run day
  * the validator and the freeze horizon

NOT included, and this is the honest limit of the test:
  * the scraped Windy numbers. Windy serves only its CURRENT
    forecast - there is no archive to ask "what did you say
    on July 30?" - so those columns are empty for every
    historical run. They can only ever be validated forward,
    one day at a time, from the day the scraper started.

So this measures the pipeline WITHOUT its Windy numbers. If
it scores well here, the Windy numbers can only help; if it
scores badly, they are the remaining hope rather than the
thing being tested.

QUOTA
-----
One Gemini call per run. The free tier allows 20 per day PER
MODEL, and the client falls through several models, but a
12-day x 7-run sweep is 84 calls and will exhaust them. The
default samples three runs a day - morning, midday,
afternoon - which is 36 calls and still covers how the
forecast behaves as a day develops.

Resumable: a run whose schedule file already exists is
skipped, so an exhausted quota means "run it again tomorrow",
not "start over".

Run:  python -m tests.backtest_llm_pipeline
      python -m tests.backtest_llm_pipeline --from 2026-07-27 --to 2026-08-07
=========================================================
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.fusion.llm_scheduler import LLMScheduler
from modules.preprocessing.windy_features import (
    WindyFeatureBuilder, load_meter_history
)
from modules.storage.s3_client import S3Storage
from modules.vision.vision_module import VisionModule
from utils.logger import get_logger


DEFAULT_RUN_TIMES = ["09:45", "12:45", "15:45"]


class PipelineBacktest:

    def __init__(self, start, end, run_times=None):

        self.logger = get_logger()

        self.start = pd.Timestamp(start).date()
        self.end = pd.Timestamp(end).date()
        self.run_times = run_times or DEFAULT_RUN_TIMES

        self.storage = S3Storage()
        self.builder = WindyFeatureBuilder()
        self.scheduler = LLMScheduler()

        self.video_dir = Path(
            settings.get("windy_capture", {}).get(
                "video_dir", "data/windy/new_videos"
            )
        )

        self.output_dir = Path(
            settings.get("fusion", {}).get(
                "output_dir", "outputs/llm_schedules"
            )
        )

    # --------------------------------------------------

    def fetch_day_clips(self, day):
        """
        Pulls that day's satellite clips out of S3 into the local video
        folder, so the feature builder's normal same-day lookup finds
        them exactly as it would on a live run.

        Returns the paths it downloaded, so they can be removed after -
        222 clips left behind would be about a gigabyte.
        """

        prefix = f"{self.storage.video_prefix}/{day:%Y-%m-%d}/"

        self.video_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []

        try:
            paginator = self.storage.client.get_paginator("list_objects_v2")

            for page in paginator.paginate(
                Bucket=self.storage.bucket, Prefix=prefix
            ):
                for obj in page.get("Contents", []):

                    key = obj["Key"]

                    if not key.lower().endswith((".webm", ".mp4")):
                        continue

                    if VisionModule.parse_video_time(Path(key).name) is None:
                        continue

                    target = self.video_dir / Path(key).name

                    if not target.exists():
                        self.storage.client.download_file(
                            self.storage.bucket, key, str(target)
                        )
                        downloaded.append(target)

        except Exception as error:
            self.logger.warning(f"Clip fetch failed for {day} ({error})")

        return downloaded

    # --------------------------------------------------

    def schedule_path(self, run_time):

        stem = (
            f"{settings['plant'].get('code', 'SIRMOUR')}_"
            f"{run_time.strftime('%Y-%m-%d_%H-%M')}"
        )

        return self.output_dir / f"{stem}_schedule.csv"

    # --------------------------------------------------

    def run(self):

        meter = load_meter_history()

        if meter is None or meter.empty:
            raise SystemExit("No meter history - nothing to backtest against.")

        day = self.start

        done = 0
        skipped = 0
        failed = 0

        while day <= self.end:

            clips = self.fetch_day_clips(day)

            self.logger.info(f"{day}: {len(clips)} clip(s) fetched")

            for clock in self.run_times:

                hour, minute = map(int, clock.split(":"))

                run_time = pd.Timestamp(
                    year=day.year, month=day.month, day=day.day,
                    hour=hour, minute=minute,
                    tz=settings["plant"]["timezone"],
                )

                path = self.schedule_path(run_time)

                if path.exists():
                    skipped += 1
                    continue

                try:
                    features, info = self.builder.build(
                        run_time=run_time, meter=meter
                    )

                    schedule, meta = self.scheduler.decide(
                        features, run_time, satellite=info["satellite"]
                    )

                    self.scheduler.save(schedule, meta, run_time)

                    done += 1

                    print(
                        f"  {run_time:%Y-%m-%d %H:%M}  "
                        f"{len(schedule)} blocks  "
                        f"regime={meta.get('regime')}  "
                        f"adjusted={meta.get('adjusted_blocks')}"
                    )

                except Exception as error:
                    failed += 1
                    self.logger.warning(
                        f"{run_time:%Y-%m-%d %H:%M} failed: {error}"
                    )

                    # A quota refusal will hit every later run too, so
                    # stop rather than burning through the whole sweep
                    # logging the same message 30 times.
                    if "quota" in str(error).lower():
                        print("\nGemini daily quota exhausted - stopping. "
                              "Re-run tomorrow; finished runs are skipped.")
                        for clip in clips:
                            clip.unlink(missing_ok=True)
                        return done, skipped, failed

            for clip in clips:
                clip.unlink(missing_ok=True)

            day += timedelta(days=1)

        return done, skipped, failed


def main():

    parser = argparse.ArgumentParser(
        description="Backtest the LLM pipeline over past days"
    )
    parser.add_argument("--from", dest="start", default="2026-07-27")
    parser.add_argument("--to", dest="end", default="2026-08-07")
    parser.add_argument(
        "--run-times", default=",".join(DEFAULT_RUN_TIMES),
        help="comma-separated, e.g. 09:45,12:45,15:45"
    )

    args = parser.parse_args()

    backtest = PipelineBacktest(
        args.start, args.end, args.run_times.split(",")
    )

    print("=" * 70)
    print(f"BACKTESTING {args.start} to {args.end}, "
          f"run times {args.run_times}")
    print("  NOTE: scraped Windy values cannot be backfilled and are")
    print("        absent from every historical run - see the module docstring")
    print("=" * 70)

    done, skipped, failed = backtest.run()

    print("-" * 70)
    print(f"new schedules : {done}")
    print(f"already done  : {skipped}")
    print(f"failed        : {failed}")
    print("\nNow score them:  python -m tests.test_llm_approach_score")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
