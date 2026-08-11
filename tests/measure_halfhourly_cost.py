"""
=========================================================
Solar Forecasting Project - NEW APPROACH
What a half-hourly cadence would actually cost in tokens
=========================================================
The production cadence is seven schedule generations a day
(06:45 ... 15:45, 90 minutes apart). Team 1's timeslot report
also priced a HALF-HOURLY cadence - a fresh schedule every 30
minutes from 07:00 to 17:30, 22 calls - by linearly
interpolating their measured figures against block count.

We can do better than interpolate, because LLMScheduler has a
dry_run mode: it builds the real prompt for any run time and
returns it WITHOUT spending a generation request. So every one
of the 22 half-hourly prompts can be built for real and counted
with Google's own tokenizer, and countTokens does not consume
generation quota.

WHAT IS MEASURED HERE AND WHAT IS NOT
-------------------------------------
Measured: input tokens, per half-hourly slot, from real prompts
built on real days - block table, meter history to that minute,
retrieved precedent, track record, satellite observation, the
lot.

NOT measured: output and thinking tokens. Those need a real
generateContent call, and 22 of them a day is above the free
tier's 20-per-model cap. tests/measure_token_cost.py --live
measures output at the seven production times; the report
interpolates output across the half-hourly slots from those
seven points and says so.

Same caveat as the backtest: scraped Windy values cannot be
backfilled, so historical prompts carry empty Windy columns.
That makes these figures a floor for a live half-hourly day,
not a ceiling.

Run:  python -m tests.measure_halfhourly_cost
      python -m tests.measure_halfhourly_cost --from 2026-08-01 --to 2026-08-07
      python -m tests.measure_halfhourly_cost --clocks 06:45,08:15,09:45
=========================================================
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path
from statistics import mean

import pandas as pd

from config.config import settings


def half_hourly_clocks(first="07:00", last="17:30", step_minutes=30):
    """Every scheduling time on the alternative cadence, as HH:MM."""

    start = pd.Timestamp(f"2000-01-01 {first}")
    end = pd.Timestamp(f"2000-01-01 {last}")

    clocks = []

    while start <= end:
        clocks.append(f"{start:%H:%M}")
        start += pd.Timedelta(minutes=step_minutes)

    return clocks


def main():

    parser = argparse.ArgumentParser(
        description="Measure input tokens for a half-hourly cadence"
    )
    parser.add_argument("--from", dest="start", default="2026-08-01")
    parser.add_argument("--to", dest="end", default="2026-08-07")
    parser.add_argument(
        "--clocks", default=None,
        help="comma-separated run times; default is the half-hourly grid"
    )
    parser.add_argument("--first", default="07:00")
    parser.add_argument("--last", default="17:30")
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument(
        "--no-clips", action="store_true",
        help="skip the S3 satellite-clip fetch (faster, slightly fewer tokens)"
    )
    parser.add_argument(
        "--out", default="outputs/reports/halfhourly_measurements.json"
    )
    parser.add_argument(
        "--cache", default="outputs/reports/cadence_cache",
        help="where built prompts and their counts are kept. Both phases "
             "resume from here: building a day's prompts costs minutes of "
             "pvlib and S3, and free-tier countTokens throttles hard, so "
             "neither should be lost to an interrupted run."
    )

    args = parser.parse_args()

    clocks = (
        [c.strip() for c in args.clocks.split(",")]
        if args.clocks else
        half_hourly_clocks(args.first, args.last, args.step)
    )

    from modules.fusion.llm_scheduler import LLMScheduler
    from modules.preprocessing.windy_features import (
        WindyFeatureBuilder, load_meter_history
    )

    builder = WindyFeatureBuilder()
    scheduler = LLMScheduler()

    meter = load_meter_history()

    if meter is None or meter.empty:
        raise SystemExit("No meter history - nothing to build prompts from.")

    timezone = settings["plant"]["timezone"]

    fetch_clips = None

    if not args.no_clips:
        from tests.backtest_llm_pipeline import PipelineBacktest

        fetch_clips = PipelineBacktest(args.start, args.end, clocks)

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / "index.json"

    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists() else {}
    )

    def flush_index():
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def prompt_file(key):
        """
        Cache path for one prompt. The colon in a run time MUST NOT reach
        the filename: on NTFS "2026-08-05_08:15.txt" writes an alternate
        data stream named "15.txt" hanging off a 0-byte file called
        "2026-08-05_08". It round-trips within one machine and looks
        empty everywhere else, which is the worst kind of working.
        """

        return cache_dir / f"{key.replace(':', '-')}.txt"

    print("=" * 78)
    print(f"HALF-HOURLY PROMPT MEASUREMENT  {args.start} to {args.end}")
    print(f"  {len(clocks)} run times: {', '.join(clocks)}")
    print("  dry_run - no generation request is spent building these")
    print(f"  cache: {cache_dir}  ({len(index)} prompt(s) already built)")
    print("=" * 78)

    wanted = []

    day = pd.Timestamp(args.start).date()
    end_day = pd.Timestamp(args.end).date()

    while day <= end_day:
        for clock in clocks:
            wanted.append((str(day), clock))
        day += timedelta(days=1)

    day = pd.Timestamp(args.start).date()

    while day <= end_day:

        todo = [
            clock for clock in clocks
            if f"{day}_{clock}" not in index
        ]

        if not todo:
            print(f"\n{day}  all {len(clocks)} prompt(s) already cached")
            day += timedelta(days=1)
            continue

        clips = fetch_clips.fetch_day_clips(day) if fetch_clips else []

        print(f"\n{day}  ({len(clips)} clip(s), {len(todo)} to build)")

        for clock in todo:

            hour, minute = map(int, clock.split(":"))

            run_time = pd.Timestamp(
                year=day.year, month=day.month, day=day.day,
                hour=hour, minute=minute, tz=timezone,
            )

            try:
                features, info = builder.build(run_time=run_time, meter=meter)

                _, meta = scheduler.decide(
                    features, run_time, dry_run=True,
                    satellite=info["satellite"],
                )

            except Exception as error:
                print(f"  {clock}  skipped ({error})")
                continue

            key = f"{day}_{clock}"

            # Written to disk BEFORE the index records it, so an index
            # entry always has a prompt behind it.
            prompt_file(key).write_text(meta["prompt"], encoding="utf-8")

            index[key] = {
                "day": str(day),
                "run_time": clock,
                "blocks": int(meta["blocks_requested"]),
                "chars": int(meta["prompt_chars"]),
                "has_satellite": bool(meta.get("satellite_clip")),
                "input_tokens": None,
            }

            flush_index()

            print(f"  {clock}  {meta['blocks_requested']:>3} blocks  "
                  f"{meta['prompt_chars']:>6,} chars")

        for clip in clips:
            clip.unlink(missing_ok=True)

        day += timedelta(days=1)

    records = [
        dict(index[f"{day}_{clock}"])
        for day, clock in wanted
        if f"{day}_{clock}" in index
    ]

    if not records:
        raise SystemExit("No prompts were built.")

    # ---- count with Google's tokenizer ----
    # countTokens does not consume generation quota, so every prompt
    # built above can be counted rather than a sample of them.
    from modules.vision.gemini_client import GeminiClient

    gemini = GeminiClient()
    model = settings.get("fusion", {}).get("model") or gemini.model

    uncounted = [
        (day, clock) for day, clock in wanted
        if f"{day}_{clock}" in index
        and index[f"{day}_{clock}"].get("input_tokens") is None
    ]

    print(f"\ncounting {len(uncounted)} prompt(s) with countTokens ({model}); "
          f"{len(records) - len(uncounted)} already counted")

    if uncounted:
        print("  free-tier countTokens throttles - each count is written to "
              "the cache as it lands, so an interrupted run resumes here")

    for position, (day, clock) in enumerate(uncounted, 1):

        key = f"{day}_{clock}"

        text = prompt_file(key).read_text(encoding="utf-8")

        index[key]["input_tokens"] = int(
            gemini.client.models.count_tokens(
                model=model, contents=text
            ).total_tokens
        )

        flush_index()

        print(f"  {position}/{len(uncounted)}  {key}  "
              f"{index[key]['input_tokens']:,} tokens", flush=True)

    records = [
        dict(index[f"{day}_{clock}"])
        for day, clock in wanted
        if f"{day}_{clock}" in index
        and index[f"{day}_{clock}"].get("input_tokens") is not None
    ]

    frame = pd.DataFrame(records)

    by_clock = {}

    for clock in clocks:

        rows = frame[frame["run_time"] == clock]

        if rows.empty:
            continue

        by_clock[clock] = {
            "days": int(len(rows)),
            "blocks": int(rows["blocks"].iloc[0]),
            "blocks_varies": bool(rows["blocks"].nunique() > 1),
            "input_tokens_mean": float(rows["input_tokens"].mean()),
            "input_tokens_min": int(rows["input_tokens"].min()),
            "input_tokens_max": int(rows["input_tokens"].max()),
            "days_with_satellite": int(rows["has_satellite"].sum()),
        }

    measured = {
        "model": model,
        "days": sorted(frame["day"].unique().tolist()),
        "prompts_measured": int(len(frame)),
        "run_times": [c for c in clocks if c in by_clock],
        "by_run_time": by_clock,
        "daily_text_input_tokens": float(
            sum(entry["input_tokens_mean"] for entry in by_clock.values())
        ),
        "total_blocks_per_day": int(
            sum(entry["blocks"] for entry in by_clock.values())
        ),
        "windy_columns_present": False,
        "note": (
            "Input tokens only. Output and thinking tokens need real "
            "generateContent calls and are measured at the seven production "
            "times by tests/measure_token_cost.py --live."
        ),
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(measured, indent=2), encoding="utf-8")

    frame.to_csv(
        output_path.with_name("halfhourly_measurements_per_prompt.csv"),
        index=False,
    )

    print("\n" + "=" * 78)
    print(f"{'run':>6} {'blocks':>7} {'days':>5} {'mean input':>12} "
          f"{'min':>8} {'max':>8}")
    print("-" * 78)

    for clock in measured["run_times"]:

        entry = by_clock[clock]

        print(f"{clock:>6} {entry['blocks']:>7} {entry['days']:>5} "
              f"{entry['input_tokens_mean']:>12,.0f} "
              f"{entry['input_tokens_min']:>8,} "
              f"{entry['input_tokens_max']:>8,}")

    print("-" * 78)
    print(f"text input for one day : "
          f"{measured['daily_text_input_tokens']:,.0f} tokens over "
          f"{len(measured['run_times'])} calls")
    print(f"blocks scheduled a day : {measured['total_blocks_per_day']:,}")
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
