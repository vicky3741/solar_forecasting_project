"""
=========================================================
Solar Forecasting Project
Full LLM Report: Gemini vs ChatGPT, every valid day
=========================================================
Combines what test_llm_comparison.py (raw readings) and
test_llm_honest_comparison.py (accuracy grading) do into
ONE report, across EVERY day that has both a Windy video
and real (measured, not reconstructed) meter data - not
just a single day.

For each such video:
  1. Analyze it with Gemini AND ChatGPT (GitHub Models free
     GPT-4o), same trimmed frame window for both, so the
     only thing that differs is which model read the
     frames. Cached per provider - safe to re-run for free.
  2. Feed each reading into the SAME forecast pipeline
     (identical physics/weather/Chronos - see
     modules/forecasting/predictor.py) and grade the result
     against REAL actual generation blocks only (Part 2's
     is_real_measurement flag - never reconstructed data).
  3. One combined row per video: raw agreement + accuracy
     for no-vision / Gemini / ChatGPT, plus an overall
     summary across every day.

Run:  python -m tests.test_llm_full_report
=========================================================
"""

import time
from pathlib import Path

import pandas as pd

from config.config import settings
from modules.vision.vision_module import VisionModule
from modules.forecasting.predictor import HybridPredictor
from modules.fusion.fusion import FeatureFusion
from modules.evaluation import metrics


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000

# The raw fields worth showing alongside the accuracy numbers, so the
# report answers both "what did each model say" and "was it right".
READING_FIELDS = [
    "cloud_coverage_pct",
    "cloud_density",
    "cloud_motion_direction",
    "expected_change",
    "trend_next_2h",
    "rain_probability_pct",
    "confidence",
]

# Same trim window validated on 2026-07-24: skips Windy's map-loading
# seconds and stops before the timeline loops back to the start, so
# both models see one clean forward sequence.
TRIM_START_FRACTION = 0.15
TRIM_END_FRACTION = 0.70
TARGET_FRAMES = 6

# Gemini's free tier allows roughly 10 requests per minute. Firing one
# video straight after another (even with the client's 5-10s retry
# backoff) trips that limit almost every time: on 2026-07-25 a batch of
# 15 videos produced 1 success and 14 "busy" failures, while a single
# isolated call the same minute worked fine. Pacing the loop is the
# actual fix - retries alone cannot beat a per-minute quota. Only
# applied when a video genuinely needs an API call, so fully-cached
# re-runs stay instant.
SECONDS_BETWEEN_API_VIDEOS = 8

# Only the auto-capture folder. The mentor's test protocol pairs each
# of the 7 official run-time videos with that slot's meter data, which
# only works when videos are recorded AT those times. The legacy
# hand-recorded clips (data/windy/videos, e.g. July 9 at 10:17, 14:16,
# 14:26, 21:19) were taken at arbitrary moments - they produced
# duplicate and nonsensical slots, so they are excluded. Every future
# capture lands here at the official times and is picked up
# automatically.
VIDEO_FOLDERS = [
    Path(settings["windy_capture"]["video_dir"]),
]

# Reuses the cache tests/test_llm_comparison.py already wrote, so days
# already analyzed are not re-sent to the (rate-limited, free) APIs.
CACHE_ROOT = Path("outputs/llm_compare")


def discover_videos():
    """Every video across both folders with a parseable timestamp."""

    found = []
    for folder in VIDEO_FOLDERS:
        if not folder.exists():
            continue
        for pattern in VisionModule.VIDEO_EXTENSIONS:
            for path in folder.glob(pattern):
                ts = VisionModule.parse_video_time(path.name)
                if ts is not None:
                    found.append((ts, path))

    found.sort(key=lambda item: item[0])
    return found


def is_cached(provider, video_path):
    """True if this provider already has a stored result for the video."""

    return (
        CACHE_ROOT / provider / Path(video_path).stem / "vision_result.json"
    ).exists()


def analyze_with_provider(provider, video_path):
    """Runs (or reuses the cache for) one provider on one video."""

    settings["vision"]["provider"] = provider

    try:
        vision = VisionModule()
    except Exception as error:
        return None, str(error)[:80]

    try:
        out = vision.analyze_video(
            str(video_path),
            str(CACHE_ROOT / provider),
            target_frames=TARGET_FRAMES,
            start_fraction=TRIM_START_FRACTION,
            end_fraction=TRIM_END_FRACTION
        )
        features = out["weather_features"]
        if "_error" in features:
            return None, features["_error"]
        return features, None
    except Exception as error:
        return None, str(error)[:80]


def build_time_slots(video_times, day_end_time="19:00"):
    """
    Gives each video its OWN slice of the day, as the mentor asked:
    a video is judged only on the period from when it was recorded
    until the NEXT video was recorded (the last video of the day runs
    to 19:00).

    This is what makes daily totals real: the 7 slices tile the day
    exactly once with no overlap. Previously every run forecast all
    the way to 19:00, so the runs overlapped and summing them counted
    the same generation up to 7 times.

    Returns {video_time: (slot_start, slot_end)}.
    """

    ordered = sorted(video_times)
    slots = {}

    for index, start in enumerate(ordered):

        if index + 1 < len(ordered):
            end = ordered[index + 1]
        else:
            hour, minute = map(int, day_end_time.split(":"))
            end = start.replace(hour=hour, minute=minute, second=0, microsecond=0)

        slots[start] = (start, end)

    return slots


def score_run(predictor, fusion, dataframe, run_time, vision_features, actual,
              slot=None):
    """
    Builds the forecast for one run with the given vision reading
    (None = no-vision baseline) and scores it against REAL actual
    blocks only.

    Returns (deviation_pct, blocks, predicted_mwh, actual_mwh) so the
    report can show energy in real units alongside the error
    percentage - a 9% deviation means little on its own, but
    "predicted 12.4 MWh vs actual 13.1 MWh" is immediately readable.
    """

    signals = predictor.compute_signals(dataframe, run_time)

    vision_adjustment = fusion.trend_adjustment_profile(
        vision_features, signals["horizon_minutes"]
    ) if vision_features else 0.0

    forecast = predictor.blend_signals(signals, vision_adjustment=vision_adjustment)

    comparison = forecast.merge(actual, on="timestamp", how="inner")

    # Score only this video's own slice of the day (see build_time_slots),
    # so each video is graded on the period it was actually recorded for.
    if slot is not None:
        slot_start, slot_end = slot
        comparison = comparison[
            (comparison["timestamp"] > slot_start)
            & (comparison["timestamp"] <= slot_end)
        ]

    if "is_real_measurement" in comparison.columns:
        comparison = comparison[comparison["is_real_measurement"].fillna(False)]

    if comparison.empty:
        return None, 0, None, None

    deviation = metrics.average_percentage_deviation(
        comparison["final_forecast_kw"], comparison["active_power_kw"], CAPACITY_KW
    )

    # Average POWER in MW over the run's blocks. MW (not MWh) is the
    # unit that compares directly against the 5.1 MW plant rating, so
    # every value here should sit below that ceiling.
    predicted_mw = comparison["final_forecast_kw"].mean() / 1000
    actual_mw = comparison["active_power_kw"].mean() / 1000

    return deviation, len(comparison), predicted_mw, actual_mw


def main():

    processed = pd.read_csv(
        "data/processed/processed_data.csv", parse_dates=["timestamp"]
    )

    actual_columns = ["timestamp", "active_power_kw"]
    if "is_real_measurement" in processed.columns:
        actual_columns.append("is_real_measurement")
        real_days = set(
            processed.loc[processed["is_real_measurement"], "timestamp"].dt.date
        )
    else:
        real_days = set(processed["timestamp"].dt.date)

    actual = processed[actual_columns]

    videos = discover_videos()
    usable = [(ts, path) for ts, path in videos if ts.date() in real_days]

    skipped_days = sorted({
        ts.date() for ts, _ in videos if ts.date() not in real_days
    })

    print("=" * 92)
    print("FULL LLM REPORT - Gemini vs ChatGPT, every day with a video AND real meter data")
    print("=" * 92)
    print(f"videos discovered total : {len(videos)}")
    print(f"usable (day has real actual data): {len(usable)}")
    print(f"days used   : {sorted({ts.date() for ts, _ in usable})}")
    if skipped_days:
        print(f"days SKIPPED (video exists, no real meter data yet): {skipped_days}")
    print()

    predictor = HybridPredictor()
    fusion = FeatureFusion()

    # Each video gets its own non-overlapping slice of its day.
    slots_by_day = {}
    for day in {ts.date() for ts, _ in usable}:
        day_times = [pd.Timestamp(ts).floor("15min")
                     for ts, _ in usable if ts.date() == day]
        slots_by_day.update(build_time_slots(day_times))

    rows = []

    for ts, path in usable:

        # The forecast is built on 15-min blocks starting just after
        # run_time, and actual meter data is resampled to the same
        # :00/:15/:30/:45 grid. Manually-recorded videos (July 9) were
        # captured at arbitrary times (e.g. 11:58, 14:10) that don't
        # land on that grid, so the generated forecast blocks never
        # matched a real actual timestamp and every such run was
        # silently skipped (0 rows for July 9 in the first pass).
        # Flooring to the nearest COMPLETED 15-min block fixes this
        # without any lookahead - it can only move run_time earlier,
        # by at most 14 minutes, never later.
        run_time = pd.Timestamp(ts).floor("15min")

        # Pace only when an actual API call is needed - a fully cached
        # re-run should stay instant.
        needs_api_call = not (
            is_cached("gemini", path) and is_cached("openai", path)
        )

        gemini_features, gem_err = analyze_with_provider("gemini", path)
        openai_features, gpt_err = analyze_with_provider("openai", path)

        if needs_api_call:
            time.sleep(SECONDS_BETWEEN_API_VIDEOS)

        slot = slots_by_day.get(run_time)

        base_dev, n, _, actual_mw = score_run(
            predictor, fusion, processed, run_time, None, actual, slot
        )

        if base_dev is None:
            continue

        gem_dev, _, gem_mw, _ = (
            score_run(predictor, fusion, processed, run_time, gemini_features,
                      actual, slot)
            if gemini_features else (None, 0, None, None)
        )
        gpt_dev, _, gpt_mw, _ = (
            score_run(predictor, fusion, processed, run_time, openai_features,
                      actual, slot)
            if openai_features else (None, 0, None, None)
        )

        # Which of the TWO MODELS produced the forecast closest to what
        # the plant actually generated. The no-vision control is still
        # computed for the summary, but it is deliberately not a
        # candidate here - this column answers "Gemini or ChatGPT?",
        # so it must always name one of them.
        model_candidates = {}
        if gem_dev is not None:
            model_candidates["gemini"] = gem_dev
        if gpt_dev is not None:
            model_candidates["chatgpt"] = gpt_dev

        if not model_candidates:
            best_result = "neither (both failed)"
        elif len(model_candidates) == 1:
            # Only one model returned a reading for this run.
            best_result = next(iter(model_candidates))
        else:
            gap = abs(model_candidates["gemini"] - model_candidates["chatgpt"])
            # Below ~0.05 pts (~2.5 kW on this plant) the difference is
            # smaller than ordinary forecast noise, so naming a winner
            # would read as a real result when it is not.
            best_result = (
                "tie"
                if gap < 0.05
                else min(model_candidates, key=model_candidates.get)
            )

        row = {
            "date": ts.date().isoformat(),
            "run_time": ts.strftime("%H:%M"),
            # The slice of the day this video is graded on.
            "covers_period": (
                f"{slot[0].strftime('%H:%M')}-{slot[1].strftime('%H:%M')}"
                if slot else ""
            ),
            "video": path.name,
            # How many 15-min blocks this run was graded on. Only blocks
            # with a REAL meter reading count (gap-filled ones are
            # excluded), so this doubles as the evidence weight per row.
            "blocks_compared_15min": n,
            # All three are the same measure: average absolute deviation
            # from actual generation, as a % of plant capacity. Lower is
            # better. no_vision is the control - the forecast with no
            # video input at all, which both LLMs must beat to be worth
            # having.
            "deviation_no_vision_pct": round(base_dev, 3),
            "deviation_gemini_pct": round(gem_dev, 3) if gem_dev is not None else None,
            "deviation_chatgpt_pct": round(gpt_dev, 3) if gpt_dev is not None else None,
            # Average POWER in MW across this run's blocks - directly
            # comparable to the 5.1 MW plant rating, so every value
            # here must sit below that ceiling.
            "actual_power_generation_mw": (
                round(actual_mw, 3) if actual_mw is not None else None
            ),
            "gemini_predicted_power_generation_mw": (
                round(gem_mw, 3) if gem_mw is not None else None
            ),
            "chatgpt_predicted_power_generation_mw": (
                round(gpt_mw, 3) if gpt_mw is not None else None
            ),
            "best_result": best_result,
        }

        # What each model actually said, alongside the accuracy verdict.
        for field in READING_FIELDS:
            row[f"gemini_{field}"] = (gemini_features or {}).get(field)
            row[f"chatgpt_{field}"] = (openai_features or {}).get(field)

        # "ok" rather than blank, so an empty cell is never ambiguous
        # between "no problem" and "we forgot to record something".
        row["gemini_status"] = gem_err or "ok"
        row["chatgpt_status"] = gpt_err or "ok"

        rows.append(row)

        gem_str = f"{gem_dev:6.2f}%" if gem_dev is not None else "   -  "
        gpt_str = f"{gpt_dev:6.2f}%" if gpt_dev is not None else "   -  "
        print(f"{ts.strftime('%Y-%m-%d %H:%M')}  ({n:2d} blocks)  "
              f"deviation: no-vision={base_dev:6.2f}%  "
              f"gemini={gem_str}  chatgpt={gpt_str}  -> best: {best_result}")

    print("-" * 92)

    if not rows:
        print("\nNo gradable runs found.")
        return

    results = pd.DataFrame(rows)

    # The no-vision control is still computed (it drives best_result and
    # the "is vision worth it at all" question), but it and the per-model
    # status columns are dropped from the saved detail file so the report
    # is a clean model-vs-model comparison. A failed analysis still shows
    # up as a blank deviation cell for that model.
    out = Path("outputs/reports/llm_full_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results.drop(
        columns=["deviation_no_vision_pct", "gemini_status", "chatgpt_status"]
    ).to_csv(out, index=False)

    print()
    print("=" * 92)
    print("OVERALL SUMMARY (this IS the combined report)")
    print("=" * 92)
    print(f"total runs graded : {len(results)}")
    print(f"days covered      : {sorted(results['date'].unique())}")
    # Compare only on runs where ALL THREE produced a forecast, so the
    # averages are like-for-like (a model must not look better simply
    # because it skipped the hardest runs).
    complete = results.dropna(
        subset=["deviation_gemini_pct", "deviation_chatgpt_pct"]
    )

    # The headline table compares the two LLMs only. The no-vision
    # control is still computed and kept per-run in the CSV (it is what
    # tells us whether vision helps at all), it is just not part of the
    # model-vs-model verdict.
    approaches = {
        "Gemini (free)": "deviation_gemini_pct",
        "ChatGPT / GPT-4o (paid)": "deviation_chatgpt_pct",
    }

    print()
    print(f"Compared on {len(complete)} runs where both models produced a forecast")
    print("(average deviation from ACTUAL generation, % of capacity - lower is better)")
    print()

    ranking = sorted(
        ((label, complete[col].mean()) for label, col in approaches.items()),
        key=lambda item: item[1]
    )

    best_label, best_value = ranking[0]

    print(f"  {'rank':<5} {'model':<28} {'avg deviation':>14}")
    print("  " + "-" * 50)
    for position, (label, value) in enumerate(ranking, start=1):
        print(f"  {position:<5} {label:<28} {value:>13.3f}%")

    # The headline table as its own small CSV, so the verdict can be
    # handed over without anyone having to re-derive it from the 16
    # per-run rows in the detail report.
    verdict_table = pd.DataFrame([
        {
            "rank": position,
            "model": label,
            "avg_deviation_pct": round(value, 3),
        }
        for position, (label, value) in enumerate(ranking, start=1)
    ])

    verdict_path = Path("outputs/reports/llm_verdict_table.csv")
    verdict_table.to_csv(verdict_path, index=False)

    print()
    # Power generation per day, from the FIRST run of that day only.
    # The 7 daily runs all forecast OVERLAPPING windows of the SAME day
    # (06:45 covers nearly the whole day, 15:45 only the last hours), so
    # summing the rows counts the same energy up to 7 times - that is
    # what produced an impossible 110 MWh total for a 5.1 MW plant.
    # The earliest run covers the most of the day, so it gives the
    # meaningful day-level actual-vs-forecast picture.
    # Now that each video covers its own non-overlapping slice, the
    # day's slices can legitimately be combined - averaging them
    # weighted by how many blocks each covers gives the true daily
    # average power.
    print("Average power per day, combining each video's own time slot")
    print(f"(plant rating {CAPACITY_KW/1000:.1f} MW - all values must be below it):")
    print()
    print(f"  {'date':<12} {'slots':>6} {'actual':>9} {'Gemini':>9} {'ChatGPT':>9}")
    print("  " + "-" * 50)

    for date in sorted(complete["date"].unique()):
        day = complete[complete["date"] == date]
        weights = day["blocks_compared_15min"]

        def weighted(column):
            return (day[column] * weights).sum() / weights.sum()

        print(f"  {date:<12} {len(day):>6} "
              f"{weighted('actual_power_generation_mw'):>9.3f} "
              f"{weighted('gemini_predicted_power_generation_mw'):>9.3f} "
              f"{weighted('chatgpt_predicted_power_generation_mw'):>9.3f}")

    print("  (MW - averaged across the day's non-overlapping slots)")

    print()
    print("best result per run (which model was closer to actual generation):")
    print(results["best_result"].value_counts().to_string())

    # ---- plain-language verdict ----
    spread = ranking[-1][1] - ranking[0][1]
    capacity_kw_gap = spread / 100 * CAPACITY_KW

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"Lowest average deviation : {best_label} ({best_value:.3f}%)")
    print(f"Spread best-to-worst     : {spread:.3f} percentage points "
          f"(about {capacity_kw_gap:.1f} kW on a {CAPACITY_KW/1000:.1f} MW plant)")
    print()

    # A gap this small is smaller than ordinary sensor/model noise, so
    # calling a "winner" on it would overstate what the data supports.
    if spread < 0.5:
        print("These are effectively TIED - the gap is far smaller than normal")
        print("forecast noise, so no approach is meaningfully more accurate here.")
        print("On this evidence there is no accuracy case for paying for ChatGPT,")
        print("and the vision signal (either model) is not yet adding much over")
        print("physics + weather alone.")
    else:
        print(f"{best_label} is meaningfully ahead on this sample.")

    print()
    print(f"NOTE: based on {len(complete)} runs across {results['date'].nunique()} "
          f"day(s) - a small sample. Re-run as more days of paired")
    print("video + meter data become available before drawing a firm conclusion.")

    print(f"\nSaved per-run detail : {out}")
    print(f"Saved verdict table  : {verdict_path}")


if __name__ == "__main__":
    main()
