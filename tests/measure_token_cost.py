"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Measure what this pipeline actually costs in tokens
=========================================================
Produces the numbers for the token-consumption report, the
same way Team 1 produced theirs: input tokens from Gemini's
countTokens API, output and thinking tokens read from
usage_metadata after a real generateContent call.

NOTHING HERE IS ESTIMATED FROM CHARACTER COUNTS. Every figure
is either counted by Google's own tokenizer or read off a
real response.

WHAT MAKES OUR COST PROFILE DIFFERENT
-------------------------------------
Two structural differences from the reference report, both of
which move the number and both of which are visible in the
measurements rather than argued:

  * NO IMAGES. Their call attaches two screenshots (satellite
    + clouds) costing ~2,184 input tokens a call. Ours sends
    the satellite clip as NUMBERS - thin/thick cloud, entropy,
    flow divergence - computed by OpenCV before the call. Same
    information, no image tokens.

  * WHOLE DAY, NOT TWO HOURS. Theirs forecasts 8 blocks
    ahead. Ours forecasts from the run time to 19:00, which is
    up to 47 blocks at the 06:45 run and 11 at 15:45. So our
    prompt and response both vary by time of day, and a single
    call is not representative - the report needs the
    per-run-time spread, which is why every saved prompt is
    counted rather than one.

Run:  python -m tests.measure_token_cost            # counts only
      python -m tests.measure_token_cost --live     # + one real call
=========================================================
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings


_RUN_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})")


def prompt_files(folder):

    return sorted(Path(folder).glob("*_prompt.txt"))


def run_clock(path):
    """The scheduling time a prompt belongs to, as HH:MM."""

    match = _RUN_STAMP.search(path.name)

    return f"{match.group(2)}:{match.group(3)}" if match else None


def count_tokens(client, model, texts):
    """
    Exact input token counts from Google's tokenizer. countTokens does
    not consume generation quota, so every saved prompt can be measured.
    """

    counts = []

    for text in texts:
        result = client.models.count_tokens(model=model, contents=text)
        counts.append(int(result.total_tokens))

    return counts


def main():

    parser = argparse.ArgumentParser(
        description="Measure this pipeline's real token consumption"
    )
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )
    parser.add_argument(
        "--live", action="store_true",
        help="also make ONE real call to read output and thinking tokens"
    )
    parser.add_argument("--out", default="outputs/reports/token_measurements.json")
    parser.add_argument(
        "--model", default=None,
        help="measure a different model than the configured one. Token "
             "counts and thinking volume are model-specific, so switching "
             "models means re-measuring, not re-pricing the old numbers."
    )

    args = parser.parse_args()

    paths = prompt_files(args.folder)

    if not paths:
        raise SystemExit(f"No saved prompts in {args.folder}")

    from modules.vision.gemini_client import GeminiClient

    gemini = GeminiClient()
    model = args.model or settings.get("fusion", {}).get("model") or gemini.model

    print("=" * 78)
    print("TOKEN MEASUREMENT")
    print(f"  {len(paths)} saved prompts from {args.folder}")
    print(f"  model: {model}")
    print("=" * 78)

    texts = [path.read_text(encoding="utf-8") for path in paths]

    print("\ncounting input tokens (countTokens - no generation quota)...")

    counts = count_tokens(gemini.client, model, texts)

    frame = pd.DataFrame({
        "prompt": [p.name for p in paths],
        "run_time": [run_clock(p) for p in paths],
        "chars": [len(t) for t in texts],
        "input_tokens": counts,
    })

    by_run = frame.groupby("run_time")["input_tokens"].agg(
        ["count", "mean", "min", "max"]
    ).round(0).astype(int)

    print("\nINPUT TOKENS BY SCHEDULING TIME")
    print(by_run.to_string())

    print(f"\nmean input tokens per call : {frame['input_tokens'].mean():,.0f}")
    print(f"min / max                  : {frame['input_tokens'].min():,} / "
          f"{frame['input_tokens'].max():,}")

    # One full day at the plant's real scheduling times, using the
    # measured mean for each - NOT the overall mean seven times, because
    # the 06:45 prompt is much longer than the 15:45 one.
    run_times = settings["forecast"]["run_times"]

    per_run_mean = frame.groupby("run_time")["input_tokens"].mean().to_dict()

    covered = [t for t in run_times if t in per_run_mean]

    daily_input = sum(per_run_mean[t] for t in covered)

    print(f"\nscheduling times measured  : {len(covered)} of {len(run_times)} "
          f"({', '.join(covered)})")
    print(f"input tokens for one day   : {daily_input:,.0f}")

    measurements = {
        "model": model,
        "prompts_measured": len(frame),
        "days_covered": int(
            frame["prompt"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0].nunique()
        ),
        "input_tokens_mean": float(frame["input_tokens"].mean()),
        "input_tokens_min": int(frame["input_tokens"].min()),
        "input_tokens_max": int(frame["input_tokens"].max()),
        "input_tokens_by_run_time": {
            k: float(v) for k, v in per_run_mean.items()
        },
        "run_times_measured": covered,
        "run_times_configured": run_times,
        "daily_input_tokens": float(daily_input),
        "images_attached": 0,
        "output": None,
    }

    if args.live:

        # ONE CALL PER SCHEDULING TIME, not one call overall. Our prompt
        # asks for every block from the run time to 19:00 - 47 blocks at
        # 06:45, 11 at 15:45 - so the response length varies by a factor
        # of four across the day. Measuring a single call and multiplying
        # by seven would misreport the daily total in whichever direction
        # that call happened to fall.
        print(f"\nmaking {len(covered)} real calls, one per scheduling time...")

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=settings["fusion"].get("temperature", 0.1),
            max_output_tokens=settings["fusion"].get("max_output_tokens", 8192),
        )

        per_run_output = {}

        for clock in covered:

            candidates = frame[frame["run_time"] == clock]

            # The longest prompt at this time, so each figure is that
            # run's worst case rather than a flattering one.
            index = candidates["input_tokens"].idxmax()

            response = gemini.client.models.generate_content(
                model=model, contents=[texts[index]], config=config
            )

            usage = response.usage_metadata

            visible = int(getattr(usage, "candidates_token_count", 0) or 0)
            thinking = int(getattr(usage, "thoughts_token_count", 0) or 0)

            per_run_output[clock] = {
                "prompt_token_count": int(
                    getattr(usage, "prompt_token_count", 0) or 0
                ),
                "visible_output_tokens": visible,
                "thinking_tokens": thinking,
                "billable_output_tokens": visible + thinking,
                "measured_on_prompt": frame.loc[index, "prompt"],
            }

            print(f"  {clock}  input {per_run_output[clock]['prompt_token_count']:>6,}"
                  f"   visible {visible:>5,}   thinking {thinking:>6,}"
                  f"   billable out {visible + thinking:>6,}")

        daily_output = sum(
            entry["billable_output_tokens"] for entry in per_run_output.values()
        )

        measurements["output"] = {
            "by_run_time": per_run_output,
            "daily_billable_output_tokens": daily_output,
            "daily_visible_output_tokens": sum(
                e["visible_output_tokens"] for e in per_run_output.values()
            ),
            "daily_thinking_tokens": sum(
                e["thinking_tokens"] for e in per_run_output.values()
            ),
        }

        print(f"\nbillable output for one day : {daily_output:,}")
        print(f"  of which thinking          : "
              f"{measurements['output']['daily_thinking_tokens']:,}")
        print(f"  of which visible           : "
              f"{measurements['output']['daily_visible_output_tokens']:,}")
        print(f"\ntotal tokens for one day    : "
              f"{daily_input + daily_output:,.0f}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")

    frame.to_csv(
        output_path.with_name("token_measurements_per_prompt.csv"), index=False
    )

    print(f"\nSaved: {output_path}")

    if not args.live:
        print("\nNo output tokens measured. Re-run with --live to make ONE "
              "real call\nand read output + thinking tokens from "
              "usage_metadata.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
