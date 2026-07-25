"""
=========================================================
Solar Forecasting Project
Preprocessing A/B Test: does OpenCV cleanup fix ChatGPT's
constant-answer bias?
=========================================================
Direct test of the mentor's question ("are you pre-
processing video before feeding it to AI models?") against
a specific, already-observed problem: ChatGPT (GPT-4o via
GitHub Models) returned an almost identical reading for
every one of 16 videos tested in test_llm_full_report.py
("clouding" / "decreasing" on 16/16), while Gemini's answers
varied genuinely. One plausible cause: raw Windy frames are
low-contrast/washed-out and ChatGPT is falling back to a
generic answer rather than discriminating.

This runs the SAME small set of videos through ChatGPT
twice - once with raw frames (current default), once with
modules/vision/frame_preprocessor.py applied (UI crop +
CLAHE contrast) - and reports whether the preprocessed run
produces more VARIED, video-specific answers. Gemini is
included too, as a sanity check that preprocessing does not
make its (already-working) answers worse.

Small sample size deliberately - this is a quick diagnostic,
not the full validation. If it looks promising, re-run the
full tests/test_llm_full_report.py-style comparison with
preprocess=True across more days before trusting it.

Run:  python -m tests.test_llm_preprocessing
=========================================================
"""

from pathlib import Path

from config.config import settings
from modules.vision.vision_module import VisionModule


SAMPLE_SIZE = 4
TRIM_START_FRACTION = 0.15
TRIM_END_FRACTION = 0.70
TARGET_FRAMES = 6

FIELDS = ["cloud_coverage_pct", "cloud_density", "expected_change", "trend_next_2h"]


def analyze(provider, video, preprocess):

    settings["vision"]["provider"] = provider
    vision = VisionModule()

    cache_root = f"outputs/llm_preprocess_test/{provider}_{'pre' if preprocess else 'raw'}"

    out = vision.analyze_video(
        str(video), cache_root,
        target_frames=TARGET_FRAMES,
        start_fraction=TRIM_START_FRACTION,
        end_fraction=TRIM_END_FRACTION,
        preprocess=preprocess
    )
    return out["weather_features"]


def main():

    videos = sorted(
        Path(settings["windy_capture"]["video_dir"]).glob("SIRMOUR_*.webm")
    )[-SAMPLE_SIZE:]

    if not videos:
        print("No videos found.")
        return

    print("=" * 88)
    print("PREPROCESSING A/B TEST - does OpenCV cleanup change ChatGPT's answers?")
    print("=" * 88)
    print("videos:", ", ".join(v.name for v in videos))
    print()

    rows = {}

    for provider in ["chatgpt", "gemini"]:
        api_provider = "openai" if provider == "chatgpt" else "gemini"

        for video in videos:
            for mode, preprocess in [("raw", False), ("pre", True)]:
                try:
                    features = analyze(api_provider, video, preprocess)
                except Exception as error:
                    features = {"_error": str(error)[:60]}

                rows[(provider, video.name, mode)] = features

    for provider in ["chatgpt", "gemini"]:
        print(f"--- {provider.upper()} ---")
        print(f"  {'video':<36} {'mode':<5} " + " ".join(f"{f:<18}" for f in FIELDS))

        for video in videos:
            for mode in ["raw", "pre"]:
                f = rows[(provider, video.name, mode)]
                if "_error" in f:
                    print(f"  {video.name:<36} {mode:<5} ERROR: {f['_error']}")
                    continue
                values = " ".join(f"{str(f.get(field, '-')):<18}" for field in FIELDS)
                print(f"  {video.name:<36} {mode:<5} {values}")
        print()

    # Distinctness check: does "pre" show MORE variety than "raw"?
    print("=" * 88)
    print("VERDICT - distinct answers across the sample (higher = more discriminating)")
    print("=" * 88)
    for provider in ["chatgpt", "gemini"]:
        for mode in ["raw", "pre"]:
            for field in FIELDS:
                values = [
                    str(rows[(provider, v.name, mode)].get(field, "-"))
                    for v in videos
                    if "_error" not in rows[(provider, v.name, mode)]
                ]
                distinct = len(set(values))
                print(f"  {provider:<8} {mode:<5} {field:<20} {distinct} distinct / {len(values)} videos")


if __name__ == "__main__":
    main()
