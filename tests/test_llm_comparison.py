"""
=========================================================
Solar Forecasting Project
LLM Comparison: Gemini vs ChatGPT (OpenAI)
=========================================================
Runs BOTH vision LLMs on the SAME Windy videos, with the
SAME frames and the SAME prompt, and prints their cloud
readings side by side. This is the fair, evidence-based
test the mentor asked for: the only thing that differs
between the two columns is which model read the frames.

Because the two providers are fully independent, this runs
whichever ones have a key. With no OPENAI_API_KEY it simply
reports the Gemini side and tells you what to add to unlock
the ChatGPT side - it never crashes on a missing key.

Cost note: OpenAI is a PAID API. SAMPLE_SIZE is deliberately
small so a comparison run costs a few cents, not dollars.
Raise it once you have decided it is worth it.

Run:  python -m tests.test_llm_comparison
=========================================================
"""

from pathlib import Path

from config.config import settings
from modules.vision.vision_module import VisionModule


PROVIDERS = ["gemini", "openai"]

SAMPLE_SIZE = 2   # videos to test - keep small; OpenAI charges per call

TEST_DATE_PREFIX = "SIRMOUR_2026-07-24"   # today's videos only

# Trim each clip before sampling frames: skip the ~first 15% (Windy's
# map still loading) and stop at ~70% (before the timeline loops back
# to the start) - see FrameExtractor.extract_frames. Without this the
# frame sequence is not one clean forward pass, which is what the
# prompt tells the model it's looking at.
TRIM_START_FRACTION = 0.15
TRIM_END_FRACTION = 0.70

# GitHub Models' free tier caps input at 8K tokens/request; each frame
# costs a few hundred tokens, so fewer frames keeps us safely under
# that limit for the comparison.
COMPARE_TARGET_FRAMES = 6

# The fields both models are asked to produce (see prompt_builder).
FIELDS = [
    "cloud_coverage_pct",
    "cloud_density",
    "cloud_motion_direction",
    "cloud_motion_speed",
    "moving_toward_plant",
    "expected_change",
    "trend_next_2h",
    "rain_probability_pct",
    "confidence",
]


def run_provider(provider, videos, preprocess=False):
    """Analyse every sample video with one provider. Never raises."""

    settings["vision"]["provider"] = provider

    try:
        vision = VisionModule()
    except Exception as error:
        return None, f"unavailable - {str(error)[:70]}"

    # Preprocessed and raw results are cached separately so switching
    # this flag never reads a stale answer from the other mode.
    cache_root = f"outputs/llm_compare/{provider}{'_preprocessed' if preprocess else ''}"

    results = {}
    for video in videos:
        try:
            # Separate cache folder per provider so one never reads
            # the other's cached answer. Both providers get the SAME
            # trimmed, clean frame window - the only thing that
            # differs between them is which model reads the frames.
            out = vision.analyze_video(
                str(video), cache_root,
                target_frames=COMPARE_TARGET_FRAMES,
                start_fraction=TRIM_START_FRACTION,
                end_fraction=TRIM_END_FRACTION,
                preprocess=preprocess
            )
            results[video.name] = out["weather_features"]
        except Exception as error:
            results[video.name] = {"_error": str(error)[:80]}

    return results, "ok"


def main():

    videos = sorted(
        Path(settings["windy_capture"]["video_dir"]).glob(f"{TEST_DATE_PREFIX}*.webm")
    )[-SAMPLE_SIZE:]

    if not videos:
        print(f"No {TEST_DATE_PREFIX}*.webm videos found to compare.")
        return

    print("=" * 70)
    print("LLM COMPARISON - Gemini vs ChatGPT on the SAME frames")
    print("=" * 70)
    print("videos:", ", ".join(v.name for v in videos))
    print(f"trim window: {TRIM_START_FRACTION:.0%}-{TRIM_END_FRACTION:.0%} of each clip "
          f"(skips map-loading + timeline loop-back), {COMPARE_TARGET_FRAMES} frames each")
    print()

    outcomes = {}
    for provider in PROVIDERS:
        results, status = run_provider(provider, videos)
        outcomes[provider] = results
        label = provider.upper()
        if status != "ok":
            print(f"  {label:8s}: {status}")
            if provider == "openai":
                print("            -> add OPENAI_API_KEY to your .env to run the ChatGPT side")
        else:
            print(f"  {label:8s}: analysed {len(results)} video(s)")

    print()

    for video in videos:
        print("-" * 70)
        print(video.name)
        print(f"  {'field':24s} {'GEMINI':>18s} {'CHATGPT':>18s}")

        g = (outcomes.get("gemini") or {}).get(video.name, {})
        o = (outcomes.get("openai") or {}).get(video.name, {})

        for field in FIELDS:
            gv = g.get(field, "-")
            ov = o.get(field, "-")
            print(f"  {field:24s} {str(gv):>18s} {str(ov):>18s}")

    print("-" * 70)
    print()
    print("Reading the result:")
    print(" - If the two columns broadly AGREE, swapping to paid ChatGPT")
    print("   changes little -> free Gemini (or free optical flow) is enough.")
    print(" - If they DISAGREE a lot, the deeper test is which reading yields")
    print("   a forecast closer to ACTUAL generation (run the backtest per")
    print("   provider on a day that has both video and meter data).")


if __name__ == "__main__":
    main()
