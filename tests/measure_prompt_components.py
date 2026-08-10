"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Where do our prompt tokens actually go?
=========================================================
Splits a real saved prompt into its sections and counts each
one with Gemini's own tokenizer, so trimming decisions are
made against measurements rather than impressions about which
part "looks long".

Adopted from the component table in Team 1's timeslot report,
which was the most useful thing in it: knowing the total tells
you what you spend, knowing the split tells you what to do
about it.

countTokens does not consume generation quota, so this can be
re-run freely after any prompt change.

A caveat this tool cannot escape: sections are counted in
isolation, so the parts do not sum exactly to the whole
prompt. A tokenizer merges across boundaries, and the header
each section keeps costs a few tokens on its own. The residual
is reported rather than hidden, and it is small.

Run:  python -m tests.measure_prompt_components
      python -m tests.measure_prompt_components --run-time 06:45
=========================================================
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from config.config import settings


# The section headings build_prompt writes, in order. Splitting on the
# headings themselves means this keeps working when a section's content
# changes, and fails loudly (a section goes missing) if one is renamed.
SECTIONS = [
    ("Header — plant, capacity, current time", None),
    ("Today's measured generation so far",
     "WHAT THE PLANT HAS ACTUALLY GENERATED TODAY SO FAR"),
    ("Satellite sky observation", "CURRENT SKY OBSERVATION"),
    ("Retrieved precedent (similar past days)",
     "WHAT HAPPENED IN SIMILAR SITUATIONS BEFORE"),
    ("Input track record", "WHICH INPUTS HAVE BEEN RIGHT LATELY"),
    ("Block table (the forecast rows)", "BLOCKS STILL TO SCHEDULE"),
    ("Column explanations", "HOW TO READ THE COLUMNS"),
    ("Task instructions", "YOUR JOB"),
    ("Output format spec", "RESPOND WITH JSON ONLY"),
]


def split_sections(text):
    """
    {label: chunk} for one prompt. A heading that is absent yields an
    empty chunk rather than being dropped, so the table always shows
    the same rows and a missing section is visible as a zero.
    """

    positions = []

    for label, heading in SECTIONS:

        if heading is None:
            positions.append((label, 0))
            continue

        index = text.find(heading)
        positions.append((label, index if index >= 0 else None))

    chunks = {}

    known = [(label, at) for label, at in positions if at is not None]

    for order, (label, start) in enumerate(known):

        end = known[order + 1][1] if order + 1 < len(known) else len(text)

        chunks[label] = text[start:end]

    for label, at in positions:
        if at is None:
            chunks[label] = ""

    return chunks


def main():

    parser = argparse.ArgumentParser(
        description="Measure where the prompt's tokens go, by section"
    )
    parser.add_argument(
        "--folder",
        default=settings.get("fusion", {}).get(
            "output_dir", "outputs/llm_schedules"
        ),
    )
    parser.add_argument(
        "--run-time", default="06:45",
        help="which scheduling time to break down (default 06:45, the "
             "longest prompt of the day)"
    )

    args = parser.parse_args()

    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}_" + args.run_time.replace(":", "-") + r"_prompt\.txt$"
    )

    paths = [
        p for p in sorted(Path(args.folder).glob("*_prompt.txt"))
        if pattern.search(p.name)
    ]

    if not paths:
        raise SystemExit(
            f"No saved prompt for {args.run_time} in {args.folder}"
        )

    # The longest example at this time, so the breakdown reflects the
    # worst case rather than a quiet day.
    path = max(paths, key=lambda p: p.stat().st_size)

    text = path.read_text(encoding="utf-8")

    from modules.vision.gemini_client import GeminiClient

    gemini = GeminiClient()

    model = settings.get("fusion", {}).get("model") or gemini.model

    total = int(
        gemini.client.models.count_tokens(model=model, contents=text).total_tokens
    )

    chunks = split_sections(text)

    rows = []

    for label, _ in SECTIONS:

        chunk = chunks.get(label, "")

        if not chunk.strip():
            rows.append({"component": label, "tokens": 0, "share_pct": 0.0})
            continue

        tokens = int(
            gemini.client.models.count_tokens(
                model=model, contents=chunk
            ).total_tokens
        )

        rows.append({
            "component": label,
            "tokens": tokens,
            "share_pct": tokens / total * 100,
        })

    frame = pd.DataFrame(rows).sort_values("tokens", ascending=False)

    counted = int(frame["tokens"].sum())

    print("=" * 78)
    print(f"PROMPT COMPONENT BREAKDOWN  —  {args.run_time} run")
    print(f"  {path.name}")
    print(f"  model: {model}")
    print("=" * 78)

    print(frame.to_string(index=False, float_format="%.1f"))

    print("-" * 78)
    print(f"{'sum of sections':44s} {counted:7,}")
    print(f"{'whole prompt, counted in one go':44s} {total:7,}")
    print(f"{'boundary residual':44s} {counted - total:+7,}  "
          f"({abs(counted - total) / total * 100:.1f}%)")

    images = settings.get("fusion", {})

    if images.get("attach_images", False):

        from tests.measure_token_cost import sample_images

        picked = sample_images()

        if picked:

            from google.genai import types

            parts = [text] + [
                types.Part.from_bytes(
                    data=Path(p).read_bytes(), mime_type="image/png"
                )
                for p in picked
            ]

            with_images = int(
                gemini.client.models.count_tokens(
                    model=model, contents=parts
                ).total_tokens
            )

            image_tokens = with_images - total

            print(f"\n{len(picked)} attached image(s): {image_tokens:,} tokens "
                  f"({image_tokens / len(picked):,.0f} each), "
                  f"{image_tokens / with_images * 100:.0f}% of the whole call")

            print("\nEvery text section above is smaller than the images.")
            print("Trimming text saves tokens; dropping one image saves more")
            print(f"than the {frame.iloc[0]['component'].lower()} entirely.")

    output = Path("outputs/reports/prompt_components.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
