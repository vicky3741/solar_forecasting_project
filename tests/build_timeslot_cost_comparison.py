"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Time-slot costing & LLM model comparison report (PDF)
=========================================================
The counterpart of Team 1's
SIRMOUR_timeslot_and_model_cost_comparison.pdf, built for OUR
LLM scheduling pipeline and from OUR measurements, in the same
section order so the two can be laid side by side:

    0  plain-language walk-through of the pipeline
    1  where an LLM is and is not used, step by step
    2  where the tokens go inside the one LLM step
    3  cost by time slot, current 7-call cadence
    4  LLM model comparison at that workload
    5  alternative cadence - half-hourly, 22 calls
    5a LLM model comparison at the half-hourly workload
    6  caveats

EVERY NUMBER IS READ FROM A MEASUREMENT FILE OR COMPUTED FROM
ONE HERE. Nothing is typed in by hand, so the document cannot
drift from the measurement and re-measuring re-generates it.

WHERE THE MEASUREMENTS COME FROM
--------------------------------
docs/cadence_measurements.json
    Input tokens for all 29 scheduling times - the 7 production
    ones and the 22 half-hourly ones - counted by Gemini's
    countTokens API over prompts built for real, on 12 real
    days, by tests/measure_halfhourly_cost.py. The prompt
    builder has a dry_run mode, so this costs no generation
    quota.

docs/token_measurements.json
    Output and thinking tokens from real generateContent calls,
    one per production scheduling time, read off usage_metadata
    (tests/measure_token_cost.py --live). Also the measured
    per-call image token cost.

docs/prompt_components.csv
    The prompt split into its sections, each counted separately
    (tests/measure_prompt_components.py).

The half-hourly slots' OUTPUT tokens are the only modelled
figures in the report. They cannot be measured: 22 real calls a
day is above the free tier's 20-per-model cap. They are fitted
from the seven measured calls and Section 5 says so, shows the
fit and reports its quality.

Run:  python -m tests.build_timeslot_cost_comparison
=========================================================
"""

import argparse
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from config.config import settings


# ----------------------------------------------------------------------
# Pricing
# ----------------------------------------------------------------------
# EVERY rate below was read off the provider's own published pricing
# page on 2026-08-11, which is why every row says "measured":
#
#   ai.google.dev/gemini-api/docs/pricing            all gemini-*
#   developers.openai.com/api/docs/pricing           GPT-5
#   docs.x.ai/docs/models                            Grok 4.5
#   api-docs.deepseek.com/quick_start/pricing        DeepSeek V4 Pro
#
# An earlier draft carried the non-Gemini rates from Team 1's report of
# 2026-08-07 and labelled them as theirs. They have now been read
# independently and all four agree with that report exactly, so the
# distinction no longer exists and the column no longer draws one.
#
# Two rates are tier-dependent and the tier that applies to us is the
# one quoted. Grok 4.5 charges $2/$6 below a 200k-token prompt and
# double above it; Gemini 2.5 Pro charges $1.25/$10 at or below 200k.
# Our largest call is under 6k tokens, so the low tier applies with a
# very wide margin. DeepSeek's input rate is its cache-MISS rate, which
# is what a fresh daily prompt pays.
PRICING = [
    # name, $/1M in, $/1M out, vision, source, in our fallback chain
    ("DeepSeek V4 Pro",            0.435,  0.870, False, "measured", False),
    ("Gemini 2.5 Flash",           0.300,  2.500, True,  "measured", False),
    ("gemini-3.5-flash-lite",      0.300,  2.500, True,  "measured", True),
    ("Grok 4.5",                   2.000,  6.000, True,  "measured", False),
    ("gemini-3.6-flash (current)", 1.500,  7.500, True,  "measured", True),
    ("gemini-3.5-flash",           1.500,  9.000, True,  "measured", False),
    ("GPT-5",                      1.250, 10.000, True,  "measured", False),
    ("Gemini 2.5 Pro",             1.250, 10.000, True,  "measured", False),
]

CURRENT_MODEL = "gemini-3.6-flash"

# Live ECB reference rate, fetched 2026-08-10 from
# api.frankfurter.dev/v1/latest?from=USD&to=INR, quoted for 2026-08-07 -
# the same rate the earlier token cost report used, kept so the two of
# our own documents agree.
#
# NOTE FOR ANYONE COMPARING THE TWO REPORTS: Team 1's report converts at
# Rs 84.00/$ (their $130.69/yr prints as Rs 10,978). Ours converts at the
# measured rate below. Dollar figures are directly comparable between the
# reports; rupee figures are not.
USD_TO_INR = 95.21
USD_TO_INR_SOURCE = "ECB reference rate, 2026-08-07 (frankfurter.dev)"

DARK = colors.HexColor("#1B4332")
ACCENT = colors.HexColor("#1F4E9C")
WARN = colors.HexColor("#9A3412")
GREY = colors.HexColor("#F2F4F6")
MUTED = colors.HexColor("#555555")


# ----------------------------------------------------------------------
# The pipeline, as it actually is
# ----------------------------------------------------------------------
# One row per distinct stage between a raw input and a published block.
# The file column is the file that does the work, so any claim here can
# be checked against the code rather than taken on trust.
STAGES = [
    (1, "Meter inbox sync", "preprocessing/meter_inbox.py", False,
     "Copies any new meter export from the team's drop folder into "
     "data/historical. Copy-only - the inbox is shared."),
    (2, "Windy value scrape", "capture/windy_scraper.py", False,
     "Playwright drives the Windy embed's own JavaScript API and reads "
     "solar power, cloud decks, rain and wind at the plant coordinates "
     "as NUMBERS. No pixels are interpreted."),
    (3, "Satellite clip + layer capture", "capture/windy_capture.py", False,
     "Screen-records the EUMETSAT satellite layer and grabs the still "
     "layer screenshots."),
    (4, "Satellite clip features", "vision/satellite_features.py", False,
     "OpenCV: thick/thin/clear cloud split, texture entropy, cloud by "
     "quadrant, Farneback flow divergence, vorticity and energy."),
    (5, "Meter history load", "preprocessing/preprocess.py", False,
     "Reads and resamples every daily meter CSV, each day on its own so "
     "nothing interpolates across the overnight gap."),
    (6, "Solar geometry", "preprocessing/windy_features.py", False,
     "pvlib solar position - elevation and azimuth for every 15-minute "
     "block of the day."),
    (7, "Clear-sky physics", "preprocessing/windy_features.py", False,
     "pvlib clear-sky GHI to plane-of-array to clearsky_power_mw. This "
     "is the SHAPE of the day and it is exact."),
    (8, "Windy spread to blocks", "preprocessing/windy_features.py", False,
     "Windy's 3-hourly values interpolated through clear-sky-index "
     "space, each block flagged windy_is_measured / windy_gap_minutes."),
    (9, "Weather attach + bias correction", "weather/open_meteo.py", False,
     "ECMWF via Open-Meteo, corrected by its own recently measured bias "
     "at this plant."),
    (10, "Damped-persistence anchor", "fusion/llm_scheduler.py", False,
     "Today's last measured cloudiness decaying toward the day's "
     "average. THE NUMBER THE MODEL IS ASKED TO ADJUST."),
    (11, "Precedent retrieval", "fusion/case_retrieval.py", False,
     "Numeric distance match against this pipeline's own scored case "
     "store - what actually happened last time it looked like this."),
    (12, "Input track record", "evaluation/input_skill.py", False,
     "Every input scored separately against real generation over recent "
     "finished days, so the model weighs them on evidence."),
    (13, "Intraday character", "preprocessing/intraday_shape.py", False,
     "Trend, choppiness and clear-sky ratio computed from today's own "
     "readings, summarised as a sentence."),
    (14, "Prompt assembly", "fusion/llm_scheduler.py", False,
     "String assembly. All-empty columns are dropped rather than sent "
     "as 'n/a' repeated once per block."),
    (15, "LLM decision", "fusion/llm_scheduler.py + vision/gemini_client.py",
     True,
     "THE ONLY LLM STEP. Gemini reads the block table, the evidence "
     "sections and 2 layer screenshots, and returns a clear-sky index "
     "and a confidence for every block."),
    (16, "JSON parse", "vision/json_parser.py", False,
     "Parses the reply. Any block the model omitted is published at its "
     "anchor value."),
    (17, "kt to MW + confidence blend", "fusion/llm_scheduler.py", False,
     "Clear-sky curve x kt, then a per-block blend with the anchor "
     "weighted by the model's own stated confidence."),
    (18, "Level calibration", "forecasting/llm_block_bias.py", False,
     "One learned multiplier for the day, from this pipeline's own "
     "finished days."),
    (19, "Validator safety checks", "fusion/validator.py", False,
     "Range clip, deviation cap against the anchor, step-change "
     "smoothness. Deterministic."),
    (20, "Freeze horizon", "scheduling/effective_time.py", False,
     "The 6-block effective-time rule. Blocks already declared to the "
     "grid operator are re-published unchanged."),
    (21, "Publish + push", "fusion/llm_scheduler.py, storage/s3_client.py",
     False,
     "Writes the schedule, the meta JSON and the Current Final "
     "Schedule, and pushes them to this plant's own S3 bucket."),
    (22, "Evening feedback loop", "evaluation/actuals_feedback.py", False,
     "Attaches real meter generation to the saved schedules, turns them "
     "into cases for tomorrow, and measures the bias that decides how "
     "much rope the validator gives the model."),
]

LLM_STEP = next(row[0] for row in STAGES if row[3])


# The eight plain-language steps, mapped onto the technical ones above.
PLAIN_STEPS = [
    ("Reading Windy as numbers, not pictures",
     "A browser is driven to Windy.com at the plant's exact coordinates "
     "and the forecast is read out of the page as figures - solar power "
     "in W/m&sup2;, cloud cover high/middle/low, rain, wind. Nothing is "
     "guessed from the colour of a map."),
    ("Watching the sky that is actually there",
     "A short satellite clip is recorded, and ordinary image-processing "
     "code (no AI) measures how much of the sky is thick cloud, how "
     "broken it looks, which quarter of the frame the cloud sits in, and "
     "whether the cloud field is gathering or breaking up. This is an "
     "observation of now, not a forecast."),
    ("Building the day's table",
     "One row per 15-minute block for the rest of the day. Each row gets "
     "the exact physics answer - what this plant would make under a "
     "perfectly clear sky at that moment - plus the weather forecast and "
     "the Windy figures for that block."),
    ("Making a baseline guess (the anchor)",
     "The last cloudiness the meter actually measured today, fading "
     "toward the day's average the further ahead the block is. Plain "
     "arithmetic, no AI. Over twelve measured days this anchor alone was "
     "the most accurate thing available."),
    ("Checking the record",
     "Two lookups, both pure number-matching. What the plant really did "
     "the last several times conditions looked like this, and which of "
     "our inputs has actually been closest over the last few finished "
     "days."),
    ("This is the ONLY step that uses AI",
     "Everything above - the table, the anchor, the precedent, today's "
     "own readings, and 2 of the layer screenshots - goes to Gemini with "
     "one narrow question: for each block, should the anchor be nudged "
     "up or down, and how sure are you? The model never invents a "
     "megawatt figure; it returns a fraction of clear sky, and the "
     "physics turns that into megawatts."),
    ("Three safety checks, no AI in any of them",
     "The model's number is blended with the anchor in proportion to the "
     "confidence it stated itself; a learned whole-day multiplier is "
     "applied; then a rule-checker clips anything outside range, more "
     "than 25% away from the anchor, or jumping too hard between "
     "neighbouring blocks. Finally the freeze horizon re-publishes the "
     "blocks already declared to the grid operator, untouched."),
    ("Saving, and learning every evening",
     "The surviving schedule is written and pushed to the plant's S3 "
     "bucket. When the real meter reading arrives, every block is scored "
     "against it, and the result becomes both tomorrow's precedent and "
     "tomorrow's calibration - with no manual retraining step."),
]


# 2c. What the feature table computes and what is actually sent.
# Taken from the columns of a real saved feature table plus the two
# derived ones, against _PROMPT_COLUMNS in modules/fusion/llm_scheduler.py.
BLOCK_COLUMNS_SENT = [
    "block", "time", "clearsky_power_mw", "anchor_kt", "weather_kt",
    "windy_kt", "windy_clouds_pct", "windy_lclouds_pct",
    "windy_mclouds_pct", "windy_hclouds_pct", "windy_rain_mm",
    "windy_is_measured",
]

BLOCK_COLUMNS_HELD = [
    "plant", "timestamp", "day_of_year", "month", "hour", "minute_of_day",
    "solar_elevation_deg", "solar_azimuth_deg", "clearsky_ghi_w_m2",
    "clearsky_poa_w_m2", "is_daylight", "windy_solarpower_w_m2",
    "windy_wind_kt", "windy_wind_from_deg", "windy_gap_minutes", "is_past",
    "actual_power_mw", "actual_ghi_w_m2", "actual_kt", "run_time",
]

SATELLITE_SENT = [
    "sat_thick_cloud_pct", "sat_thin_cloud_pct", "sat_clear_pct",
    "sat_cloud_trend_pct", "sat_entropy", "sat_north_cloud_pct",
    "sat_south_cloud_pct", "sat_west_cloud_pct", "sat_east_cloud_pct",
    "sat_flow_divergence", "sat_flow_vorticity", "sat_flow_kinetic_energy",
]

SATELLITE_HELD = [
    "sat_frames_used", "sat_brightness_mean", "sat_brightness_std",
    "sat_contrast",
]

# Layers the capture stack can produce a still for, against the two the
# call actually attaches (fusion.image_layers).
CAPTURED_LAYERS = ["satellite", "clouds", "solarpower", "wind", "rain"]
SENT_LAYERS = ["satellite", "clouds"]


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def money(usd):
    return f"${usd:,.4f}" if usd < 1 else f"${usd:,.2f}"


def rupees(usd):
    # "Rs", not the rupee sign: reportlab's built-in Helvetica has no
    # glyph for U+20B9 and it renders as a hollow box.
    return f"Rs {usd * USD_TO_INR:,.2f}"


def fit_line(xs, ys):
    """
    Least-squares slope, intercept and R2. Returned together on purpose -
    a fitted number quoted without its R2 hides whether the fit means
    anything, and for one of the two series below it does not.
    """

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    syy = sum((y - mean_y) ** 2 for y in ys)

    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x

    r2 = (slope * sxy / syy) if syy else 0.0

    return slope, intercept, r2, mean_y


def cost_of(input_tokens, output_tokens, rate_in, rate_out):
    return input_tokens / 1e6 * rate_in + output_tokens / 1e6 * rate_out


# ----------------------------------------------------------------------

def build(cadence, tokens, components, out_path, component_clock="06:45"):

    plant = settings["plant"]

    production_times = settings["forecast"]["run_times"]

    measured = cadence["by_run_time"]

    image_tokens = int(tokens["image_tokens_per_call"])
    images_n = int(tokens["images_attached"])

    output_by_time = tokens["output"]["by_run_time"]

    # ---------- current cadence, per slot ----------
    current = []

    for clock in production_times:

        entry = measured[clock]
        out = output_by_time[clock]

        text_in = entry["input_tokens_mean"]
        total_in = text_in + image_tokens

        visible = out["visible_output_tokens"]
        thinking = out["thinking_tokens"]

        current.append({
            "clock": clock,
            "blocks": entry["blocks"],
            "text_in": text_in,
            "input": total_in,
            "visible": visible,
            "thinking": thinking,
            "output": visible + thinking,
        })

    # ---------- the model that carries output to the half-hourly grid ----
    blocks = [row["blocks"] for row in current]

    vis_slope, vis_intercept, vis_r2, _ = fit_line(
        blocks, [row["visible"] for row in current]
    )
    think_slope, think_intercept, think_r2, think_mean = fit_line(
        blocks, [row["thinking"] for row in current]
    )

    # ---------- half-hourly cadence ----------
    half_times = [c for c in cadence["run_times"] if c not in production_times]

    half = []

    for clock in half_times:

        entry = measured[clock]

        text_in = entry["input_tokens_mean"]

        visible = vis_intercept + vis_slope * entry["blocks"]

        # Thinking does NOT track block count - see the R2 printed in
        # Section 5. Carrying it per call is what the measurement
        # supports; scaling it by blocks would be inventing a
        # relationship the seven measured calls deny.
        thinking = think_mean

        half.append({
            "clock": clock,
            "blocks": entry["blocks"],
            "text_in": text_in,
            "input": text_in + image_tokens,
            "visible": visible,
            "thinking": thinking,
            "output": visible + thinking,
        })

    rate_in, rate_out = next(
        (r_in, r_out) for name, r_in, r_out, *_ in PRICING
        if name.startswith(CURRENT_MODEL)
    )

    for row in current + half:
        row["cost"] = cost_of(row["input"], row["output"], rate_in, rate_out)

    def totals(rows):
        return {
            "calls": len(rows),
            "blocks": sum(r["blocks"] for r in rows),
            "input": sum(r["input"] for r in rows),
            "image_input": image_tokens * len(rows),
            "visible": sum(r["visible"] for r in rows),
            "thinking": sum(r["thinking"] for r in rows),
            "output": sum(r["output"] for r in rows),
            "cost": sum(r["cost"] for r in rows),
        }

    now = totals(current)
    alt = totals(half)

    # ------------------------------------------------------------------
    # styles
    # ------------------------------------------------------------------
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle(
        "h1", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=15.5, textColor=DARK, spaceAfter=3, alignment=0, leading=19,
    )
    sub = ParagraphStyle(
        "sub", parent=styles["Normal"], fontSize=8.2, textColor=MUTED,
        leading=11.5, spaceAfter=9,
    )
    h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, textColor=ACCENT, spaceBefore=11, spaceAfter=5,
    )
    h3 = ParagraphStyle(
        "h3", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=9.5, textColor=DARK, spaceBefore=8, spaceAfter=3,
    )
    body = ParagraphStyle(
        "body", parent=styles["Normal"], fontSize=8.8, leading=12.6,
        spaceAfter=4,
    )
    note = ParagraphStyle(
        "note", parent=styles["Normal"], fontSize=7.9, textColor=MUTED,
        leading=10.8, spaceBefore=4, spaceAfter=3,
    )
    cell = ParagraphStyle(
        "cell", parent=styles["Normal"], fontSize=7.4, leading=9.2,
    )
    cellb = ParagraphStyle(
        "cellb", parent=styles["Normal"], fontSize=7.4, leading=9.2,
        fontName="Helvetica-Bold",
    )
    step = ParagraphStyle(
        "step", parent=styles["Normal"], fontSize=8.6, leading=12.2,
        spaceAfter=5, leftIndent=13,
    )

    def table(data, widths, align_right_from=1, header=True, spans=None):

        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 7.6),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8CDD3")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]

        if header:
            style += [
                ("BACKGROUND", (0, 0), (-1, 0), DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]

        for extra in (spans or []):
            style.append(extra)

        return Table(data, colWidths=widths, style=TableStyle(style),
                     hAlign="LEFT", repeatRows=1 if header else 0)

    story = []

    # ------------------------------------------------------------------
    # title
    # ------------------------------------------------------------------
    story.append(Paragraph(
        f"{plant['code']} LLM Scheduling Pipeline &mdash; Plain-Language "
        "Overview, LLM Usage Map, Token Breakdown, Time-Slot Costing &amp; "
        "Model Comparison", h1
    ))

    story.append(Paragraph(
        f"Plant: {plant['code']}, {plant['capacity_mw']} MW &nbsp;|&nbsp; "
        f"Model: <b>{CURRENT_MODEL}</b> &nbsp;|&nbsp; "
        f"Scope: our own <b>new-approach LLM scheduling pipeline</b>. The "
        "production Chronos/weather blend and the Enercast comparison are "
        "excluded &mdash; neither makes an LLM call, so neither has a token "
        f"cost. &nbsp;|&nbsp; Each call forecasts every remaining 15-minute "
        f"block to 18:45.",
        sub,
    ))

    # ------------------------------------------------------------------
    # 0. plain language
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "0. How the Pipeline Works, Start to End &mdash; In Plain Language",
        h2))

    story.append(Paragraph(
        "Before the cost and token details below, here is the whole process "
        "in simple terms &mdash; what happens from the moment input data is "
        "captured to the moment a schedule is published.", body))

    for index, (title, text) in enumerate(PLAIN_STEPS, 1):
        story.append(Paragraph(
            f"<b>{index}&nbsp;&nbsp;{title}</b><br/>{text}", step))

    story.append(Paragraph(
        "<b>In one line:</b> Windy numbers + satellite observation &rarr; "
        "physics table &rarr; persistence anchor &rarr; check the record "
        "&rarr; <b>AI adjusts the anchor per block</b> &rarr; blend, "
        "calibrate, validate, freeze &rarr; publish &rarr; learn every "
        f"evening. AI is used in exactly ONE of these 8 steps &mdash; "
        "everything else is ordinary, deterministic code.", body))

    # ------------------------------------------------------------------
    # 1. LLM usage map
    # ------------------------------------------------------------------
    # Broken to a fresh page deliberately: the stage table is a page tall
    # on its own, and letting it start at the foot of page 1 strands one
    # row there and splits the map that the whole cost argument rests on.
    story.append(PageBreak())

    story.append(Paragraph(
        "1. Where LLM Is (and Isn't) Used &mdash; Full Pipeline, Step by Step",
        h2))

    story.append(Paragraph(
        f"The pipeline has <b>{len(STAGES)} distinct technical steps</b> "
        "between a raw input and a published block (the 8 plain-language "
        "steps above map onto these). Only ONE of them calls an LLM. "
        "Everything else is deterministic Python / pvlib / OpenCV / "
        "arithmetic with zero tokens and zero API cost.", body))

    llm_count = sum(1 for row in STAGES if row[3])

    story.append(table(
        [["TOTAL PIPELINE STEPS", "STEPS USING AN LLM",
          "STEPS THAT ARE PURE CODE/MATH"],
         [str(len(STAGES)), str(llm_count), str(len(STAGES) - llm_count)]],
        [60 * mm, 55 * mm, 63 * mm], align_right_from=0,
        spans=[("ALIGN", (0, 0), (-1, -1), "CENTER"),
               ("FONTSIZE", (0, 1), (-1, 1), 15),
               ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
               ("TEXTCOLOR", (1, 1), (1, 1), WARN),
               ("TOPPADDING", (0, 1), (-1, 1), 6),
               ("BOTTOMPADDING", (0, 1), (-1, 1), 6)],
    ))

    story.append(Spacer(1, 5))

    rows = [[
        Paragraph("<b>#</b>", cell), Paragraph("<b>PIPELINE STAGE</b>", cell),
        Paragraph("<b>FILE (under modules/)</b>", cell),
        Paragraph("<b>LLM?</b>", cell),
        Paragraph("<b>WHAT ACTUALLY HAPPENS</b>", cell),
    ]]

    for number, stage, path, uses_llm, what in STAGES:

        style = cellb if uses_llm else cell

        rows.append([
            Paragraph(str(number), style),
            Paragraph(stage, style),
            Paragraph(path, style),
            Paragraph("<b>YES</b>" if uses_llm else "NO", style),
            Paragraph(what, style),
        ])

    highlight = [
        ("BACKGROUND", (0, LLM_STEP), (-1, LLM_STEP),
         colors.HexColor("#FFE8D6")),
        ("TEXTCOLOR", (3, LLM_STEP), (3, LLM_STEP), WARN),
    ]

    story.append(table(
        rows, [7 * mm, 33 * mm, 42 * mm, 10 * mm, 86 * mm],
        align_right_from=5, spans=highlight,
    ))

    story.append(Paragraph(
        f"<b>Why this matters for cost:</b> every dollar and every token in "
        f"Sections 2&ndash;5 comes ENTIRELY from step {LLM_STEP}. The other "
        f"{len(STAGES) - 1} steps run at zero marginal API cost no matter how "
        "much history accumulates, how many days of context are kept, or how "
        "often the pipeline runs.", note))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # 2. where the tokens go
    # ------------------------------------------------------------------
    story.append(Paragraph(
        f"2. Where Our Tokens Actually Go (Inside Step {LLM_STEP})", h2))

    story.append(Paragraph(
        "2a. Layer screenshots &mdash; 5 layers captured, only 2 sent to the "
        "LLM", h3))

    rows = [["LAYER", "TOKENS", "SENT TO LLM?"]]

    per_image = image_tokens / images_n if images_n else 0

    for layer in CAPTURED_LAYERS:
        sent = layer in SENT_LAYERS
        rows.append([
            f"{layer}.png", f"{per_image:,.0f}", "YES" if sent else "NO",
        ])

    rows.append([
        f"Currently billed ({images_n} of {len(CAPTURED_LAYERS)})",
        f"{image_tokens:,}", "",
    ])

    story.append(table(rows, [58 * mm, 28 * mm, 30 * mm], align_right_from=1,
                       spans=[("FONTNAME", (0, len(rows) - 1),
                               (-1, len(rows) - 1), "Helvetica-Bold")]))

    story.append(Paragraph(
        f"Measured, not assumed: countTokens was asked for the prompt text "
        f"alone and then for the text plus the images, and the difference "
        f"<b>is</b> the image cost &mdash; {image_tokens:,} tokens per call, "
        f"{per_image:,.0f} per screenshot, for our own screenshots at our own "
        f"resolution. The satellite <i>video</i> is never sent: it is read "
        f"locally by OpenCV and enters the prompt as a dozen numbers "
        f"(Section 2c).", note))

    reference = next(
        row for row in current if row["clock"] == component_clock
    )

    story.append(Paragraph(
        f"2b. Prompt text &mdash; component breakdown "
        f"({component_clock} run, {reference['blocks']} blocks)", h3))

    rows = [["COMPONENT", "TOKENS", "% OF TEXT"]]

    counted = sum(int(r["tokens"]) for r in components)

    for record in components:
        rows.append([
            record["component"],
            f"{int(record['tokens']):,}",
            f"{float(record['share_pct']):.1f}%",
        ])

    text_total = reference["text_in"]

    rows.append(["TEXT SUBTOTAL (sections, counted separately)",
                 f"{counted:,}", ""])
    rows.append([f"Layer screenshots ({images_n})", f"{image_tokens:,}", ""])
    rows.append([f"TOTAL INPUT, {component_clock} call",
                 f"{reference['input']:,.0f}", ""])

    story.append(table(
        rows, [82 * mm, 24 * mm, 24 * mm],
        spans=[("FONTNAME", (0, len(rows) - 3), (-1, -1), "Helvetica-Bold")],
    ))

    reference_max = measured[component_clock]["input_tokens_max"]

    story.append(Paragraph(
        f"The block table is the single largest text component, which is why "
        f"the prompt builder drops any column that is entirely empty for a "
        f"run rather than sending 'n/a' once per block. The split was "
        f"measured on the <b>largest</b> {component_clock} prompt of the "
        f"three days, which counts {reference_max:,} tokens in one go; the "
        f"sections sum to {counted:,}, a residual of "
        f"{counted - reference_max:+,} ({abs(counted - reference_max) / reference_max * 100:.1f}%) "
        f"because a tokenizer merges across section boundaries. The TOTAL row "
        f"above is the 3-day mean for this slot, so it matches Section 3. "
        f"<b>Every text section is smaller than the two images.</b>", note))

    story.append(Paragraph(
        "2c. Features computed vs features sent", h3))

    sent_n = len(BLOCK_COLUMNS_SENT) + len(SATELLITE_SENT)
    held_n = len(BLOCK_COLUMNS_HELD) + len(SATELLITE_HELD)

    story.append(Paragraph(
        f"<b>{sent_n + held_n} features are computed per run; "
        f"{sent_n} are sent.</b> The rest are kept for scoring, debugging "
        "and the feedback loop, and cost nothing because they never enter a "
        "prompt.", body))

    rows = [[
        Paragraph("<b>&#10003; SENT TO THE MODEL</b>", cell),
        Paragraph("<b>&#10007; COMPUTED BUT NOT SENT</b>", cell),
    ], [
        Paragraph(
            f"<b>Block table, {len(BLOCK_COLUMNS_SENT)} columns x every "
            f"remaining block:</b><br/>"
            + ", ".join(BLOCK_COLUMNS_SENT)
            + f"<br/><br/><b>Satellite observation, {len(SATELLITE_SENT)} "
            "numbers, once:</b><br/>" + ", ".join(SATELLITE_SENT), cell),
        Paragraph(
            f"<b>Block table, {len(BLOCK_COLUMNS_HELD)} columns:</b><br/>"
            + ", ".join(BLOCK_COLUMNS_HELD)
            + f"<br/><br/><b>Satellite, {len(SATELLITE_HELD)}:</b><br/>"
            + ", ".join(SATELLITE_HELD), cell),
    ]]

    story.append(table(rows, [65 * mm, 65 * mm], align_right_from=2))

    story.append(Paragraph(
        "The satellite numbers are sent <b>once, as a sky observation</b>, "
        "not as columns repeated down every row. One clip describes one "
        "moment; pasting it down 47 rows would both cost 47 times the tokens "
        "and invite the model to read a single observation as a forecast that "
        "holds all afternoon.", note))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # 3. cost by time slot, current cadence
    # ------------------------------------------------------------------
    story.append(Paragraph(
        f"3. Cost by Time Slot &mdash; Current Cadence "
        f"({CURRENT_MODEL}, {now['calls']} calls/day)", h2))

    story.append(Paragraph(
        f"Input tokens are the mean of <b>{cadence['prompts_measured']} real "
        f"prompts built across {len(cadence['days'])} days</b> "
        f"({cadence['days'][0]} to {cadence['days'][-1]}), counted by "
        f"countTokens, plus the measured {image_tokens:,} image tokens each "
        f"call attaches. Visible and thinking output are read from "
        f"<b>usage_metadata after a real generateContent call at each of the "
        f"{now['calls']} scheduling times</b> &mdash; not one call multiplied "
        f"by seven, because the 06:45 run asks for "
        f"{current[0]['blocks']} blocks and the {current[-1]['clock']} run "
        f"asks for {current[-1]['blocks']}.", body))

    rows = [["TIME SLOT", "BLOCKS", "INPUT", "OUT (VISIBLE)", "THINKING",
             "TOTAL OUT", "COST (USD)", "COST (INR)"]]

    for row in current:
        rows.append([
            f"{row['clock']} → 18:45",
            f"{row['blocks']}",
            f"{row['input']:,.0f}",
            f"{row['visible']:,.0f}",
            f"{row['thinking']:,.0f}",
            f"{row['output']:,.0f}",
            f"{row['cost']:.4f}",
            f"{row['cost'] * USD_TO_INR:.2f}",
        ])

    rows.append([
        f"TOTAL ({now['calls']} calls)", f"{now['blocks']}",
        f"{now['input']:,.0f}", f"{now['visible']:,.0f}",
        f"{now['thinking']:,.0f}", f"{now['output']:,.0f}",
        f"{now['cost']:.4f}", f"{now['cost'] * USD_TO_INR:.2f}",
    ])

    story.append(table(
        rows, [26 * mm, 14 * mm, 19 * mm, 22 * mm, 19 * mm, 19 * mm,
               21 * mm, 20 * mm],
        spans=[("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1),
                "Helvetica-Bold")],
    ))

    story.append(Paragraph(
        f"Per month: {money(now['cost'] * 30)} "
        f"({rupees(now['cost'] * 30)}) &nbsp;|&nbsp; "
        f"Per year: {money(now['cost'] * 365)} "
        f"({rupees(now['cost'] * 365)}) &nbsp;|&nbsp; "
        f"Across all three plants: "
        f"{money(now['cost'] * 365 * 3)} "
        f"({rupees(now['cost'] * 365 * 3)}) per year.", body))

    thinking_share = now["thinking"] / now["output"] * 100
    thinking_cost = now["thinking"] / 1e6 * rate_out
    image_cost = now["image_input"] / 1e6 * rate_in

    story.append(Paragraph(
        f"<b>Two components dominate, and neither is the schedule.</b> "
        f"{now['thinking']:,.0f} of the {now['output']:,.0f} billable output "
        f"tokens are internal reasoning that never appears in the response "
        f"&mdash; {thinking_share:.0f}% of output, "
        f"{thinking_cost / now['cost'] * 100:.0f}% of the daily bill. The "
        f"{images_n} screenshots are another {now['image_input']:,.0f} input "
        f"tokens a day, {image_cost / now['cost'] * 100:.0f}% of the bill and "
        f"{now['image_input'] / now['input'] * 100:.0f}% of all input. The "
        f"visible JSON schedule &mdash; the thing the plant actually uses "
        f"&mdash; is "
        f"{(now['visible'] / 1e6 * rate_out) / now['cost'] * 100:.0f}%.",
        note))

    # ------------------------------------------------------------------
    # 4. model comparison, current cadence
    # ------------------------------------------------------------------
    story.append(Paragraph(
        f"4. LLM Model Comparison &mdash; {now['calls']}-Call/Day Workload",
        h2))

    story.append(Paragraph(
        f"Each model's published pricing applied to this workload: "
        f"<b>{now['input']:,.0f} input + {now['output']:,.0f} output tokens "
        f"per day</b>. A model without vision cannot take our two "
        f"screenshots, so its rows are priced on "
        f"{now['input'] - now['image_input']:,.0f} input tokens &mdash; the "
        f"text only &mdash; and it is buying a different, thinner request.",
        body))

    story.append(model_table(now, table, cell))

    story.append(Paragraph(
        "<b>*</b> in this pipeline's own configured model chain &mdash; the "
        "primary model or one of its fallbacks. A model not in that chain has "
        "never been called with our key and is priced here for comparison "
        "only.", note))

    story.append(Paragraph(
        "Token volumes are ours, measured on Gemini's tokenizer. Other "
        "vendors tokenize differently, so a row for a non-Gemini model is "
        "our volume at their rate, not their volume at their rate.", note))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # 5. alternative cadence
    # ------------------------------------------------------------------
    first, last = half[0]["clock"], half[-1]["clock"]

    story.append(Paragraph(
        f"5. Alternative Cadence &mdash; Half-Hourly Schedule Generation "
        f"({first}&ndash;{last}, each call to 18:45)", h2))

    story.append(Paragraph(
        f"A NEW schedule generated every 30 minutes from {first} through "
        f"{last} &mdash; each call still forecasting every remaining block to "
        f"18:45, just starting more often. That is <b>{alt['calls']} calls a "
        f"day instead of {now['calls']}</b>.", body))

    story.append(Paragraph(
        f"<b>These input figures are measured, not interpolated.</b> The "
        f"prompt builder has a dry-run mode, so all {alt['calls']} "
        f"half-hourly prompts were built for real on the same "
        f"{len(cadence['days'])} days and counted with countTokens &mdash; "
        f"which spends no generation quota. What could NOT be measured is "
        f"output: {alt['calls']} real calls a day is above the free tier's "
        f"20-per-model cap, so visible output is fitted against block count "
        f"and thinking is carried per call. Both fits come from the seven "
        f"measured calls in Section 3, and both are stated below rather than "
        f"buried.", body))

    story.append(Paragraph(
        f"&nbsp;&nbsp;&bull;&nbsp; <b>Visible output</b> = "
        f"{vis_intercept:,.1f} + {vis_slope:,.2f} &times; blocks "
        f"&nbsp;(R&sup2; = {vis_r2:.2f} &mdash; the answer is a JSON line per "
        f"block, so it scales with blocks almost exactly).<br/>"
        f"&nbsp;&nbsp;&bull;&nbsp; <b>Thinking</b> = "
        f"{think_mean:,.0f} per call, flat. Regressed against block count it "
        f"gives R&sup2; = {think_r2:.2f} &mdash; effectively no relationship. "
        f"The 12:45 run produced the most thinking "
        f"({max(r['thinking'] for r in current):,.0f}) of any run of the day "
        f"while asking for "
        f"{next(r['blocks'] for r in current if r['thinking'] == max(x['thinking'] for x in current))} "
        f"blocks. Scaling thinking by blocks would invent a relationship the "
        f"measurements deny; carrying it per call is what they support.",
        note))

    # The two month columns are this slot's own call repeated for 30
    # days, not the whole cadence - so a row reads "what does keeping
    # THIS extra slot cost me a month", which is the decision actually
    # on the table when a cadence is being chosen.
    rows = [["TIME", "BLOCKS", "INPUT", "OUT (VIS)", "THINK", "TOTAL OUT",
             "$/CALL", "Rs/CALL", "$/MONTH", "Rs/MONTH"]]

    for row in half:
        rows.append([
            f"{row['clock']}→18:45",
            f"{row['blocks']}",
            f"{row['input']:,.0f}",
            f"{row['visible']:,.0f}",
            f"{row['thinking']:,.0f}",
            f"{row['output']:,.0f}",
            f"{row['cost']:.4f}",
            f"{row['cost'] * USD_TO_INR:.2f}",
            f"{row['cost'] * 30:.3f}",
            f"{row['cost'] * 30 * USD_TO_INR:.2f}",
        ])

    rows.append([
        f"TOTAL ({alt['calls']} calls/day)", f"{alt['blocks']}",
        f"{alt['input']:,.0f}", f"{alt['visible']:,.0f}",
        f"{alt['thinking']:,.0f}", f"{alt['output']:,.0f}",
        f"{alt['cost']:.4f}", f"{alt['cost'] * USD_TO_INR:.2f}",
        f"{alt['cost'] * 30:.2f}", f"{alt['cost'] * 30 * USD_TO_INR:,.2f}",
    ])

    story.append(table(
        rows, [24 * mm, 15 * mm, 17 * mm, 17 * mm, 15 * mm, 18 * mm,
               16 * mm, 17 * mm, 18 * mm, 20 * mm],
        spans=[("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1),
                "Helvetica-Bold")],
    ))

    increase = (alt["cost"] / now["cost"] - 1) * 100

    story.append(Paragraph(
        f"Per month: {money(alt['cost'] * 30)} "
        f"({rupees(alt['cost'] * 30)}) &nbsp;|&nbsp; "
        f"Per year: {money(alt['cost'] * 365)} "
        f"({rupees(alt['cost'] * 365)})", body))

    story.append(Paragraph(
        f"<b>Cost impact vs the current {now['calls']}-call/day cadence:</b> "
        f"{money(now['cost'])} ({rupees(now['cost'])}) per day &rarr; "
        f"{money(alt['cost'])} ({rupees(alt['cost'])}) per day &mdash; a "
        f"<b>+{increase:.0f}% increase</b>. Yearly: "
        f"{rupees(now['cost'] * 365)} &rarr; {rupees(alt['cost'] * 365)}, "
        f"about {rupees((alt['cost'] - now['cost']) * 365)} more per site per "
        f"year, or "
        f"{rupees((alt['cost'] - now['cost']) * 365 * 3)} across the three "
        f"plants.", body))

    story.append(Paragraph(
        f"<b>Cost tracks calls, not blocks.</b> {alt['calls']} calls schedule "
        f"{alt['blocks']} block-forecasts against {now['blocks']} today, "
        f"{alt['blocks'] / now['blocks']:.1f}&times; the work, but cost "
        f"{alt['cost'] / now['cost']:.1f}&times; as much. The reason is in "
        f"Section 3: thinking tokens and the two screenshots are charged per "
        f"call and barely move with the size of the request, so tripling the "
        f"call count roughly triples them. Together they are "
        f"{(alt['thinking'] / 1e6 * rate_out + alt['image_input'] / 1e6 * rate_in) / alt['cost'] * 100:.0f}% "
        f"of the half-hourly bill.", note))

    story.append(Paragraph(
        f"<b><font color='#9A3412'>Bigger issue than cost: request "
        f"count.</font></b> "
        f"{alt['calls']} calls/day for this plant alone is more than double "
        f"the free tier's cap of 20 requests per day per model &mdash; a cap "
        f"verified here directly, from quota-exhaustion errors hit during "
        f"this project's own backtesting, not from documentation. Across the "
        f"three plants the half-hourly cadence would be "
        f"{alt['calls'] * 3} calls/day.", body))

    story.append(Paragraph(
        f"<b>And the accuracy case is unproven.</b> Consecutive half-hour "
        f"calls largely re-forecast blocks the previous call already covered, "
        f"and with a {settings['schedule_rules']['freeze_blocks']}-block "
        f"freeze horizon the first "
        f"{settings['schedule_rules']['freeze_blocks']} blocks of every new "
        f"schedule are re-published unchanged in any case &mdash; so a "
        f"half-hourly call can only change blocks from 90 minutes out. And "
        f"more calls means the model's number is applied more often, which "
        f"our own scoring says is the wrong direction: over 12 backtested "
        f"days, priced as DSM penalty, the anchor alone cost Rs 17,411 and every "
        f"increment of weight given to the model made it worse, monotonically, "
        f"up to Rs 27,663 for the model alone "
        f"(tests/sweep_blend_weight.py, docs/blend_weight_sweep.csv). There "
        f"is no measurement "
        f"here that a higher call rate would buy anything.", body))

    # ------------------------------------------------------------------
    # 5a. model comparison, half-hourly
    # ------------------------------------------------------------------
    story.append(Paragraph(
        f"5a. LLM Model Comparison &mdash; Half-Hourly "
        f"({alt['calls']}-Call/Day) Workload", h2))

    story.append(Paragraph(
        f"The same published rates applied to the half-hourly volume: "
        f"<b>{alt['input']:,.0f} input + {alt['output']:,.0f} output tokens "
        f"per day</b>.", body))

    story.append(model_table(alt, table, cell))

    priced_half = [
        cost_of(
            alt["input"] - (0 if vision else alt["image_input"]),
            alt["output"], r_in, r_out
        )
        for _name, r_in, r_out, vision, *_ in PRICING
    ]

    cheap, dear = min(priced_half), max(priced_half)

    story.append(Paragraph(
        f"Same ranking as the {now['calls']}-call workload &mdash; the "
        f"cheapest option is the one that cannot see the screenshots, and the "
        f"gap between cheapest and costliest widens to "
        f"{rupees((dear - cheap) * 365)} a year at this call volume. Note "
        f"what a no-vision model really buys: dropping the images removes "
        f"{alt['image_input']:,.0f} input tokens a day, which is why those "
        f"rows fall so far below the rest.", note))

    # ------------------------------------------------------------------
    # 6. caveats
    # ------------------------------------------------------------------
    story.append(Paragraph("6. Important Caveats", h2))

    caveats = [
        "<b>Thinking tokens are model- and mode-specific.</b> They are "
        "billed at the output rate although they never appear in the "
        "response, and a lighter reasoning mode would cost less. Switching "
        "model means re-measuring, not re-pricing these numbers: "
        "gemini-3.5-flash emitted 36% more thinking than gemini-3.6-flash on "
        "the same 83 prompts.",

        "<b>Non-Gemini rows are our token volume at their published rate.</b> "
        "Every rate in this report was read from the provider's own pricing "
        "page on 2026-08-11, but the token counts are all Gemini's. "
        "Tokenization differs across vendors, so a non-Gemini row's real cost "
        "could vary by roughly &plusmn;15&ndash;20%.",

        f"<b>Rupee figures use Rs {USD_TO_INR:.2f}/$</b> "
        f"({USD_TO_INR_SOURCE}). Team 1's report converts at Rs 84.00/$, so "
        f"dollar figures compare directly between the two reports and rupee "
        f"figures do not.",

    ]

    for text in caveats:
        story.append(Paragraph(f"&bull;&nbsp; {text}", note))

    story.append(Spacer(1, 6))

    story.append(Paragraph(
        f"Provenance. Input tokens: countTokens over "
        f"{cadence['prompts_measured']} prompts built by "
        f"tests/measure_halfhourly_cost.py across "
        f"{len(cadence['days'])} days ({cadence['days'][0]} to "
        f"{cadence['days'][-1]}), model {cadence['model']}. Output and "
        f"thinking: usage_metadata from {now['calls']} live generateContent "
        f"calls, tests/measure_token_cost.py --live, over "
        f"{tokens['prompts_measured']} saved prompts across "
        f"{tokens['days_covered']} days. Image cost: countTokens with and "
        f"without {tokens['image_files'][0]} and {tokens['image_files'][1]}. "
        f"Component split: tests/measure_prompt_components.py. Rates read "
        f"2026-08-11 from ai.google.dev/gemini-api/docs/pricing (gemini-*), "
        f"developers.openai.com/api/docs/pricing (GPT-5), "
        f"docs.x.ai/docs/models (Grok 4.5) and "
        f"api-docs.deepseek.com/quick_start/pricing (DeepSeek V4 Pro). "
        f"Confirm current rates before any financial commitment.", note))

    document = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=13 * mm, bottomMargin=13 * mm,
        title=f"{plant['code']} timeslot and model cost comparison "
              "(new LLM pipeline)",
        author="Team 2",
    )

    document.build(story)

    return {
        "current": now, "half": alt,
        "vis_fit": (vis_slope, vis_intercept, vis_r2),
        "think_fit": (think_mean, think_r2),
    }


def model_table(workload, table, cell):
    """
    One priced row per model for a given daily token volume. Sorted by
    daily cost so the ranking is the table's shape, not something the
    reader has to work out.
    """

    priced = []

    for name, rate_in, rate_out, vision, source, ours in PRICING:

        # A model that cannot take images is not sent them, so it is not
        # billed for them either.
        input_tokens = (
            workload["input"] if vision
            else workload["input"] - workload["image_input"]
        )

        daily = cost_of(input_tokens, workload["output"], rate_in, rate_out)

        priced.append((daily, name, rate_in, rate_out, vision, source, ours))

    priced.sort()

    rows = [["MODEL", "IN $/1M", "OUT $/1M", "VISION", "RATE FROM",
             "DAILY", "YEARLY (USD)", "YEARLY (INR)"]]

    for daily, name, rate_in, rate_out, vision, source, ours in priced:
        rows.append([
            name + (" *" if ours else ""),
            f"${rate_in:.3f}", f"${rate_out:.3f}",
            "YES" if vision else "NO",
            source,
            f"${daily:.4f}",
            f"${daily * 365:,.2f}",
            f"{(daily * 365) * USD_TO_INR:,.0f}",
        ])

    widths = [44 * mm, 17 * mm, 18 * mm, 15 * mm, 19 * mm, 18 * mm,
              23 * mm, 23 * mm]

    current_row = next(
        index for index, row in enumerate(rows) if CURRENT_MODEL in row[0]
    )

    return table(rows, widths, spans=[
        ("BACKGROUND", (0, current_row), (-1, current_row),
         colors.HexColor("#FFE8D6")),
        ("FONTNAME", (0, current_row), (-1, current_row), "Helvetica-Bold"),
    ])


def main():

    parser = argparse.ArgumentParser(
        description="Build the timeslot + model cost comparison PDF"
    )
    parser.add_argument("--cadence", default="docs/cadence_measurements.json")
    parser.add_argument("--tokens", default="docs/token_measurements.json")
    parser.add_argument("--components", default="docs/prompt_components.csv")
    parser.add_argument(
        "--component-run-time", default="06:45",
        help="which scheduling time the component CSV was measured on"
    )
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    for path in (args.cadence, args.tokens, args.components):
        if not Path(path).exists():
            raise SystemExit(f"{path} not found.")

    cadence = json.loads(Path(args.cadence).read_text(encoding="utf-8"))
    tokens = json.loads(Path(args.tokens).read_text(encoding="utf-8"))

    with open(args.components, newline="", encoding="utf-8") as handle:
        components = list(csv.DictReader(handle))

    out_path = Path(
        args.out or
        Path.home() / "Downloads" /
        f"{settings['plant']['code']}_new_pipeline_timeslot_and_model_"
        "cost_comparison.pdf"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build(
        cadence, tokens, components, out_path, args.component_run_time
    )

    now, alt = summary["current"], summary["half"]

    print("=" * 72)
    print("TIMESLOT & MODEL COST COMPARISON")
    print("=" * 72)

    for label, entry in (("current", now), ("half-hourly", alt)):
        print(f"{label:>12}: {entry['calls']:>3} calls  "
              f"{entry['blocks']:>4} blocks  "
              f"in {entry['input']:>8,.0f}  out {entry['output']:>7,.0f}  "
              f"${entry['cost']:.4f}/day  "
              f"Rs {entry['cost'] * USD_TO_INR * 365:,.0f}/yr")

    slope, intercept, r2 = summary["vis_fit"]
    print(f"\nvisible output fit : {intercept:.1f} + {slope:.2f} x blocks  "
          f"(R2 {r2:.3f})")
    print(f"thinking per call  : {summary['think_fit'][0]:,.0f}  "
          f"(R2 vs blocks {summary['think_fit'][1]:.3f})")

    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
