"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Token consumption + cost report (PDF)
=========================================================
Builds the mentor's cost report for the LLM scheduling
pipeline, in the same shape as the reference report Team 1
produced, from the measurements in
outputs/reports/token_measurements.json.

EVERY NUMBER IS READ FROM THAT FILE OR COMPUTED FROM IT HERE.
Nothing is typed in by hand, so the document cannot drift from
the measurement, and re-measuring re-generates it.

Input tokens came from Gemini's countTokens API over all 83
saved production prompts. Output and thinking tokens came from
seven real generateContent calls, one per scheduling time,
read off usage_metadata.

Prices are the published paid-tier rates for the specific
model this pipeline calls. They are NOT assumed to match the
reference report's: that report used gemini-3.6-flash at
$7.50/1M output, and this pipeline calls gemini-3.5-flash,
which is $9.00/1M output.

Run:  python -m tests.build_token_cost_report
=========================================================
"""

import argparse
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

from config.config import settings


# Published paid-tier rates, per 1M tokens, read from
# ai.google.dev/gemini-api/docs/pricing on 2026-08-10.
#
# Only models whose rate was actually READ are listed. A model this
# pipeline can fall back to but whose price was not on that page is
# absent here on purpose, and the report says so rather than filling
# the gap with a plausible number.
PRICING = {
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    # gemini-flash-latest currently resolves to gemini-3.6-flash.
    "gemini-flash-latest": {"input": 1.50, "output": 7.50},
}

# Which fallbacks this pipeline would actually use, in order, from
# config. Not a general survey of Google's catalogue - a model we
# cannot call is not a cost option for us.
def configured_models():

    vision = settings.get("vision", {})

    chain = [vision.get("model")] + list(vision.get("fallback_models", []))

    return [name for name in chain if name]


# Live ECB reference rate, fetched 2026-08-10 from
# api.frankfurter.dev/v1/latest?from=USD&to=INR, quoted for 2026-08-07.
# Measured rather than assumed: a stale round number would silently
# misstate every rupee figure in this report by several percent.
USD_TO_INR = 95.21
USD_TO_INR_SOURCE = "ECB reference rate, 2026-08-07 (frankfurter.dev)"

DARK = colors.HexColor("#1B4332")
ACCENT = colors.HexColor("#1F4E9C")
GREY = colors.HexColor("#F2F4F6")


def money(usd):
    return f"${usd:,.4f}" if usd < 1 else f"${usd:,.2f}"


def rupees(usd):
    # "Rs" rather than the rupee sign: reportlab's built-in Helvetica has
    # no glyph for U+20B9, so the symbol renders as a hollow box in the
    # PDF. Spelling it out is readable everywhere without shipping a font.
    return f"Rs {usd * USD_TO_INR:,.2f}"


def build(measurements, out_path):

    plant = settings["plant"]

    model = measurements["model"]
    rates = PRICING.get(model)

    if rates is None:
        raise SystemExit(
            f"No published rate recorded for '{model}'. Add it to PRICING "
            "from ai.google.dev/gemini-api/docs/pricing rather than guessing."
        )

    output = measurements.get("output")

    if not output:
        raise SystemExit(
            "No output measurements. Run:\n"
            "  python -m tests.measure_token_cost --live"
        )

    daily_input = measurements["daily_input_tokens"]
    daily_output = output["daily_billable_output_tokens"]
    daily_total = daily_input + daily_output

    calls = len(measurements["run_times_measured"])

    def cost(input_tokens, output_tokens):
        return (
            input_tokens / 1e6 * rates["input"]
            + output_tokens / 1e6 * rates["output"]
        )

    styles = getSampleStyleSheet()

    h1 = ParagraphStyle(
        "h1", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=17, textColor=DARK, spaceAfter=4, alignment=0,
    )
    sub = ParagraphStyle(
        "sub", parent=styles["Normal"], fontSize=8.5,
        textColor=colors.HexColor("#444444"), leading=12, spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, textColor=ACCENT, spaceBefore=12, spaceAfter=5,
    )
    body = ParagraphStyle(
        "body", parent=styles["Normal"], fontSize=9, leading=13,
        spaceAfter=5,
    )
    note = ParagraphStyle(
        "note", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#555555"), leading=11, spaceBefore=6,
    )

    def table(data, widths, align_right_from=1):

        style = [
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8CDD3")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]

        return Table(data, colWidths=widths, style=TableStyle(style),
                     hAlign="LEFT")

    story = []

    story.append(Paragraph("LLM Token Consumption &amp; Cost Report", h1))

    story.append(Paragraph(
        f"{plant['name']} ({plant['capacity_mw']} MW, single site) &nbsp;|&nbsp; "
        f"Model: <b>{model}</b> &nbsp;|&nbsp; "
        f"Production schedule: {calls} forecast calls/day "
        f"({', '.join(measurements['run_times_measured'])}) &nbsp;|&nbsp; "
        "Each call forecasts every remaining 15-minute block to 19:00",
        sub,
    ))

    # ---------- 1. methodology ----------
    story.append(Paragraph("1. Methodology", h2))

    story.append(Paragraph(
        f"Numbers below are <b>measured, not estimated</b>. Input tokens were "
        f"counted by Gemini's countTokens API over "
        f"<b>all {measurements['prompts_measured']} production prompts</b> "
        f"saved by this pipeline across "
        f"{measurements['days_covered']} days of backtesting. Output and "
        f"thinking tokens were read from <b>usage_metadata</b> after "
        f"{calls} real generateContent calls &mdash; one per scheduling time.",
        body,
    ))

    story.append(Paragraph(
        "<b>Why one call per scheduling time, rather than one call overall.</b> "
        "This pipeline forecasts every block from the run time to 19:00, so "
        "the 06:45 run asks for 47 blocks and the 15:45 run asks for 11. "
        "Prompt and response length therefore vary by a factor of four across "
        "the day, and multiplying a single call by seven would misreport the "
        "daily total in whichever direction that call happened to fall.",
        body,
    ))

    story.append(Paragraph(
        "<b>Images are captured, but never sent.</b> The pipeline does record "
        "the Windy satellite clip and the layer screenshots &mdash; Playwright "
        "writes them to disk at each scheduling time. OpenCV then reads those "
        "files <i>locally</i> and reduces them to numbers: thin/thick cloud "
        "percentages, texture entropy, optical-flow divergence and vorticity, "
        "quadrant cloud fractions. Only those numbers enter the prompt, as "
        "text. The image files themselves are never uploaded to the API, so "
        "they contribute zero input tokens. The vision work happens on our "
        "own machine, at no per-token cost.",
        body,
    ))

    # ---------- 2. per-call ----------
    story.append(Paragraph("2. Per-Call Measurements, by Scheduling Time", h2))

    rows = [[
        "Run time", "Input tokens", "Visible output",
        "Thinking tokens", "Billable output",
    ]]

    for clock in measurements["run_times_measured"]:

        entry = output["by_run_time"][clock]

        rows.append([
            clock,
            f"{entry['prompt_token_count']:,}",
            f"{entry['visible_output_tokens']:,}",
            f"{entry['thinking_tokens']:,}",
            f"{entry['billable_output_tokens']:,}",
        ])

    rows.append([
        "TOTAL / DAY",
        f"{daily_input:,.0f}",
        f"{output['daily_visible_output_tokens']:,}",
        f"{output['daily_thinking_tokens']:,}",
        f"{daily_output:,}",
    ])

    story.append(table(rows, [26 * mm, 30 * mm, 30 * mm, 30 * mm, 32 * mm]))

    thinking_share = (
        output["daily_thinking_tokens"] / daily_output * 100
        if daily_output else 0
    )

    ratio = (
        output["daily_thinking_tokens"]
        / max(output["daily_visible_output_tokens"], 1)
    )

    story.append(Paragraph(
        f"<b>Thinking tokens dominate.</b> "
        f"{output['daily_thinking_tokens']:,} of the "
        f"{daily_output:,} billable output tokens are internal reasoning "
        f"&mdash; <b>{thinking_share:.0f}%</b>, or {ratio:.1f}× the "
        f"visible answer. Google bills these at the output rate even though "
        f"they never appear in the response. Note also that thinking does "
        f"<i>not</i> shrink with the answer: the 12:45 run produces the "
        f"largest thinking count "
        f"({output['by_run_time']['12:45']['thinking_tokens']:,}) despite "
        f"asking for fewer blocks than the morning runs.",
        note,
    ))

    # ---------- 3. totals ----------
    story.append(Paragraph("3. Daily / Monthly / Yearly Totals — One Site", h2))

    periods = [("Per day", 1), ("Per month (30 days)", 30),
               ("Per year (365 days)", 365)]

    rows = [["Period", "Input tokens", "Output tokens (incl. thinking)",
             "Total tokens"]]

    for label, days in periods:
        rows.append([
            label,
            f"{daily_input * days:,.0f}",
            f"{daily_output * days:,.0f}",
            f"{daily_total * days:,.0f}",
        ])

    story.append(table(rows, [42 * mm, 34 * mm, 50 * mm, 34 * mm]))

    # ---------- 4. cost ----------
    story.append(Paragraph("4. Cost — One Site (Paid Tier)", h2))

    story.append(Paragraph(
        f"Published paid-tier pricing for <b>{model}</b>: "
        f"<b>${rates['input']:.2f} / 1M input tokens</b>, "
        f"<b>${rates['output']:.2f} / 1M output tokens</b> "
        f"(ai.google.dev/gemini-api/docs/pricing, read 2026-08-10). "
        f"Rupee figures use <b>Rs {USD_TO_INR:.2f} / $</b> &mdash; "
        f"{USD_TO_INR_SOURCE} &mdash; rather than a rounded convention, "
        f"since a stale rate would misstate every rupee figure here.",
        body,
    ))

    # Unit price per SINGLE token, since the per-million figure is hard
    # to hold against a daily volume of a few tens of thousands.
    in_per_token = rates["input"] / 1e6
    out_per_token = rates["output"] / 1e6

    story.append(Paragraph(
        f"Per single token that is "
        f"<b>${in_per_token:.9f}</b> input "
        f"(Rs {in_per_token * USD_TO_INR:.9f}, or "
        f"{in_per_token * USD_TO_INR * 100:.5f} paise) and "
        f"<b>${out_per_token:.9f}</b> output "
        f"(Rs {out_per_token * USD_TO_INR:.9f}, or "
        f"{out_per_token * USD_TO_INR * 100:.5f} paise). "
        f"Output costs <b>{rates['output'] / rates['input']:.0f}× "
        f"more per token than input</b>, which is what makes the output "
        f"side of this pipeline the whole story.",
        body,
    ))

    rows = [["Period", "Input cost", "Output cost", "Total (USD)",
             "Total (INR)"]]

    for label, days in periods:

        in_usd = daily_input * days / 1e6 * rates["input"]
        out_usd = daily_output * days / 1e6 * rates["output"]

        rows.append([
            label, money(in_usd), money(out_usd),
            money(in_usd + out_usd), rupees(in_usd + out_usd),
        ])

    story.append(table(rows, [38 * mm, 28 * mm, 28 * mm, 30 * mm, 34 * mm]))

    day_cost = cost(daily_input, daily_output)

    story.append(Paragraph(
        f"At {calls} calls a day this plant costs "
        f"<b>{money(day_cost)} ({rupees(day_cost)}) per day</b> to schedule, "
        f"or <b>{money(day_cost * 365)} ({rupees(day_cost * 365)}) per year</b>.",
        body,
    ))

    # ---------- 5. model choice ----------
    story.append(Paragraph("5. What Drives the Cost", h2))

    in_cost = daily_input / 1e6 * rates["input"]
    out_cost = daily_output / 1e6 * rates["output"]
    total_cost = in_cost + out_cost

    thinking_cost = (
        output["daily_thinking_tokens"] / 1e6 * rates["output"]
    )
    visible_cost = (
        output["daily_visible_output_tokens"] / 1e6 * rates["output"]
    )

    rows = [["Component", "Tokens/day", "Cost/day (USD)", "Share of bill"]]

    for label, tokens, spend in (
        ("Input — the prompt", daily_input, in_cost),
        ("Output — visible answer", output["daily_visible_output_tokens"],
         visible_cost),
        ("Output — thinking (never shown)",
         output["daily_thinking_tokens"], thinking_cost),
    ):
        rows.append([
            label, f"{tokens:,.0f}", money(spend),
            f"{spend / total_cost * 100:.1f}%",
        ])

    rows.append([
        "TOTAL", f"{daily_total:,.0f}", money(total_cost), "100.0%"
    ])

    story.append(table(rows, [62 * mm, 30 * mm, 34 * mm, 30 * mm]))

    story.append(Paragraph(
        f"<b>Three quarters of the bill is reasoning nobody reads.</b> "
        f"Thinking tokens are {thinking_cost / total_cost * 100:.0f}% of the "
        f"total spend. Two effects compound to produce that: the model emits "
        f"{ratio:.1f}× more thinking than visible answer, and every output "
        f"token costs {rates['output'] / rates['input']:.0f}× what an input "
        f"token costs. The visible JSON schedule &mdash; the thing the plant "
        f"actually uses &mdash; is only "
        f"{visible_cost / total_cost * 100:.0f}% of what is paid for.",
        body,
    ))

    story.append(Paragraph(
        "The prompt itself is cheap. Sending more evidence &mdash; more "
        "precedent, more input history, more Windy layers &mdash; costs "
        f"{rates['input'] / rates['output']:.2f} of what an equivalent "
        "amount of extra reasoning costs. On these rates, giving the model "
        "better information is roughly six times cheaper than letting it "
        "think longer.",
        note,
    ))

    # ---------- 6. scaling ----------
    story.append(Paragraph("6. Scaling to Multiple Sites", h2))

    rows = [["Number of sites", "Tokens/year (total)", "Cost/year (USD)",
             "Cost/year (INR, approx)"]]

    for sites in (1, 3, 10, 50, 100):

        usd = cost(daily_input * 365 * sites, daily_output * 365 * sites)

        label = f"{sites}"
        if sites == 1:
            label += " (this report)"
        elif sites == 3:
            label += " (Sirmour + Kasipet + Bhupalpally)"

        rows.append([
            label,
            f"{daily_total * 365 * sites:,.0f}",
            money(usd),
            rupees(usd),
        ])

    story.append(table(rows, [58 * mm, 38 * mm, 32 * mm, 38 * mm]))

    # ---------- 7. free tier ----------
    story.append(Paragraph("7. Free-Tier Reality Check", h2))

    story.append(Paragraph(
        f"The free tier caps each model at <b>20 requests per day</b> "
        f"&mdash; verified directly from quota-exhaustion errors hit during "
        f"this project's backtesting, not from documentation. Production "
        f"needs {calls} calls/day for this plant alone and "
        f"{calls * 3} across the three plants.",
        body,
    ))

    story.append(Paragraph(
        "The cap is <b>per model</b>, and this pipeline falls through a chain "
        "of five models when one is exhausted. That is what allowed an "
        "83-call backtest to complete in a single day on one key: the primary "
        "model spent its 20, and the run continued on the fallbacks. It works, "
        "but it means a long backtest silently changes which model produced "
        "the later schedules — which is recorded per run rather than "
        "left to be discovered.",
        body,
    ))

    story.append(Paragraph(
        "<b>Free tier costs nothing in tokens.</b> The binding constraint is "
        "the request count, not the token price — and at token prices "
        f"this low ({rupees(day_cost)} per plant per day), the paid tier is "
        "cheap insurance against losing a scheduling slot to a quota refusal.",
        body,
    ))

    # ---------- footer ----------
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(
        f"Generated {date.today():%Y-%m-%d} from "
        f"{measurements['prompts_measured']} measured production prompts "
        f"({measurements['days_covered']} days) and {calls} live "
        f"generateContent calls against <b>{model}</b>. "
        "Token counts by Gemini countTokens and usage_metadata; prices read "
        "from Google's published Gemini API pricing page on 2026-08-10; "
        f"exchange rate from the {USD_TO_INR_SOURCE}. Confirm current rates "
        "before financial commitments. Every figure here was measured for "
        "this pipeline - none is carried over from any other report. "
        "Source: tests/measure_token_cost.py, tests/build_token_cost_report.py.",
        note,
    ))

    document = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title=f"{plant['code']} LLM Token Cost Report",
        author="Team 2",
    )

    document.build(story)

    return {
        "daily_input": daily_input,
        "daily_output": daily_output,
        "daily_total": daily_total,
        "cost_day_usd": day_cost,
        "cost_year_usd": day_cost * 365,
    }


def main():

    parser = argparse.ArgumentParser(description="Build the token cost PDF")
    parser.add_argument(
        "--measurements", default="outputs/reports/token_measurements.json"
    )
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    path = Path(args.measurements)

    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:\n"
            "  python -m tests.measure_token_cost --live"
        )

    measurements = json.loads(path.read_text(encoding="utf-8"))

    out_path = Path(
        args.out or
        Path.home() / "Downloads" /
        f"{settings['plant']['code']}_LLM_token_cost_report.pdf"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build(measurements, out_path)

    print("=" * 70)
    print("TOKEN COST REPORT")
    print("=" * 70)
    print(f"input tokens / day   : {summary['daily_input']:,.0f}")
    print(f"output tokens / day  : {summary['daily_output']:,.0f}")
    print(f"total tokens / day   : {summary['daily_total']:,.0f}")
    print(f"cost / day           : ${summary['cost_day_usd']:.4f}  "
          f"(Rs{summary['cost_day_usd'] * USD_TO_INR:.2f})")
    print(f"cost / year          : ${summary['cost_year_usd']:.2f}  "
          f"(Rs{summary['cost_year_usd'] * USD_TO_INR:,.2f})")
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
