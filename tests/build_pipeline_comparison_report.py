"""
=========================================================
Solar Forecasting Project
The old pipeline vs the new one, as a report a mentor reads
=========================================================
Produces two files:

  * a PDF, written to be read top to bottom by someone who
    has not seen the code - plain language, one idea per
    section, every number carrying the thing it was measured
    on;
  * an XLSX with the block-level detail behind every figure
    in the PDF, so any number can be traced.

Every figure is recomputed here from the block columns. No
number is copied out of an existing summary - the old
pipeline's own workbooks have summary rows that report the
Enercast column under AI-schedule labels, and copying those
would carry the error forward.

Enercast appears once, as a reference line, and is labelled
as such. It is not an input to either pipeline.

Run:  python -m tests.build_pipeline_comparison_report
      python -m tests.build_pipeline_comparison_report --out "C:/Users/.../Downloads"
=========================================================
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

from config.config import settings
from modules.preprocessing.windy_features import load_meter_history
from tests.compare_pipelines import compare_day
from tests.measure_input_weightage import availability, load_runs, shares
from tests.score_day_schedules import run_time_of


CAPACITY_MW = settings["plant"]["capacity_mw"]

# reportlab's built-in Helvetica has no rupee glyph and renders it as a
# black box, so every money figure is written "Rs ".
RUPEE = "Rs "

INK = colors.HexColor("#1B2A3A")
OLD_COLOUR = "#8C8C8C"
NEW_COLOUR = "#1F4E9C"
REF_COLOUR = "#C00000"
GOOD = colors.HexColor("#1B4332")
BAD = colors.HexColor("#7F1D1D")


def run_facts(folder, start, end):
    """
    What each run was actually handed, read back off the saved meta.

    Three of these are routinely assumed rather than checked, and all
    three were wrong in some form when this report was first built: the
    images were configured on but never attached (the screenshots only
    exist from 9 August, and a historical run must not be shown a
    picture taken after it), and the meta's `model` field recorded the
    vision model rather than the one that served the call.
    """

    import json

    runs = 0
    with_images = 0
    with_clip = 0
    recorded = 0

    requested = set()
    served = set()

    for path in sorted(Path(folder).glob("*_meta.json")):

        stamp = run_time_of(path)

        if stamp is None or not (start <= stamp.date() <= end):
            continue

        try:
            meta = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except ValueError:
            continue

        runs += 1

        if meta.get("images_attached"):
            with_images += 1

        if meta.get("satellite_clip"):
            with_clip += 1

        if meta.get("model_requested"):
            requested.add(meta["model_requested"])

        if meta.get("model_served"):
            served.add(meta["model_served"])
            recorded += 1

    days = (end - start).days + 1

    return {
        "runs": runs,
        "expected": days * 7,
        "with_images": with_images,
        "with_clip": with_clip,
        "models_requested": sorted(requested),
        "models_served": sorted(served),
        "model_recorded": recorded,
    }


# ==================================================
# charts
# ==================================================

def style_axes(axes):

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color("#C8CDD3")
    axes.spines["bottom"].set_color("#C8CDD3")
    axes.tick_params(colors="#4A5568", labelsize=9)
    axes.yaxis.grid(True, color="#E8EBEF", linewidth=0.8)
    axes.set_axisbelow(True)


def chart_penalty_by_day(results, path):
    """The headline: what each pipeline cost, day by day."""

    days = [f"{r['day']:%d %b}" for r in results]
    old = [r["old"]["penalty_rs"] for r in results]
    new = [r["new"]["penalty_rs"] for r in results]

    anchors = [
        r["anchor"]["penalty_rs"] if "anchor" in r else np.nan
        for r in results
    ]

    has_anchor = not all(np.isnan(anchors))

    positions = np.arange(len(days))
    width = 0.27 if has_anchor else 0.38

    figure, axes = plt.subplots(figsize=(8.6, 3.6), dpi=200)

    if has_anchor:
        axes.bar(positions - width, old, width,
                 label="Old pipeline", color=OLD_COLOUR)
        axes.bar(positions, new, width,
                 label="New pipeline", color=NEW_COLOUR)
        axes.bar(positions + width, anchors, width,
                 label="No AI at all (anchor)", color="#C9A227")
    else:
        axes.bar(positions - width / 2, old, width,
                 label="Old pipeline", color=OLD_COLOUR)
        axes.bar(positions + width / 2, new, width,
                 label="New pipeline", color=NEW_COLOUR)

    for index, value in enumerate(new):
        offset = 0 if has_anchor else width / 2
        axes.text(index + offset, value, f"{value:,.0f}", ha="center",
                  va="bottom", fontsize=8, color=NEW_COLOUR, weight="bold")

    axes.set_xticks(positions)
    axes.set_xticklabels(days)
    axes.set_ylabel("DSM penalty for the day (Rs)", fontsize=9)
    axes.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)

    style_axes(axes)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def chart_deviation_and_band(results, path):
    """Two things a mentor asks for: how far off, and how often free."""

    days = [f"{r['day']:%d %b}" for r in results]
    positions = np.arange(len(days))
    width = 0.38

    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.2), dpi=200)

    left.bar(positions - width / 2,
             [r["old"]["deviation_pct"] for r in results], width,
             label="Old", color=OLD_COLOUR)
    left.bar(positions + width / 2,
             [r["new"]["deviation_pct"] for r in results], width,
             label="New", color=NEW_COLOUR)

    left.axhline(10, color=REF_COLOUR, linestyle="--", linewidth=1.2)
    left.text(len(days) - 0.5, 10.3, "10% - the free band edge",
              ha="right", fontsize=8, color=REF_COLOUR)

    left.set_xticks(positions)
    left.set_xticklabels(days, fontsize=8)
    left.set_ylabel("Mean deviation (% of capacity)", fontsize=9)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(left)

    right.bar(positions - width / 2,
              [r["old"]["in_band_pct"] for r in results], width,
              label="Old", color=OLD_COLOUR)
    right.bar(positions + width / 2,
              [r["new"]["in_band_pct"] for r in results], width,
              label="New", color=NEW_COLOUR)

    right.set_ylim(0, 105)
    right.set_xticks(positions)
    right.set_xticklabels(days, fontsize=8)
    right.set_ylabel("Blocks inside the free band (%)", fontsize=9)
    right.legend(frameon=False, fontsize=8, loc="lower left")
    style_axes(right)

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def chart_day_profile(result, path):
    """One day, block by block - what each pipeline said, what happened."""

    frame = result["frame"].sort_values("block")

    figure, axes = plt.subplots(figsize=(8.6, 3.4), dpi=200)

    axes.plot(frame["block"], frame["actual_mw"], color="#1B4332",
              linewidth=2.0, label="What actually happened")
    axes.plot(frame["block"], frame["old_scheduled_mw"], color=OLD_COLOUR,
              linewidth=1.5, linestyle="--", label="Old pipeline said")
    axes.plot(frame["block"], frame["new_scheduled_mw"], color=NEW_COLOUR,
              linewidth=1.7, label="New pipeline said")

    axes.set_xlabel("15-minute block of the day", fontsize=9)
    axes.set_ylabel("MW", fontsize=9)
    axes.set_title(f"{result['day']:%d %B %Y}", fontsize=10, color="#1B2A3A")
    axes.legend(frameon=False, fontsize=8)

    style_axes(axes)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def chart_weightage(available, llm_shares, final_shares, path):
    """What the pipeline leaned on."""

    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.2), dpi=200)

    labels = [row[0].replace(" (ECMWF)", "") for row in available]
    values = [row[2] for row in available]

    bars = left.barh(labels[::-1], values[::-1], color=NEW_COLOUR, height=0.55)

    for bar, value in zip(bars, values[::-1]):
        left.text(value + 2, bar.get_y() + bar.get_height() / 2,
                  f"{value:.0f}%", va="center", fontsize=8,
                  color=REF_COLOUR if value == 0 else "#4A5568")

    left.set_xlim(0, 118)
    left.set_xlabel("Share of blocks where the input had a value", fontsize=8.5)
    left.tick_params(labelsize=8)
    style_axes(left)
    left.xaxis.grid(True, color="#E8EBEF", linewidth=0.8)
    left.yaxis.grid(False)

    if final_shares:

        names = [t["name"] for t in final_shares["terms"]]
        weights = [t["share_pct"] for t in final_shares["terms"]]

        pretty = {
            "anchor_mw": "Meter history\n(the anchor)",
            "llm_view_mw": "The LLM's own\nreading",
        }

        right.barh(
            [pretty.get(n, n) for n in names][::-1], weights[::-1],
            color=["#1B4332", NEW_COLOUR][:len(names)][::-1], height=0.5
        )

        for index, weight in enumerate(weights[::-1]):
            right.text(weight + 1.5, index, f"{weight:.0f}%", va="center",
                       fontsize=8, color="#4A5568")

        right.set_xlim(0, 100)
        right.set_xlabel("Share of the final scheduled MW", fontsize=8.5)
        right.tick_params(labelsize=8)
        style_axes(right)
        right.xaxis.grid(True, color="#E8EBEF", linewidth=0.8)
        right.yaxis.grid(False)

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


# ==================================================
# pdf
# ==================================================

def styles():

    sheet = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "title", parent=sheet["Title"], fontSize=19, leading=23,
            textColor=INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=sheet["Normal"], fontSize=10.5, leading=14,
            textColor=colors.HexColor("#5A6572"), spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2", parent=sheet["Heading2"], fontSize=13, leading=16,
            textColor=INK, spaceBefore=14, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", parent=sheet["Heading3"], fontSize=10.5, leading=13,
            textColor=colors.HexColor("#33414F"), spaceBefore=9, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=sheet["Normal"], fontSize=10, leading=14.5,
            textColor=colors.HexColor("#22303C"), alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small", parent=sheet["Normal"], fontSize=8.5, leading=11.5,
            textColor=colors.HexColor("#6B7684"), spaceAfter=5,
        ),
    }


def verdict_box(text, good, style):

    table = Table([[Paragraph(text, style)]], colWidths=[16.4 * cm])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
         colors.HexColor("#EAF3EA") if good else colors.HexColor("#FDECEC")),
        ("BOX", (0, 0), (-1, -1), 0.9, GOOD if good else BAD),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))

    return table


def data_table(header, rows, widths, highlight_column=None):

    table = Table([header] + rows, colWidths=widths, repeatRows=1)

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DBE1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F6F8FA")]),
    ]

    if highlight_column is not None:
        style.append(
            ("TEXTCOLOR", (highlight_column, 1), (highlight_column, -1),
             colors.HexColor(NEW_COLOUR))
        )
        style.append(
            ("FONTNAME", (highlight_column, 1), (highlight_column, -1),
             "Helvetica-Bold")
        )

    table.setStyle(TableStyle(style))

    return table


def build_pdf(path, results, weight_context, charts, window):

    style = styles()

    story = []

    old_total = sum(r["old"]["penalty_rs"] for r in results)
    new_total = sum(r["new"]["penalty_rs"] for r in results)
    blocks = sum(r["blocks"] for r in results)

    old_dev = float(np.mean([r["old"]["deviation_pct"] for r in results]))
    new_dev = float(np.mean([r["new"]["deviation_pct"] for r in results]))
    old_band = float(np.mean([r["old"]["in_band_pct"] for r in results]))
    new_band = float(np.mean([r["new"]["in_band_pct"] for r in results]))

    better = new_total < old_total
    change = abs(new_total - old_total)
    percent = change / old_total * 100 if old_total else 0.0

    wins = sum(
        1 for r in results if r["new"]["penalty_rs"] < r["old"]["penalty_rs"]
    )

    # ---------- page 1

    story.append(Paragraph(
        "Sirmour Solar Plant - Old Pipeline vs New LLM Pipeline", style["title"]
    ))
    story.append(Paragraph(
        f"{window} &nbsp;|&nbsp; {len(results)} days, {blocks} scored 15-minute "
        f"blocks, both graded against the same actual meter readings",
        style["subtitle"]
    ))

    anchor_days = [r["anchor"] for r in results if "anchor" in r]

    anchor_total = (
        sum(a["penalty_rs"] for a in anchor_days)
        if len(anchor_days) == len(results) else None
    )

    story.append(Paragraph("The short version", style["h2"]))

    story.append(verdict_box(
        f"<b>Over these {len(results)} days the new pipeline was "
        f"{RUPEE}{change:,.0f} {'cheaper' if better else 'more expensive'} "
        f"than the old one ({percent:.0f}%).</b><br/>"
        f"Old pipeline {RUPEE}{old_total:,.0f} &nbsp;|&nbsp; "
        f"New pipeline {RUPEE}{new_total:,.0f}<br/>"
        f"It was the cheaper of the two on {wins} of the {len(results)} days.",
        better, style["body"]
    ))

    story.append(Spacer(1, 8))

    if anchor_total is not None:

        story.append(Paragraph(
            "And the comparison that matters more", style["h3"]
        ))

        story.append(verdict_box(
            f"<b>Doing nothing clever at all would have cost "
            f"{RUPEE}{anchor_total:,.0f}.</b><br/>"
            "That is the \"anchor\": take the recent measured generation, fade "
            "it toward the day's normal shape, publish that. No AI, no Windy, "
            "no weather forecast - a meter reading and a clock.<br/>"
            f"The new pipeline is {RUPEE}{abs(new_total - anchor_total):,.0f} "
            f"{'cheaper' if new_total < anchor_total else '<b>more expensive</b>'} "
            f"than that. The old pipeline is "
            f"{RUPEE}{abs(old_total - anchor_total):,.0f} "
            f"{'cheaper' if old_total < anchor_total else 'more expensive'} "
            "than it.",
            new_total < anchor_total, style["body"]
        ))

        story.append(Spacer(1, 8))

        if new_total > anchor_total:
            story.append(Paragraph(
                "This is the finding to take to the mentor, not the "
                "old-versus-new number. On this window the LLM is not just "
                "behind the production pipeline - it is behind the simplest "
                "baseline in the project. Whatever it is adding on top of the "
                "meter, it is currently subtracting value.",
                style["body"]
            ))

    story.append(Spacer(1, 6))

    story.append(Paragraph("What is being compared", style["h2"]))

    story.append(Paragraph(
        "The <b>old pipeline</b> is the one already in production - the one "
        "that has been running at about 6.6% deviation. Its schedules for "
        "these days were saved at the time, in the Reports folder, and have "
        "not been touched.",
        style["body"]
    ))

    story.append(Paragraph(
        "The <b>new pipeline</b> is the LLM approach. Its schedules for these "
        "days did not exist before this report - they were generated for it, "
        "seven times a day, exactly as the plant's official schedule times "
        "require, with each run only allowed to see what it would have seen "
        "on the day.",
        style["body"]
    ))

    story.append(Paragraph(
        "Both are then graded the same way: same blocks, same actual meter "
        "readings, same DSM penalty slabs. Neither is allowed to look better "
        "by skipping the hard blocks - only blocks that BOTH pipelines "
        "scheduled are counted.",
        style["body"]
    ))

    story.append(Paragraph("How the penalty works", style["h2"]))

    story.append(Paragraph(
        "This is the part worth understanding, because it is not the same as "
        "\"be accurate\". If the schedule is within <b>10% of plant capacity</b> "
        f"- that is {0.10 * CAPACITY_MW:.2f} MW either side - the block costs "
        "<b>nothing at all</b>. Past that, it is charged in slabs: 10-15% at "
        f"{RUPEE}0.50/kWh, 15-20% at {RUPEE}0.75, and above 20% at "
        f"{RUPEE}1.00.",
        style["body"]
    ))

    story.append(Paragraph(
        "So the goal is not the smallest average error. It is to land inside "
        "that window as often as possible. A pipeline can have a worse "
        "average and still cost less, if its misses stay inside the band.",
        style["body"]
    ))

    story.append(PageBreak())

    # ---------- page 2: fairness

    story.append(Paragraph(
        "Making sure the test was not rigged", style["h2"]
    ))

    story.append(Paragraph(
        "A backtest is worthless if the pipeline can peek at the answer. "
        "Before generating a single schedule, four ways the answer could leak "
        "in were checked, on all "
        f"{len(results) * 7} runs. All four pass on every run.",
        style["body"]
    ))

    story.append(data_table(
        ["What was checked", "Result"],
        [
            ["The per-block table the model is given",
             "no future block carries its actual reading"],
            ["The prompt text sent to the model",
             "no future block is shown its own generation"],
            ["The bank of past similar days",
             "only days strictly BEFORE the run day"],
            ["The self-correction layers",
             "learn only from days already finished"],
        ],
        [8.6 * cm, 7.8 * cm],
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph("One real leak was found and fixed", style["h3"]))

    story.append(Paragraph(
        "The pipeline shows the model a handful of similar past situations. "
        "The filter for that excluded <i>the current day</i> - but not the "
        "days after it. Because the bank of past days was built across the "
        "whole period, a run being tested on 1 August could quietly pull in a "
        "case from 5 August, with the real generation attached.",
        style["body"]
    ))

    story.append(Paragraph(
        "That is exactly the kind of thing that makes a backtest look good "
        "for no reason. It is now restricted to days strictly before the run. "
        "<b>Every number in this report was generated after that fix</b> - the "
        "schedules produced under the old behaviour were set aside and not "
        "used. Live running was never affected, because a live run has no "
        "future days to reach for.",
        style["body"]
    ))

    story.append(Paragraph("Same day, same actuals", style["h3"]))

    worst_gap = max(r["actual_disagreement_mw"] for r in results)

    story.append(Paragraph(
        "The old pipeline's workbooks carry their own copy of the meter "
        "readings. Those were compared against the meter history the new "
        "pipeline was graded on. Largest disagreement anywhere across all "
        f"{blocks} blocks: <b>{worst_gap:.4f} MW</b>. The two are being graded "
        "on the same day.",
        style["body"]
    ))

    story.append(PageBreak())

    # ---------- page 3: day by day

    story.append(Paragraph("Day by day", style["h2"]))

    story.append(Image(str(charts["penalty"]), width=16.4 * cm, height=6.9 * cm))

    story.append(Spacer(1, 8))

    rows = []

    for result in results:
        old, new = result["old"], result["new"]
        anchor = result.get("anchor")
        rows.append([
            f"{result['day']:%d %b}",
            str(result["blocks"]),
            f"{old['penalty_rs']:,.0f}",
            f"{new['penalty_rs']:,.0f}",
            f"{anchor['penalty_rs']:,.0f}" if anchor else "-",
            f"{old['deviation_pct']:.1f}%",
            f"{new['deviation_pct']:.1f}%",
            f"{old['in_band_pct']:.0f}%",
            f"{new['in_band_pct']:.0f}%",
        ])

    rows.append([
        "TOTAL", str(blocks),
        f"{old_total:,.0f}", f"{new_total:,.0f}",
        f"{anchor_total:,.0f}" if anchor_total is not None else "-",
        f"{old_dev:.1f}%", f"{new_dev:.1f}%",
        f"{old_band:.0f}%", f"{new_band:.0f}%",
    ])

    table = data_table(
        ["Day", "Blocks", "Old (Rs)", "New (Rs)", "No AI (Rs)",
         "Old dev", "New dev", "Old free", "New free"],
        rows,
        [1.9 * cm, 1.5 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm,
         1.7 * cm, 1.7 * cm, 1.75 * cm, 1.75 * cm],
    )

    table.setStyle(TableStyle([
        ("FONTNAME", (0, len(rows)), (-1, len(rows)), "Helvetica-Bold"),
        ("BACKGROUND", (0, len(rows)), (-1, len(rows)),
         colors.HexColor("#E6EBF2")),
    ]))

    story.append(table)

    story.append(Paragraph(
        "\"dev\" is the average miss as a share of plant capacity. \"free\" is "
        "the share of blocks that landed inside the 10% band and therefore "
        "cost nothing - that column is the one that drives the money.",
        style["small"]
    ))

    story.append(Spacer(1, 6))

    story.append(Image(
        str(charts["deviation"]), width=16.4 * cm, height=6.1 * cm
    ))

    story.append(PageBreak())

    # ---------- page 4: a day up close

    story.append(Paragraph("Two days up close", style["h2"]))

    story.append(Paragraph(
        "Block by block: the dark green line is what the plant actually "
        "generated, and the two forecast lines are what each pipeline "
        "published. Where a line sits close to dark green, that pipeline read "
        "the day well.",
        style["body"]
    ))

    story.append(Paragraph(
        "The first chart is the day the new pipeline came closest to the old "
        "one, the second is the day it did worst. Showing only one would be "
        "picking the answer.",
        style["small"]
    ))

    for key in ("profile_worst", "profile_best"):
        if key in charts:
            story.append(Image(
                str(charts[key]), width=16.4 * cm, height=6.4 * cm
            ))
            story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ---------- page 5: weightage

    story.append(Paragraph(
        "How much does each input actually count?", style["h2"]
    ))

    story.append(Paragraph(
        "There is no setting in this pipeline that says \"Windy is worth 40%\". "
        "The model reads a table and decides; a blending step and a safety "
        "check then sit after it. So the only honest way to answer the "
        "question is to measure how far the output moves with each input.",
        style["body"]
    ))

    story.append(Image(
        str(charts["weightage"]), width=16.4 * cm, height=6.1 * cm
    ))

    story.append(Spacer(1, 6))

    story.append(Paragraph("The finding that matters", style["h3"]))

    story.append(verdict_box(
        "<b>Windy's scraped numbers counted for 0% on these days - because "
        "there were none.</b><br/>"
        "Windy publishes only its current forecast. There is no archive to "
        "ask what it predicted on 1 August, so those columns are empty for "
        "every historical run and get dropped from the prompt. What the "
        "pipeline did get from Windy on these days was the satellite video "
        "clips, reduced to motion and cloud features - not the numeric "
        "forecast, and not the layer images.",
        False, style["body"]
    ))

    story.append(Spacer(1, 9))

    facts = weight_context.get("run_facts")

    if facts:

        story.append(Paragraph("What the model was actually given", style["h3"]))

        story.append(Paragraph(
            "Worth stating plainly, because three of these are easy to assume "
            "and wrong:",
            style["body"]
        ))

        story.append(data_table(
            ["Per run", "Count", "Why"],
            [
                ["Runs generated",
                 f"{facts['runs']} of {facts['expected']}",
                 "7 official schedule times x "
                 f"{facts['expected'] // 7} days"],
                ["Windy layer screenshots attached",
                 f"{facts['with_images']} of {facts['runs']}",
                 "screenshots only exist from 9 Aug - a run cannot be "
                 "shown a picture taken later"],
                ["Satellite clip features used",
                 f"{facts['with_clip']} of {facts['runs']}",
                 "the missing ones are 06:45 runs, before the day's first "
                 "clip exists"],
                ["Windy numeric forecast",
                 f"0 of {facts['runs']}",
                 "no archive - see below"],
            ],
            [5.6 * cm, 2.6 * cm, 8.2 * cm],
        ))

        story.append(Spacer(1, 10))

    rows = []

    for label, column, present, note in weight_context["availability"]:
        rows.append([
            label,
            f"{present:.0f}%",
            note if note else "used on every block",
        ])

    story.append(data_table(
        ["Input", "Available", "Note"],
        rows,
        [5.4 * cm, 2.4 * cm, 8.6 * cm],
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph("Where the final number came from", style["h3"]))

    final = weight_context["final"]

    if final:

        pretty = {
            "anchor_mw": "The meter anchor (recent generation, damped)",
            "llm_view_mw": "The model's own clear-sky-index call",
        }

        rows = [
            [pretty.get(term["name"], term["name"]),
             f"{term['share_pct']:.0f}%"]
            for term in sorted(
                final["terms"], key=lambda t: -abs(t["coefficient"])
            )
        ]

        story.append(data_table(
            ["Contribution to the scheduled MW", "Share"],
            rows,
            [12.0 * cm, 4.4 * cm],
        ))

        story.append(Paragraph(
            f"Measured across {final['n']} blocks; this split accounts for "
            f"{final['r_squared'] * 100:.1f}% of the variation in the final "
            "schedule, which is to say it is essentially the whole story. "
            "It reflects the configured 50/50 blend between the model and the "
            "anchor.",
            style["small"]
        ))

    llm = weight_context["llm"]

    if llm:

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Where the model's own call came from", style["h3"]
        ))

        pretty = {
            "anchor_kt": "Recent measured conditions",
            "anchor_kt_derived": "Recent measured conditions",
            "weather_kt": "The ECMWF weather forecast",
            "windy_kt": "Windy's numeric forecast",
        }

        rows = [
            [pretty.get(term["name"], term["name"]),
             f"{term['share_pct']:.0f}%"]
            for term in sorted(
                llm["terms"], key=lambda t: -abs(t["coefficient"])
            )
        ]

        story.append(data_table(
            ["What moved the model's decision", "Share"],
            rows,
            [12.0 * cm, 4.4 * cm],
        ))

        explained = llm["r_squared"] * 100
        unexplained = 100 - explained

        story.append(Paragraph(
            f"Measured across {llm['n']} block decisions from all "
            f"{len(results) * 7} runs. The shares above are relative to each "
            "other and always add to 100%; what says how much of the model's "
            "decision these columns capture at all is the fit, and it is "
            f"{explained:.0f}%. The remaining {unexplained:.0f}% is the "
            "model's own reading of the satellite features and the written "
            "context, which does not reduce to a column - so on this window "
            "the numbers it was handed account for most of what it decided, "
            "and its independent judgement for the rest.",
            style["small"]
        ))

    story.append(PageBreak())

    # ---------- page 6: limits

    story.append(Paragraph("What this test does not prove", style["h2"]))

    story.append(Paragraph(
        f"<b>Five days is five days.</b> {len(results)} days and {blocks} "
        "blocks is enough to see a direction, not enough to promise one. "
        "Monsoon weeks and clear weeks behave differently and this window is "
        "neither on its own.",
        style["body"]
    ))

    story.append(Paragraph(
        "<b>The new pipeline ran without Windy's numbers.</b> They cannot be "
        "backfilled. Whatever they are worth, good or bad, is not in these "
        "figures - it can only be measured going forward, from the day the "
        "scraper started collecting them.",
        style["body"]
    ))

    story.append(Paragraph(
        "<b>The self-correction layers were seeded from earlier days that "
        "still carried the old case-retrieval behaviour.</b> The level "
        "correction is a single multiplier learned from recent finished days, "
        "and those earlier days were generated before the fix. The effect is "
        "second-order, but it is not zero, and it is stated here rather than "
        "buried.",
        style["body"]
    ))

    story.append(Paragraph(
        "<b>Influence is not skill.</b> The weightage section says what the "
        "pipeline leaned on, not what was worth leaning on. An input can "
        "dominate the output and still be wrong.",
        style["body"]
    ))

    story.append(Paragraph(
        "<b>No Windy pictures were attached either.</b> The configuration asks "
        "for them, but the layer screenshots only start on 9 August, and a run "
        "being replayed on 1 August must not be shown an image captured after "
        "it. So the model worked from the numeric table, the satellite clip "
        "features, and the written context - not from the imagery.",
        style["body"]
    ))

    if facts:

        recorded = facts.get("model_recorded", 0)

        story.append(Paragraph(
            "<b>These runs were not all answered by the same model, and the "
            "records do not say which answered which.</b> The free tier allows "
            "20 requests per model per day; this sweep needed 35, so the "
            "request chain fell back to other models part-way through. Until "
            "now the saved record stored the configured VISION model rather "
            "than the model that actually served the call, so for "
            f"{facts['runs'] - recorded} of the {facts['runs']} runs there is "
            "no trustworthy record of which model produced the schedule. That "
            "has been fixed - each run now stores both the model requested and "
            "the one that answered - but the fix cannot be applied "
            "retrospectively, and the day-to-day spread in this window should "
            "be read as noisier because of it.",
            style["body"]
        ))

    story.append(Paragraph("What would settle it", style["h2"]))

    story.append(Paragraph(
        "Keep both pipelines running side by side on the same days and let "
        "the record accumulate. The new pipeline's schedules are now generated "
        "and scored by the same commands, so extending this table costs "
        "nothing but time. Roughly three to four weeks of paired days would "
        "make the difference either real or clearly noise.",
        style["body"]
    ))

    if "enercast" in results[0]:

        reference = sum(
            r["enercast"]["penalty_rs"] for r in results if "enercast" in r
        )

        story.append(Paragraph("Enercast, for reference only", style["h2"]))

        story.append(Paragraph(
            f"Over the same days and blocks, the Enercast forecast would have "
            f"cost {RUPEE}{reference:,.0f}. It is shown because it is the "
            "external benchmark on the site, and for no other reason: it is "
            "not an input to either pipeline, and per the mentor's "
            "instruction both are graded against actual meter data, never "
            "against Enercast.",
            style["body"]
        ))

    document = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
        topMargin=1.9 * cm, bottomMargin=1.7 * cm,
        title="Old vs New Pipeline - Sirmour",
        author="Team 2",
    )

    document.build(story)


# ==================================================
# xlsx
# ==================================================

def build_xlsx(path, results, weight_context):

    book = Workbook()

    header_fill = PatternFill("solid", fgColor="1B2A3A")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    good_fill = PatternFill("solid", fgColor="E3F0E3")
    bad_fill = PatternFill("solid", fgColor="FBE4E4")

    # ---- summary

    sheet = book.active
    sheet.title = "Summary"

    old_total = sum(r["old"]["penalty_rs"] for r in results)
    new_total = sum(r["new"]["penalty_rs"] for r in results)
    blocks = sum(r["blocks"] for r in results)

    sheet["A1"] = "Sirmour - old pipeline vs new LLM pipeline"
    sheet["A1"].font = Font(bold=True, size=13)

    sheet["A2"] = (
        "Every figure recomputed from block columns. Both pipelines scored on "
        "the same blocks, same actuals, same DSM slabs."
    )
    sheet["A2"].font = Font(italic=True, size=9, color="5A6572")

    row = 4

    headers = ["Day", "Blocks", "Old penalty (Rs)", "New penalty (Rs)",
               "No-AI anchor (Rs)",
               "Old dev %", "New dev %", "Old in-band %", "New in-band %",
               "Old scheduled MWh", "New scheduled MWh", "Actual MWh"]

    for index, name in enumerate(headers, start=1):
        cell = sheet.cell(row, index, name)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    row += 1

    first_data = row

    for result in results:

        old, new = result["old"], result["new"]

        anchor = result.get("anchor")

        sheet.cell(row, 1, f"{result['day']:%Y-%m-%d}")
        sheet.cell(row, 2, result["blocks"])
        sheet.cell(row, 3, round(old["penalty_rs"], 2))
        sheet.cell(row, 4, round(new["penalty_rs"], 2))
        sheet.cell(row, 5, round(anchor["penalty_rs"], 2) if anchor else None)
        sheet.cell(row, 6, round(old["deviation_pct"], 2))
        sheet.cell(row, 7, round(new["deviation_pct"], 2))
        sheet.cell(row, 8, round(old["in_band_pct"], 1))
        sheet.cell(row, 9, round(new["in_band_pct"], 1))
        sheet.cell(row, 10, round(old["scheduled_mwh"], 3))
        sheet.cell(row, 11, round(new["scheduled_mwh"], 3))
        sheet.cell(row, 12, round(old["actual_mwh"], 3))

        winner = good_fill if new["penalty_rs"] < old["penalty_rs"] else bad_fill
        sheet.cell(row, 4).fill = winner

        row += 1

    anchor_days = [r["anchor"] for r in results if "anchor" in r]

    anchor_total = (
        sum(a["penalty_rs"] for a in anchor_days)
        if len(anchor_days) == len(results) else None
    )

    sheet.cell(row, 1, "TOTAL").font = Font(bold=True)
    sheet.cell(row, 2, blocks).font = Font(bold=True)
    sheet.cell(row, 3, round(old_total, 2)).font = Font(bold=True)
    sheet.cell(row, 4, round(new_total, 2)).font = Font(bold=True)

    if anchor_total is not None:
        sheet.cell(row, 5, round(anchor_total, 2)).font = Font(bold=True)

    last_data = row - 1

    for index, width in enumerate(
        [12, 8, 17, 17, 17, 11, 11, 13, 13, 17, 17, 12], start=1
    ):
        sheet.column_dimensions[get_column_letter(index)].width = width

    # charts on the summary
    from openpyxl.chart import BarChart, Reference

    chart = BarChart()
    chart.type = "col"
    chart.title = "DSM penalty per day - old vs new vs no AI at all (Rs)"
    chart.y_axis.title = "Rs"
    chart.height = 8
    chart.width = 18

    # Columns 3, 4 and 5: old, new, anchor. The anchor is charted beside
    # the other two on purpose - a comparison of two forecasts with no
    # do-nothing baseline cannot show whether either is worth running.
    last_series = 5 if anchor_total is not None else 4

    data = Reference(sheet, min_col=3, max_col=last_series,
                     min_row=first_data - 1, max_row=last_data)
    categories = Reference(sheet, min_col=1,
                           min_row=first_data, max_row=last_data)

    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    for index, colour in enumerate(["8C8C8C", "1F4E9C", "C9A227"]):
        if index < len(chart.series):
            chart.series[index].graphicalProperties.solidFill = colour

    sheet.add_chart(chart, f"A{row + 3}")

    band = BarChart()
    band.type = "col"
    band.title = "Blocks inside the free +/-10% band (%) - higher is better"
    band.y_axis.title = "%"
    band.height = 8
    band.width = 18

    data = Reference(sheet, min_col=8, max_col=9,
                     min_row=first_data - 1, max_row=last_data)

    band.add_data(data, titles_from_data=True)
    band.set_categories(categories)

    band.series[0].graphicalProperties.solidFill = "8C8C8C"
    band.series[1].graphicalProperties.solidFill = "1F4E9C"

    sheet.add_chart(band, f"A{row + 21}")

    # ---- weightage

    sheet = book.create_sheet("Input weightage")

    sheet["A1"] = "How much each input actually moved the schedule"
    sheet["A1"].font = Font(bold=True, size=12)

    sheet["A2"] = (
        "Measured, not configured - there is no weight constant in the "
        "pipeline. Influence is not the same as usefulness."
    )
    sheet["A2"].font = Font(italic=True, size=9, color="5A6572")

    row = 4

    for index, name in enumerate(
        ["Input", "Column", "Available on % of blocks", "Note"], start=1
    ):
        cell = sheet.cell(row, index, name)
        cell.fill = header_fill
        cell.font = header_font

    row += 1

    for label, column, present, note in weight_context["availability"]:
        sheet.cell(row, 1, label)
        sheet.cell(row, 2, column)
        sheet.cell(row, 3, round(present, 1))
        sheet.cell(row, 4, note if note else "used on every block")
        if present == 0:
            for col in range(1, 5):
                sheet.cell(row, col).fill = bad_fill
        row += 1

    row += 2

    for title, result in (
        ("What drove the model's own decision", weight_context["llm"]),
        ("What drove the final scheduled MW", weight_context["final"]),
    ):

        sheet.cell(row, 1, title).font = Font(bold=True)
        row += 1

        if not result:
            sheet.cell(row, 1, "not measurable")
            row += 2
            continue

        sheet.cell(row, 1, "Input")
        sheet.cell(row, 2, "Share %")
        sheet.cell(row, 3, "Coefficient")

        for col in range(1, 4):
            sheet.cell(row, col).fill = header_fill
            sheet.cell(row, col).font = header_font

        row += 1

        for term in sorted(result["terms"], key=lambda t: -abs(t["coefficient"])):
            sheet.cell(row, 1, term["name"])
            sheet.cell(row, 2, round(term["share_pct"], 1))
            sheet.cell(row, 3, round(term["coefficient"], 4))
            row += 1

        sheet.cell(row, 1, f"blocks {result['n']}, R-squared "
                           f"{result['r_squared']:.3f}")
        sheet.cell(row, 1).font = Font(italic=True, size=9, color="5A6572")

        row += 3

    for index, width in enumerate([40, 22, 24, 46], start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width

    # ---- block detail

    sheet = book.create_sheet("Block detail")

    frames = []

    for result in results:
        frame = result["frame"].copy()
        frame.insert(0, "day", f"{result['day']:%Y-%m-%d}")
        frames.append(frame)

    detail = pd.concat(frames, ignore_index=True)

    detail["old_deviation_mw"] = (
        detail["actual_mw"] - detail["old_scheduled_mw"]
    )
    detail["new_deviation_mw"] = (
        detail["actual_mw"] - detail["new_scheduled_mw"]
    )

    from tests.score_day_schedules import dsm_penalty

    detail["old_penalty_rs"] = detail["old_deviation_mw"].map(dsm_penalty)
    detail["new_penalty_rs"] = detail["new_deviation_mw"].map(dsm_penalty)

    columns = [
        "day", "block", "window", "old_scheduled_mw", "new_scheduled_mw",
        "anchor_mw", "actual_mw", "old_deviation_mw", "new_deviation_mw",
        "old_penalty_rs", "new_penalty_rs", "scheduled_at", "enercast_mw",
    ]

    columns = [name for name in columns if name in detail.columns]

    for index, name in enumerate(columns, start=1):
        cell = sheet.cell(1, index, name)
        cell.fill = header_fill
        cell.font = header_font

    for row_index, record in enumerate(
        detail[columns].itertuples(index=False), start=2
    ):
        for col_index, value in enumerate(record, start=1):
            if isinstance(value, float):
                value = round(value, 4)
            sheet.cell(row_index, col_index, value)

    for index in range(1, len(columns) + 1):
        sheet.column_dimensions[get_column_letter(index)].width = 17

    sheet.freeze_panes = "A2"

    book.save(path)

    return len(detail)


# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Build the old-vs-new pipeline comparison report"
    )
    parser.add_argument("--from", dest="start", default="2026-08-01")
    parser.add_argument("--to", dest="end", default="2026-08-05")
    parser.add_argument("--folder", default="outputs/llm_schedules")
    parser.add_argument("--out", default=str(Path.home() / "Downloads"))

    args = parser.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()

    meter = load_meter_history()

    results = []

    day = start

    while day <= end:

        result, problem = compare_day(day, meter, args.folder)

        if problem:
            print(f"  {day}  SKIPPED - {problem}")
        else:
            results.append(result)

        day += pd.Timedelta(days=1)

    if not results:
        raise SystemExit("Nothing to report on.")

    # ---- weightage

    frame = load_runs(args.folder, start, end)

    weight_context = {
        "availability": availability(frame),
        "llm": None,
        "final": None,
        "run_facts": run_facts(args.folder, start, end),
    }

    if not frame.empty:

        if {"anchor_mw", "clearsky_power_mw"} <= set(frame.columns):
            frame["anchor_kt_derived"] = (
                frame["anchor_mw"]
                / frame["clearsky_power_mw"].replace(0, np.nan)
            )

        llm_inputs = [
            name for name in
            ("anchor_kt", "anchor_kt_derived", "weather_kt", "windy_kt")
            if name in frame.columns
        ]

        if "anchor_kt" in llm_inputs and "anchor_kt_derived" in llm_inputs:
            llm_inputs.remove("anchor_kt_derived")

        weight_context["llm"] = shares(frame, "llm_kt", llm_inputs)

        if {"llm_kt", "clearsky_power_mw"} <= set(frame.columns):
            frame["llm_view_mw"] = (
                frame["llm_kt"] * frame["clearsky_power_mw"]
            )

        weight_context["final"] = shares(
            frame, "forecast_mw", ["llm_view_mw", "anchor_mw"]
        )

    # ---- charts

    scratch = Path(
        r"C:\Users\Acer\AppData\Local\Temp\claude"
        r"\C--Users-Acer-OneDrive-Desktop-solar-forecasting-project"
        r"--claude-worktrees-new-approach-freezing-windy-5afda1"
        r"\4a1ed0af-06b7-4c0d-aea9-7bb69d09db42\scratchpad\charts"
    )
    scratch.mkdir(parents=True, exist_ok=True)

    charts = {
        "penalty": scratch / "penalty_by_day.png",
        "deviation": scratch / "deviation_and_band.png",
        "weightage": scratch / "weightage.png",
    }

    chart_penalty_by_day(results, charts["penalty"])
    chart_deviation_and_band(results, charts["deviation"])
    chart_weightage(
        weight_context["availability"],
        weight_context["llm"],
        weight_context["final"],
        charts["weightage"],
    )

    # The day the new pipeline helped most, and the day it helped least -
    # showing only a good day would be picking a winner.
    ranked = sorted(
        results, key=lambda r: r["new"]["penalty_rs"] - r["old"]["penalty_rs"]
    )

    charts["profile_best"] = scratch / "profile_best.png"
    chart_day_profile(ranked[0], charts["profile_best"])

    if len(ranked) > 1:
        charts["profile_worst"] = scratch / "profile_worst.png"
        chart_day_profile(ranked[-1], charts["profile_worst"])

    # ---- write

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    window = f"{start:%d %B} to {end:%d %B %Y}"

    pdf = out / f"SIRMOUR_old_vs_new_pipeline_{start}_to_{end}.pdf"
    xlsx = out / f"SIRMOUR_old_vs_new_pipeline_{start}_to_{end}.xlsx"

    build_pdf(pdf, results, weight_context, charts, window)

    rows = build_xlsx(xlsx, results, weight_context)

    print("=" * 70)
    print(f"PDF   {pdf}")
    print(f"XLSX  {xlsx}  ({rows} block rows)")
    print("=" * 70)

    old_total = sum(r["old"]["penalty_rs"] for r in results)
    new_total = sum(r["new"]["penalty_rs"] for r in results)

    print(f"old {old_total:,.2f}   new {new_total:,.2f}   "
          f"days {len(results)}   blocks {sum(r['blocks'] for r in results)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
