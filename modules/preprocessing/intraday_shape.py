"""
=========================================================
Solar Forecasting Project - NEW APPROACH
What KIND of day has today been so far?
=========================================================
Turns today's measured blocks into a description of the day's
CHARACTER - steady or choppy, clearing or clouding, and how
violently it has been moving - and hands that to the model as
a sentence rather than as twelve numbers to squint at.

Adopted from step 8 of Team 1's pipeline map ("intraday
actuals analysis: trend / choppiness / clear-sky-ratio math on
today's own readings"). We were already sending the raw
readings; what was missing was the summary.

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
Our measured failure is the model over-extrapolating. On
2026-07-27 it published 16.14 MWh against 8.95 actual with
weather, and 5.81 without - over on 31 of 35 blocks one way,
under on 33 of 35 the other - while damped persistence sat
within 1% all day.

A choppy day is exactly when extrapolating a recent trend is
worst: the trend you are extending is noise. But nothing in
the prompt SAID the day was choppy. The model had to notice it
by reading a column of numbers, and it did not.

Choppiness is measured here as REVERSALS - how often the
clear-sky index changed direction - rather than as variance
alone. A day that falls steadily from 0.9 to 0.3 has high
variance and is perfectly predictable; a day that bounces
0.4/0.8/0.35/0.75 has similar variance and is not. Direction
changes separate the two, and it is the second kind that
punishes extrapolation.
=========================================================
"""

import numpy as np
import pandas as pd


def describe(features, min_blocks=4):
    """
    A dict describing today so far, or None when too little of the day
    has been measured for any of it to mean anything.

    Reads the same feature table the prompt is built from, so it can
    never disagree with the numbers shown beside it.
    """

    if "actual_kt" not in features.columns:
        return None

    today = features[
        (features["is_past"] == 1) & features["actual_kt"].notna()
    ].sort_values("timestamp")

    if len(today) < min_blocks:
        return None

    kt = today["actual_kt"].to_numpy(dtype=float)

    steps = np.diff(kt)

    # A direction change only counts when both moves are big enough to
    # be weather rather than meter noise. Without a floor, a flat
    # afternoon wobbling in the fourth decimal reads as violently choppy.
    meaningful = 0.03

    reversals = 0

    last_direction = 0

    for step in steps:

        if abs(step) < meaningful:
            continue

        direction = 1 if step > 0 else -1

        if last_direction and direction != last_direction:
            reversals += 1

        last_direction = direction

    # Trend over the last two hours, which is what a forecast for the
    # next block would actually be extrapolating.
    recent = kt[-8:] if len(kt) >= 8 else kt

    if len(recent) >= 3:
        slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
    else:
        slope = 0.0

    biggest_jump = float(np.max(np.abs(steps))) if len(steps) else 0.0

    return {
        "blocks": int(len(kt)),
        "kt_mean": float(kt.mean()),
        "kt_min": float(kt.min()),
        "kt_max": float(kt.max()),
        "kt_std": float(kt.std()),
        "kt_now": float(kt[-4:].mean()),
        "reversals": reversals,
        "reversals_per_hour": reversals / (len(kt) / 4) if len(kt) else 0.0,
        "slope_per_block": slope,
        "biggest_jump": biggest_jump,
    }


def history_choppiness(before_date, days=10, meter=None):
    """
    Reversals-per-hour for each recent finished day, so today can be
    placed against this plant's own normal rather than an absolute
    threshold.

    Reads the METER, not the feature table: the feature table holds one
    day, and the comparison needs many. Clear-sky index is recomputed
    here through the same pvlib model the rest of the pipeline uses, so
    the history and today are measured identically.

    Returns [] when there is not enough history, which leaves the
    description purely numeric - better than a label with nothing
    behind it.
    """

    from modules.forecasting.clearsky import ClearSkyModel
    from modules.preprocessing.windy_features import load_meter_history

    if meter is None:
        meter = load_meter_history()

    if meter is None or meter.empty or "ghi_w_m2" not in meter.columns:
        return []

    before = pd.Timestamp(before_date)

    if before.tz is not None:
        before = before.tz_localize(None)

    frame = meter.copy()

    if frame["timestamp"].dt.tz is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)

    frame = frame[frame["timestamp"] < before]

    if frame.empty:
        return []

    wanted = sorted({t.date() for t in frame["timestamp"]})[-days:]

    frame = frame[frame["timestamp"].dt.date.isin(wanted)]

    if frame.empty:
        return []

    with_kt = ClearSkyModel().compute_clear_sky_index(
        frame, ghi_column="ghi_w_m2", timestamp_column="timestamp"
    )

    with_kt = with_kt.dropna(subset=["clear_sky_index"])

    values = []

    for day in wanted:

        kt = with_kt.loc[
            with_kt["timestamp"].dt.date == day, "clear_sky_index"
        ].to_numpy(dtype=float)

        if len(kt) < 8:
            continue

        values.append(_reversals_per_hour(kt))

    return values


def _reversals_per_hour(kt, meaningful=0.03):

    steps = np.diff(kt)

    reversals = 0
    last = 0

    for step in steps:

        if abs(step) < meaningful:
            continue

        direction = 1 if step > 0 else -1

        if last and direction != last:
            reversals += 1

        last = direction

    return reversals / (len(kt) / 4) if len(kt) else 0.0


def classify(shape, reference=None):
    """
    (character, guidance) - what kind of day this is, and what that
    implies for extrapolating.

    JUDGED AGAINST THIS PLANT'S OWN RECENT DAYS, not against a fixed
    threshold. The first version used absolute cut-offs and labelled 35
    of 36 backtested runs VOLATILE - true, in that monsoon Sirmour
    genuinely is choppy most days, and useless, because a label that
    fires 97% of the time carries no information. What a forecaster
    needs is whether today is choppier than usual HERE.

    `reference` is the recent distribution of reversals-per-hour. With
    fewer than four days of it the label is withheld entirely and the
    raw numbers speak for themselves - the numbers were always the
    information, the label is only a convenience.
    """

    if shape is None:
        return None, None

    if not reference or len(reference) < 4:
        return None, None

    reference = np.asarray(reference, dtype=float)

    today = shape["reversals_per_hour"]

    percentile = float((reference < today).mean() * 100)

    if percentile >= 75:
        return (
            f"CHOPPIER THAN USUAL (calmer on {percentile:.0f}% of recent days)",
            "the sky is changing direction more often than it normally does "
            "here, so any trend you extend from the last few blocks is as "
            "likely to be noise as signal. Stay close to the anchor and use "
            "low confidence.",
        )

    if percentile <= 25:
        return (
            f"CALMER THAN USUAL (choppier on {100 - percentile:.0f}% of "
            "recent days)",
            "conditions are steadier than this plant's normal, so the last "
            "measured cloudiness is a stronger guide than it usually would "
            "be and a confident adjustment is more defensible.",
        )

    return (
        f"TYPICAL for this plant ({percentile:.0f}th percentile of recent "
        "days)",
        "the day is behaving about as this plant usually does. Weigh the "
        "anchor and the forecasts on their own merits rather than treating "
        "today as unusual in either direction.",
    )


def prompt_section(features, reference=None):
    """
    The description as prompt text, or None when today is too young to
    describe.

    `reference` is this plant's recent reversals-per-hour distribution.
    Without it the numbers are still reported - they are the actual
    information - but no comparative label is claimed.
    """

    shape = describe(features)

    if shape is None:
        return None

    character, guidance = classify(shape, reference)

    direction = (
        "clearing" if shape["slope_per_block"] > 0.005
        else "clouding over" if shape["slope_per_block"] < -0.005
        else "flat"
    )

    lines = [
        f"Measured over {shape['blocks']} block(s) so far today:",
        f"  clear-sky index: now {shape['kt_now']:.2f}, "
        f"today's range {shape['kt_min']:.2f} to {shape['kt_max']:.2f}, "
        f"mean {shape['kt_mean']:.2f}",
        f"  direction changes: {shape['reversals']} "
        f"({shape['reversals_per_hour']:.1f} per hour)",
        f"  recent 2-hour trend: {direction} "
        f"({shape['slope_per_block']:+.3f} per block)",
        f"  largest single-block move: {shape['biggest_jump']:.2f}",
    ]

    if character:
        lines.append(f"\nAgainst this plant's recent days: {character}.")
        lines.append(f"What that implies: {guidance}")

    return "\n".join(lines)
