"""
=========================================================
Solar Forecasting Project - NEW APPROACH
LLM Fusion and Scheduling Decision
=========================================================
The sub-mentor's architecture, final stage: everything we
know arrives as ONE table, an LLM reads it, and the LLM
decides the schedule.

    Windy values (scraped)  ----\
    meter data so far       -----> feature table --> LLM --> schedule
    clear-sky physics       ----/
    weather forecast        ---/

WHAT THE MODEL IS ASKED FOR, AND WHY IT IS NOT MEGAWATTS
--------------------------------------------------------
The model returns a CLEAR-SKY INDEX (kt) per block - "what
fraction of a perfectly clear sky will actually reach the
panels" - and this module multiplies that by the pvlib
clear-sky curve to get MW.

The model is still making the scheduling decision. It sets
the number that decides every block. What it is NOT asked to
do is arithmetic:

  * kt x clear-sky is exact physics. pvlib already computes
    the sunrise, sunset and seasonal shape to the second.
    Asking a language model to redo that multiplication 96
    times invites 96 chances to slip a digit, and the
    reference run showed exactly that pattern - its output
    was 1.9918 x sin(solar elevation), a trigonometric
    identity it had rederived rather than reasoned about.
  * Raw MW hides mistakes. A kt of 1.4 is instantly wrong
    (nothing beats clear sky by 40%) and gets clipped here;
    "4.9 MW" at 5 pm looks perfectly reasonable until it is
    graded a day later.
  * kt is comparable across plants and seasons, so one
    prompt works for Sirmour, Kasipet and Bhupalpally
    without retuning.

Set fusion.llm_output to "mw" in config to have the model
emit megawatts directly instead - the parsing handles both.

THE FREEZE HORIZON IS NOT THE MODEL'S JOB
-----------------------------------------
Blocks already declared to the grid operator cannot be
rewritten, whatever the model says about them. That rule is
applied AFTER the response, by the same
modules/scheduling/effective_time.py the existing pipeline
uses. An LLM asked politely to leave the first six blocks
alone will sometimes not, and "sometimes" is not a control.
=========================================================
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from modules.fusion.validator import ScheduleValidator, recent_daily_bias
from modules.scheduling.effective_time import apply_freeze, freeze_window
from modules.vision.json_parser import JSONParser
from utils.logger import get_logger


# Columns handed to the model, in this order. Deliberately short:
# every extra column is tokens spent and one more thing to be
# distracted by, and these are the ones that carry the decision.
_PROMPT_COLUMNS = [
    "block", "time", "clearsky_power_mw", "windy_kt",
    "windy_clouds_pct", "windy_lclouds_pct", "windy_mclouds_pct",
    "windy_hclouds_pct", "windy_rain_mm", "windy_is_measured",
]


class LLMScheduler:

    def __init__(self):

        self.logger = get_logger()

        self.parser = JSONParser()

        plant = settings["plant"]

        self.plant_name = plant.get("name", "the plant")
        self.plant_tag = plant.get("code", "SIRMOUR")
        self.capacity_mw = plant["capacity_mw"]
        self.timezone = plant["timezone"]

        self.interval_minutes = settings["forecast"]["interval_minutes"]

        fusion = settings.get("fusion", {})

        self.output_mode = fusion.get("llm_output", "kt")
        self.max_kt = fusion.get("max_clear_sky_index", 1.2)
        self.temperature = fusion.get("temperature", 0.1)
        self.max_output_tokens = fusion.get("max_output_tokens", 8192)

        self.freeze_blocks = settings.get("schedule_rules", {}).get(
            "freeze_blocks", 0
        )

        self.output_dir = Path(fusion.get("output_dir", "outputs/llm_schedules"))

    # --------------------------------------------------

    def anchor_mw(self, ahead):
        """
        The physics baseline each block is measured against.

        Windy's own forecast clear-sky index where it exists, since
        that is a real forward-looking number; otherwise the last
        measured kt held flat; otherwise clear sky. Multiplied by the
        pvlib curve, so the anchor always carries exact solar geometry
        whatever the cloudiness estimate came from.

        This is what the validator bounds the LLM against - so it has
        to be a number we would be willing to publish on its own.
        """

        clearsky = ahead["clearsky_power_mw"].to_numpy(dtype=float)

        kt = ahead["windy_kt"].to_numpy(dtype=float).copy()

        if "actual_kt" in ahead.columns:
            fallback = float(
                pd.Series(ahead["actual_kt"]).dropna().tail(4).mean()
            ) if ahead["actual_kt"].notna().any() else np.nan
        else:
            fallback = np.nan

        if not np.isfinite(fallback):
            fallback = 1.0

        kt = np.where(np.isfinite(kt), kt, fallback)

        return np.clip(kt * clearsky, 0.0, self.capacity_mw)

    # --------------------------------------------------

    def deviation_limit(self, run_time):
        """
        How far the model may stray from the anchor on this run - wider
        when recent finished days were consistently biased one way (see
        modules/fusion/validator.py).
        """

        try:
            bias = recent_daily_bias(
                self.output_dir, as_of=pd.Timestamp(run_time).date()
            )
        except Exception:
            bias = []

        return self.validator.suggested_max_deviation_fraction(bias)

    # --------------------------------------------------

    def client(self):
        """
        Built lazily so importing this module never requires an API
        key - the prompt can be inspected, and the whole table built
        and tested, without one.
        """

        from modules.vision.gemini_client import GeminiClient

        return GeminiClient()

    # --------------------------------------------------

    def recent_history(self, features, limit=12):
        """
        Today's measured blocks so far, as compact text.

        This is the single strongest clue about the rest of the day -
        what the plant has actually done in the last three hours beats
        any forecast of what it might do.
        """

        past = features[
            (features["is_past"] == 1)
            & features["actual_power_mw"].notna()
        ].tail(limit)

        if past.empty:
            return "(no measured generation for today yet)"

        lines = []

        for _, row in past.iterrows():

            kt = row.get("actual_kt")
            kt_text = f", measured kt {kt:.2f}" if pd.notna(kt) else ""

            lines.append(
                f"  block {int(row['block'])} {row['time']}  "
                f"{row['actual_power_mw']:.3f} MW "
                f"(clear-sky would be {row['clearsky_power_mw']:.3f} MW"
                f"{kt_text})"
            )

        return "\n".join(lines)

    # --------------------------------------------------

    def forecast_table(self, features, run_time):
        """
        The blocks still to be scheduled, as a fixed-width text table.

        Only daylight blocks are sent. Night blocks are zero by
        physics, so spending tokens on them - and giving the model a
        chance to put something other than zero there - buys nothing.
        """

        ahead = features[
            (features["timestamp"] > run_time)
            & (features["is_daylight"] == 1)
        ]

        if ahead.empty:
            return "", ahead

        columns = [c for c in _PROMPT_COLUMNS if c in ahead.columns]

        header = " | ".join(f"{c}" for c in columns)

        lines = [header, "-" * len(header)]

        for _, row in ahead.iterrows():

            cells = []

            for column in columns:

                value = row[column]

                if isinstance(value, str):
                    cells.append(value)
                elif pd.isna(value):
                    cells.append("n/a")
                elif column in ("block", "windy_is_measured"):
                    cells.append(str(int(value)))
                else:
                    cells.append(f"{value:.2f}")

            lines.append(" | ".join(cells))

        return "\n".join(lines), ahead

    # --------------------------------------------------

    def satellite_section(self, satellite, run_time):
        """
        The satellite clip as a CURRENT SKY OBSERVATION block.

        Deliberately not columns in the block table. One clip
        describes one moment; pasting it down 30 rows would make the
        model read a single observation as if it were a forecast that
        held all afternoon - the precise mistake that made the
        reference features log worthless. Presenting it as "here is
        the sky right now, and here is how old that is" says what it
        actually is.
        """

        if not satellite:
            return "(no satellite clip available for this run)"

        captured = satellite.get("sat_captured_at")

        age = ""

        if captured:
            minutes = (run_time - pd.Timestamp(captured)).total_seconds() / 60
            age = f", captured {minutes:.0f} minutes ago"

        trend = satellite["sat_cloud_trend_pct"]
        entropy = satellite["sat_entropy"]

        building = (
            "converging (cloud building)"
            if satellite["sat_flow_divergence"] < 0
            else "spreading out (cloud dissipating)"
        )

        direction = "clouding over" if trend > 0 else "clearing"

        texture = (
            "broken and patchy - expect volatile blocks" if entropy > 6
            else "fairly uniform sky - blocks should be steady"
        )

        return f"""Satellite imagery over the plant{age}:
  cloud cover      : {satellite['sat_thick_cloud_pct']:.0f}% thick, \
{satellite['sat_thin_cloud_pct']:.0f}% thin, \
{satellite['sat_clear_pct']:.0f}% clear
  trend over clip  : {trend:+.1f}% cloud ({direction})
  texture entropy  : {entropy:.2f} of 8 ({texture})
  where the cloud is: north {satellite['sat_north_cloud_pct']:.0f}%, \
south {satellite['sat_south_cloud_pct']:.0f}%, \
west {satellite['sat_west_cloud_pct']:.0f}%, \
east {satellite['sat_east_cloud_pct']:.0f}%
  motion structure : field is {building}; \
rotation {satellite['sat_flow_vorticity']:.3f}, \
movement energy {satellite['sat_flow_kinetic_energy']:.1f}

This is a MEASUREMENT of the sky that is actually there, unlike the Windy \
columns below which are a forecast. Where the two disagree about conditions \
NOW, the satellite is the observation. It says nothing directly about later \
blocks - use it to judge whether the forecast has the current state right, \
and note that Windy's satellite animation dissolves between hourly stills, so \
cloud DIRECTION is not measurable from it and is not reported."""

    # --------------------------------------------------

    def build_prompt(self, features, run_time, ahead, satellite=None):

        table, _ = self.forecast_table(features, run_time)

        history = self.recent_history(features)

        sky = self.satellite_section(satellite, run_time)

        # Counted over the blocks actually IN the table, not over every
        # remaining block of the day. Windy's 3-hourly steps land at
        # 08:30/11:30/14:30/17:30/20:30/23:30, so counting the whole
        # remainder told the model "3 real readings" for a table that
        # contained one - the other two were night blocks it never saw.
        measured = (
            int(ahead["windy_is_measured"].sum())
            if "windy_is_measured" in ahead.columns else 0
        )

        first_block = int(ahead["block"].iloc[0])
        last_block = int(ahead["block"].iloc[-1])
        count = len(ahead)

        wanted = (
            "clear-sky index kt (0.00-1.20)"
            if self.output_mode == "kt"
            else "power in MW"
        )

        return f"""You are scheduling grid export for {self.plant_name}, a \
{self.capacity_mw} MW solar plant in India. Blocks are 15 minutes; block 1 is \
00:00-00:15. Current time is {run_time:%Y-%m-%d %H:%M} local.

WHAT THE PLANT HAS ACTUALLY GENERATED TODAY SO FAR
{history}

CURRENT SKY OBSERVATION (satellite)
{sky}

BLOCKS STILL TO SCHEDULE ({count} blocks, {first_block} to {last_block})
{table}

HOW TO READ THE COLUMNS
- clearsky_power_mw: what this plant produces under a perfectly clear sky at \
that moment. Exact physics from pvlib - solar geometry, tilt and capacity. \
Treat it as ground truth for the SHAPE of the day.
- windy_kt: Windy's ECMWF solar-power forecast divided by clear-sky irradiance, \
i.e. the fraction of clear sky it expects to get through.
- windy_clouds_pct / lclouds / mclouds / hclouds: total, low, middle and high \
cloud cover. Low thick cloud attenuates far more than thin high cloud, so the \
split matters more than the total.
- windy_rain_mm: forecast rainfall. Rain implies heavy cloud.
- windy_is_measured: 1 means Windy had a REAL reading at that block. 0 means the \
value was interpolated between readings up to 90 minutes apart. Free-tier Windy \
only updates every 3 hours, so today only {measured} of the remaining blocks \
carry a real reading. Trust the 1s; treat the 0s as a smooth guess between them.

YOUR JOB
For every block listed above, decide the {wanted} you expect. Weigh what the \
plant has actually been doing today against what Windy forecasts. Where the two \
disagree, say which you trusted and why. Cloud fields move, so a change Windy \
shows at one reading usually arrives gradually across the blocks around it \
rather than instantly.

Being wrong is penalised in both directions - over-forecasting costs the plant \
money in deviation charges just as under-forecasting does. Do not pad the \
number for safety.

RESPOND WITH JSON ONLY, no prose outside it, in exactly this form:
{{
  "regime": "clear" | "partly_cloudy" | "overcast" | "storm",
  "confidence": 0.0-1.0,
  "reasoning": "2-4 sentences on what drove the decision",
  "blocks": [[{first_block}, 0.00], [{first_block + 1}, 0.00], ...]
}}

"blocks" must be a list of [block_number, value] pairs covering every one of \
the {count} blocks from {first_block} to {last_block}, in order, with no gaps.\
"""

    # --------------------------------------------------

    def parse_blocks(self, payload, ahead):
        """
        The model's block list -> a kt value per forecast block.

        Missing blocks are filled by interpolating the ones that did
        arrive rather than defaulting to clear sky: a truncated
        response must not silently become an optimistic forecast for
        the rest of the day.
        """

        values = {}

        for entry in payload.get("blocks", []):

            if isinstance(entry, dict):
                number, value = entry.get("block"), entry.get("value")
                if value is None:
                    value = entry.get("kt", entry.get("power_mw"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                number, value = entry[0], entry[1]
            else:
                continue

            try:
                values[int(number)] = float(value)
            except (TypeError, ValueError):
                continue

        blocks = ahead["block"].to_numpy()

        missing = [b for b in blocks if b not in values]

        if len(missing) == len(blocks):
            raise ValueError("model returned no usable blocks")

        if missing:
            self.logger.warning(
                f"Model omitted {len(missing)} of {len(blocks)} blocks "
                f"({missing[:6]}{'...' if len(missing) > 6 else ''}) "
                "- filled by interpolating the blocks it did return"
            )

            known = np.array(sorted(values))
            series = np.array([values[b] for b in known], dtype=float)

            for block in missing:
                values[block] = float(np.interp(block, known, series))

        return np.array([values[b] for b in blocks], dtype=float)

    # --------------------------------------------------

    def to_power(self, raw, ahead):
        """
        The model's numbers -> MW, clipped to what the plant can
        physically do.
        """

        clearsky = ahead["clearsky_power_mw"].to_numpy(dtype=float)

        if self.output_mode == "kt":

            kt = np.clip(raw, 0.0, self.max_kt)

            return kt, np.clip(kt * clearsky, 0.0, self.capacity_mw)

        power = np.clip(raw, 0.0, self.capacity_mw)

        kt = np.divide(
            power, clearsky,
            out=np.full(len(power), np.nan),
            where=clearsky > 0.01,
        )

        return kt, power

    # --------------------------------------------------

    def apply_freeze_horizon(self, schedule, run_time, previous):
        """
        Holds blocks already declared to the grid operator at their
        published values, whatever the model said about them.
        """

        if self.freeze_blocks <= 1 or not previous:
            return schedule, []

        published, frozen = apply_freeze(
            new_values=dict(zip(schedule["timestamp"], schedule["forecast_mw"])),
            previous_values=previous,
            run_time=run_time,
            freeze_blocks=self.freeze_blocks,
            interval_minutes=self.interval_minutes,
        )

        schedule["forecast_mw"] = schedule["timestamp"].map(published)

        first, last, effective = freeze_window(
            run_time, self.freeze_blocks, self.interval_minutes
        )

        self.logger.info(
            f"Effective time: blocks {first}-{last} held at the previous "
            f"schedule ({len(frozen)} actually frozen); this run takes "
            f"effect from block {effective}"
        )

        return schedule, frozen

    # --------------------------------------------------

    def decide(self, features, run_time, previous=None, dry_run=False,
               satellite=None):
        """
        One fusion + scheduling decision.

        Returns (schedule, meta). `dry_run` builds and returns the
        prompt without spending a request, so the prompt can be
        reviewed before any quota is used.
        """

        run_time = pd.Timestamp(run_time)

        if run_time.tz is None:
            run_time = run_time.tz_localize(self.timezone)

        _, ahead = self.forecast_table(features, run_time)

        if ahead.empty:
            raise ValueError(
                f"No daylight blocks left to schedule after {run_time:%H:%M}"
            )

        prompt = self.build_prompt(features, run_time, ahead, satellite)

        meta = {
            "run_time": str(run_time),
            "plant": self.plant_tag,
            "blocks_requested": len(ahead),
            "output_mode": self.output_mode,
            "satellite_clip": (satellite or {}).get("sat_video"),
            "prompt_chars": len(prompt),
            "prompt": prompt,
        }

        if dry_run:
            return None, meta

        response = self.client().generate_text(
            prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        payload = self.parser.parse(response)

        raw = self.parse_blocks(payload, ahead)

        kt, power = self.to_power(raw, ahead)

        schedule = pd.DataFrame({
            "block": ahead["block"].to_numpy(),
            "timestamp": ahead["timestamp"].to_numpy(),
            "time": ahead["time"].to_numpy(),
            "clearsky_power_mw": ahead["clearsky_power_mw"].to_numpy(),
            "windy_kt": ahead["windy_kt"].to_numpy(),
            "anchor_mw": self.anchor_mw(ahead),
            "llm_kt": kt,
            "forecast_mw": power,
        })

        # Guard rails BEFORE the freeze horizon. Freezing republishes
        # values already sent to the grid operator, and those were
        # validated when they were first published - re-validating them
        # against today's anchor would rewrite a committed block.
        schedule, validator_notes = self.validator.validate(
            schedule, max_deviation_fraction=self.deviation_limit(run_time)
        )

        schedule, frozen = self.apply_freeze_horizon(schedule, run_time, previous)

        meta.update({
            "regime": payload.get("regime"),
            "confidence": payload.get("confidence"),
            "reasoning": payload.get("reasoning"),
            "frozen_blocks": len(frozen),
            "adjusted_blocks": int(schedule["was_adjusted"].sum()),
            "validator_notes": validator_notes,
            "model": settings["vision"]["model"],
        })

        return schedule, meta

    # --------------------------------------------------

    def save(self, schedule, meta, run_time):

        self.output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.plant_tag}_{pd.Timestamp(run_time).strftime('%Y-%m-%d_%H-%M')}"

        prompt_path = self.output_dir / f"{stem}_prompt.txt"
        prompt_path.write_text(meta["prompt"], encoding="utf-8")

        if schedule is None:
            return {"prompt": prompt_path}

        schedule_path = self.output_dir / f"{stem}_schedule.csv"
        schedule.to_csv(schedule_path, index=False)

        meta_path = self.output_dir / f"{stem}_meta.json"
        meta_path.write_text(
            json.dumps(
                {k: v for k, v in meta.items() if k != "prompt"},
                indent=2, default=str
            ),
            encoding="utf-8",
        )

        self.logger.info(f"LLM schedule saved: {schedule_path}")

        return {
            "prompt": prompt_path,
            "schedule": schedule_path,
            "meta": meta_path,
        }


# --------------------------------------------------

def main():
    """
    `python -m modules.fusion.llm_scheduler [--dry-run]`

    --dry-run builds the feature table and the prompt and stops,
    spending no Gemini quota. Do that first: the free tier allows
    20 requests a day across all three plants.
    """

    import argparse

    parser = argparse.ArgumentParser(description="LLM fusion + scheduling")
    parser.add_argument("--run-time", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="build the prompt only, spend no quota")

    args = parser.parse_args()

    from modules.preprocessing.windy_features import (
        WindyFeatureBuilder, load_meter_history
    )

    builder = WindyFeatureBuilder()

    try:
        meter = load_meter_history()
    except Exception as error:
        print(f"(meter history unavailable: {error})")
        meter = None

    features, info = builder.build(run_time=args.run_time, meter=meter)

    run_time = features["run_time"].iloc[0]

    satellite = info["satellite"]

    print(f"windy values : {info['values_path']}")
    print(f"satellite    : {satellite['sat_video'] if satellite else 'none'}")
    print(f"run time     : {run_time}")

    scheduler = LLMScheduler()

    schedule, meta = scheduler.decide(
        features, run_time, dry_run=args.dry_run, satellite=satellite
    )

    paths = scheduler.save(schedule, meta, run_time)

    print(f"prompt       : {meta['prompt_chars']:,} chars -> {paths['prompt']}")

    if schedule is None:
        print("\nDRY RUN - no request sent. Review the prompt, then rerun "
              "without --dry-run.")
        return 0

    print(f"regime       : {meta['regime']} (confidence {meta['confidence']})")
    print(f"reasoning    : {meta['reasoning']}")
    print()

    with pd.option_context("display.max_rows", 100, "display.width", 160):
        print(schedule[[
            "block", "time", "clearsky_power_mw", "windy_kt",
            "llm_kt", "forecast_mw"
        ]].to_string(index=False, float_format="%.3f"))

    print(f"\nenergy: {schedule['forecast_mw'].sum() * 0.25:.3f} MWh")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
