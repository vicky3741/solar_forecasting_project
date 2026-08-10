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
    "block", "time", "clearsky_power_mw", "anchor_kt",
    "weather_kt", "windy_kt",
    "windy_clouds_pct", "windy_lclouds_pct", "windy_mclouds_pct",
    "windy_hclouds_pct", "windy_rain_mm", "windy_is_measured",
]


class LLMScheduler:

    def __init__(self):

        self.logger = get_logger()

        self.parser = JSONParser()

        self.validator = ScheduleValidator()

        from modules.fusion.case_retrieval import CaseRetriever

        self.cases = CaseRetriever()

        plant = settings["plant"]

        self.plant_name = plant.get("name", "the plant")
        self.plant_tag = plant.get("code", "SIRMOUR")
        self.capacity_mw = plant["capacity_mw"]
        self.timezone = plant["timezone"]

        self.interval_minutes = settings["forecast"]["interval_minutes"]

        fusion = settings.get("fusion", {})

        self.output_mode = fusion.get("llm_output", "kt")
        self.max_kt = fusion.get("max_clear_sky_index", 1.2)

        # How much of the published number is the model's, the rest
        # being the physics anchor. 1.0 = the model alone, which is what
        # every run before 2026-08-09 did.
        self.blend_weight = float(fusion.get("blend_weight", 1.0))
        self.temperature = fusion.get("temperature", 0.1)
        self.max_output_tokens = fusion.get("max_output_tokens", 8192)

        self.freeze_blocks = settings.get("schedule_rules", {}).get(
            "freeze_blocks", 0
        )

        self.output_dir = Path(fusion.get("output_dir", "outputs/llm_schedules"))

    # --------------------------------------------------

    def anchor_mw(self, features, ahead):
        """
        The physics baseline each block is measured against.

        Windy's own forecast clear-sky index where it exists, because
        that is a real forward-looking number. Where it does not, the
        last MEASURED cloudiness from today's meter, decayed toward the
        day's average as the horizon grows.

        THE FALLBACK IS NOT CLEAR SKY, AND THAT MATTERS. An earlier
        version defaulted to kt = 1.0 whenever Windy was missing, which
        makes the anchor a perfectly sunny day. On 2026-07-27 - an
        overcast day with no scraped Windy values - the model correctly
        forecast about half of clear sky, and the validator then pulled
        34 of 35 blocks back UP toward sunshine because they sat more
        than 40% from that anchor. The guard rail was making the
        forecast worse.

        Damped persistence is the right fallback on the evidence: it
        scored 9.80% in the 2026-08-09 walk-forward, beating every
        model built on the clips. An anchor should be something we
        would publish on its own, and that one is.
        """

        clearsky = ahead["clearsky_power_mw"].to_numpy(dtype=float)

        # THE ANCHOR IS DAMPED PERSISTENCE, AND NOTHING ELSE BY DEFAULT.
        #
        # A forecast has two possible jobs here: information the model
        # reads, and the reference the validator enforces. Letting one
        # signal do BOTH means a wrong forecast is applied twice - the
        # model is told to believe it, and then bounded toward it when
        # it does not.
        #
        # 2026-07-27 is what that costs. ECMWF said sunny; the day was
        # overcast. With weather as anchor AND prompt column the model
        # published 16.14 MWh against 8.95 actual - over on 31 of 35
        # blocks, Rs 3,456. Damped persistence on the same day: 8.88
        # against 8.95, Rs 129. The plainest signal was right within 1%
        # while both forecast-led versions were wrong by 35-80%.
        #
        # So the anchor is the thing that has earned it. Forecasts stay
        # in the prompt, where the model may weigh them and say why.
        kt = np.full(len(ahead), np.nan)

        source = settings.get("fusion", {}).get("anchor_source", "persistence")

        if source in ("windy", "forecast") and "windy_kt" in ahead.columns:
            kt = ahead["windy_kt"].to_numpy(dtype=float).copy()

        if source == "forecast" and "weather_kt" in ahead.columns:
            weather = ahead["weather_kt"].to_numpy(dtype=float)
            kt = np.where(np.isfinite(kt), kt, weather)

        measured = pd.Series(dtype=float)

        if "actual_kt" in features.columns:
            measured = features.loc[
                features["is_past"] == 1, "actual_kt"
            ].dropna()

        if measured.empty:
            fallback = np.full(len(ahead), 1.0)

        else:
            kt_now = float(measured.tail(4).mean())
            kt_day = float(measured.mean())

            horizon = (
                (ahead["timestamp"] - ahead["timestamp"].iloc[0])
                .dt.total_seconds().to_numpy() / 60.0
            )

            weight = np.exp(-horizon / 120.0)

            fallback = kt_now * weight + kt_day * (1 - weight)

        kt = np.where(np.isfinite(kt), kt, fallback)

        return np.clip(
            np.clip(kt, 0.0, self.max_kt) * clearsky, 0.0, self.capacity_mw
        )

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
        ].copy()

        if ahead.empty:
            return "", ahead

        # The anchor has to be IN the table, because the model is now
        # asked to adjust it rather than to invent a number. Shown as a
        # clear-sky index so it is on the same scale as everything else
        # it is being compared against.
        anchor = self.anchor_mw(features, ahead)
        clearsky = ahead["clearsky_power_mw"].to_numpy(dtype=float)

        ahead["anchor_kt"] = np.divide(
            anchor, clearsky,
            out=np.full(len(ahead), np.nan),
            where=clearsky > 0.01,
        )

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

    def precedent_section(self, features, ahead, run_time):
        """
        What actually happened the last several times conditions looked
        like this (modules/fusion/case_retrieval.py).

        Sampled at three horizons rather than all of them - drift from
        forecast to outcome depends mostly on how far ahead the call
        was made, and three points show that shape without spending
        tokens on thirty near-identical summaries.
        """

        if not self.cases.available:
            return "(no past cases available)"

        past = features[
            (features["is_past"] == 1) & features["actual_kt"].notna()
        ]

        kt_now = (
            float(past["actual_kt"].tail(4).mean()) if not past.empty
            else float(pd.Series(ahead["windy_kt"]).dropna().head(1).mean())
            if ahead["windy_kt"].notna().any() else 1.0
        )

        minutes = (
            (ahead["timestamp"] - run_time).dt.total_seconds() / 60
        ).to_numpy()

        horizons = {}

        for target in (30, 90, 180):

            index = int(np.argmin(np.abs(minutes - target)))

            # Skip a horizon the remaining day cannot reach - late runs
            # have no block 3 hours out, and pretending otherwise would
            # quote precedent for a block that does not exist.
            if abs(minutes[index] - target) > 60:
                continue

            horizons[int(round(minutes[index]))] = float(
                ahead["clearsky_power_mw"].iloc[index] * kt_now * 1000
            )

        if not horizons:
            return "(no past cases matched this situation)"

        return self.cases.prompt_section(
            kt_now, horizons, exclude_date=pd.Timestamp(run_time).date()
        )

    # --------------------------------------------------

    def track_record_section(self, run_time):
        """
        How each input has actually been performing lately
        (modules/evaluation/input_skill.py).

        This is what "let the model decide which input to trust" has to
        mean if it is to mean anything. Asking the model which input it
        relied on produces a confident answer and no information - it
        cannot see last Tuesday, so it would be reporting a feeling.
        Handing it measured error per input makes the judgement an
        inference from evidence.
        """

        try:
            from modules.evaluation.input_skill import InputSkill

            section = InputSkill().prompt_section(run_time)

        except Exception as error:
            self.logger.warning(f"Input track record unavailable ({error})")
            return None

        return section

    # --------------------------------------------------

    def build_prompt(self, features, run_time, ahead, satellite=None):

        table, _ = self.forecast_table(features, run_time)

        history = self.recent_history(features)

        sky = self.satellite_section(satellite, run_time)

        precedent = self.precedent_section(features, ahead, run_time)

        track_record = self.track_record_section(run_time)

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

WHAT HAPPENED IN SIMILAR SITUATIONS BEFORE
{precedent}
{("WHICH INPUTS HAVE BEEN RIGHT LATELY" + chr(10) + track_record + chr(10))
 if track_record else ""}

BLOCKS STILL TO SCHEDULE ({count} blocks, {first_block} to {last_block})
{table}

HOW TO READ THE COLUMNS
- clearsky_power_mw: what this plant produces under a perfectly clear sky at \
that moment. Exact physics from pvlib - solar geometry, tilt and capacity. \
Treat it as ground truth for the SHAPE of the day.
- anchor_kt: THE NUMBER YOU ARE ADJUSTING. It is today's last measured \
cloudiness, decaying toward the day's average as the horizon grows. Over the \
last twelve days this anchor alone was the most accurate thing available, so \
departing from it needs a reason you can name.
- weather_kt: the ECMWF weather forecast for this location, as a fraction of \
clear sky, already corrected for its recent measured bias at this plant. This \
is the ONLY input that knows about weather still to come, so it should carry \
most of the weight for blocks several hours out, where the satellite picture \
and today's meter readings say little.
- windy_kt: Windy's own ECMWF solar-power forecast, same fraction. It is the \
same underlying model as weather_kt, so the two agreeing is NOT independent \
confirmation - but the two disagreeing means one of them was sampled or \
interpolated badly, and the disagreement itself is a warning.
- windy_clouds_pct / lclouds / mclouds / hclouds: total, low, middle and high \
cloud cover. Low thick cloud attenuates far more than thin high cloud, so the \
split matters more than the total.
- windy_rain_mm: forecast rainfall. Rain implies heavy cloud.
- windy_is_measured: 1 means Windy had a REAL reading at that block. 0 means the \
value was interpolated between readings up to 90 minutes apart. Free-tier Windy \
only updates every 3 hours, so today only {measured} of the remaining blocks \
carry a real reading. Trust the 1s; treat the 0s as a smooth guess between them.

YOUR JOB
Each block above already has an anchor - the physics baseline, listed as \
anchor_kt. Your job is to ADJUST each anchor using the evidence above, NOT to \
invent a new number independently of it. Start from the anchor and move it only \
where something in the evidence justifies moving it.

Weigh what the plant has actually been doing today against what the forecasts \
say. Where they disagree, say which you trusted and why. Cloud fields move, so \
a change one reading shows usually arrives gradually across the blocks around \
it rather than instantly.

Being wrong is penalised in both directions - over-forecasting costs the plant \
money in deviation charges just as under-forecasting does. Do not pad the \
number for safety.

Give each block its own confidence, and be honest with it. "high" means the \
evidence genuinely points one way; "low" means you are guessing, and a low \
confidence block will be pulled back toward the anchor rather than published as \
you wrote it. Marking everything high does not make your numbers count more, it \
only removes the safety net where you needed it.

RESPOND WITH JSON ONLY, no prose outside it, in exactly this form:
{{
  "regime": "clear" | "partly_cloudy" | "overcast" | "storm",
  "reasoning": "2-4 sentences on what drove the decision",
  "blocks": [
    [{first_block}, 0.00, "high"],
    [{first_block + 1}, 0.00, "medium"],
    ...
  ]
}}

"blocks" must cover every one of the {count} blocks from {first_block} to \
{last_block}, in order, with no gaps. Each entry is \
[block_number, {wanted}, confidence] where confidence is "high", "medium" or \
"low". Any block you omit will be published at its anchor value.\
"""

    # --------------------------------------------------

    # How much of the model's number survives, per confidence level.
    # Adopted from Kushal's Windy-Project-3, which returns High/Medium/
    # Low per block; ours turns that into a blend weight rather than
    # only a label, so a block the model is unsure about is genuinely
    # pulled back toward the anchor instead of merely being annotated.
    CONFIDENCE_WEIGHT = {"high": 0.8, "medium": 0.5, "low": 0.2}

    def parse_blocks(self, payload, ahead):
        """
        The model's block list -> (values, per-block weights).

        A missing block returns NaN and takes the ANCHOR, matching
        their _parse_llm_response, which falls back per block rather
        than per response. The previous behaviour interpolated missing
        blocks from the ones that arrived, which quietly invented model
        opinions the model never expressed.
        """

        values = {}
        weights = {}

        default_weight = self.CONFIDENCE_WEIGHT["medium"]

        for entry in payload.get("blocks", []):

            confidence = None

            if isinstance(entry, dict):
                number, value = entry.get("block"), entry.get("value")
                if value is None:
                    value = entry.get("kt", entry.get("power_mw"))
                confidence = entry.get("confidence")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                number, value = entry[0], entry[1]
                if len(entry) >= 3:
                    confidence = entry[2]
            else:
                continue

            try:
                block = int(number)
                values[block] = float(value)
            except (TypeError, ValueError):
                continue

            weights[block] = self.CONFIDENCE_WEIGHT.get(
                str(confidence).strip().lower(), default_weight
            )

        blocks = ahead["block"].to_numpy()

        if not any(b in values for b in blocks):
            raise ValueError("model returned no usable blocks")

        missing = [b for b in blocks if b not in values]

        if missing:
            self.logger.warning(
                f"Model omitted {len(missing)} of {len(blocks)} block(s) "
                f"({missing[:6]}{'...' if len(missing) > 6 else ''}) "
                "- those will publish at the anchor"
            )

        raw = np.array(
            [values.get(b, np.nan) for b in blocks], dtype=float
        )
        weight = np.array(
            [weights.get(b, 0.0) for b in blocks], dtype=float
        )

        return raw, weight

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

        raw, confidence_weight = self.parse_blocks(payload, ahead)

        kt, power = self.to_power(raw, ahead)

        anchor = self.anchor_mw(features, ahead)

        # A block the model did not answer takes the anchor outright.
        missing = ~np.isfinite(power)
        power = np.where(missing, anchor, power)

        # BLEND WITH THE ANCHOR, DO NOT MERELY BOUND BY IT.
        #
        # Bounding lets the model publish anything inside a wide band
        # and gives the anchor no say within it. Blending gives the
        # anchor weight on EVERY block, which is how the production
        # pipeline treats every signal it has - Chronos at 0.2, weather
        # at 0.25 - and that pipeline is the one at 6.6%.
        #
        # The evidence for doing it here: on 2026-07-27 the model swung
        # from +80% (with weather) to -35% (without) while damped
        # persistence sat within 1% of truth all day. A signal that
        # unstable should move the published number, not be it.
        # PER-BLOCK weight, not one number for the whole run. The model
        # states its own confidence per block (their idea), and that
        # becomes how much of its number survives: high 0.8, medium 0.5,
        # low 0.2, omitted 0. fusion.blend_weight scales the whole thing,
        # so 0 still means "publish the anchor" and 1.0 means "let the
        # stated confidence decide alone".
        weight = np.clip(confidence_weight * self.blend_weight * 2.0, 0.0, 1.0)

        weight = np.where(missing, 0.0, weight)

        power = weight * power + (1.0 - weight) * anchor

        schedule = pd.DataFrame({
            "block": ahead["block"].to_numpy(),
            "timestamp": ahead["timestamp"].to_numpy(),
            "time": ahead["time"].to_numpy(),
            "clearsky_power_mw": ahead["clearsky_power_mw"].to_numpy(),
            "windy_kt": ahead["windy_kt"].to_numpy(),
            "anchor_mw": anchor,
            # Saved so modules/evaluation/input_skill.py can score each
            # input separately later, without re-running anything.
            "weather_kt": (
                ahead["weather_kt"].to_numpy()
                if "weather_kt" in ahead.columns else np.nan
            ),
            "llm_raw_mw": np.clip(
                kt * ahead["clearsky_power_mw"].to_numpy(dtype=float),
                0.0, self.capacity_mw,
            ) if self.output_mode == "kt" else np.clip(raw, 0, self.capacity_mw),
            "llm_kt": kt,
            "llm_weight": weight,
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
