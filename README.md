# Solar Forecasting Project — Sirmour, Kasipet, Bhupalpally

AI-based intraday solar generation forecasting for **three plants**. Each
one forecasts its own generation for the same day up to 19:00 IST at
15-minute resolution, re-running at fixed times through the day, and
continuously maintains its own **Current Final Schedule**.

| Plant | State | Capacity | Runs/day | Effective time | Since |
|---|---|---|---|---|---|
| Sirmour | Madhya Pradesh | 5.1 MW | 7 | none (see below) | project start |
| Kasipet | Telangana | 15 MW | 8 (adds 17:15) | 3 blocks / 45 min | 2026-08-08 |
| Bhupalpally | Telangana | 10 MW | 8 (adds 17:15) | 3 blocks / 45 min | 2026-08-08 |

Built by **Team 2** (Vikrant, Abhijit, Sandhyarani, Arpita) using only
free / open-source tooling — no paid APIs.

---

## Three plants, one codebase, zero shared state

Everything below describes one pipeline. It runs three times over, once
per plant, and the plants never touch each other's anything: separate
meter folders, separate outputs, separate model state, separate S3
prefixes, separate weather caches, separate logs, separate services.

The mechanism is one environment variable:

```bash
SOLAR_PLANT=kasipet python -m modules.orchestrator.pipeline
```

`config/config.py` reads it, loads `config/settings.yaml` as the base
layer, and deep-merges `config/plants/<key>.yaml` over it. Every module
already read a single `settings` object, so re-pointing that object
re-points the entire pipeline — no module in the codebase knows there is
more than one plant.

**`config/settings.yaml` is, and stays, Sirmour.** Its overlay
(`config/plants/sirmour.yaml`) is nearly empty on purpose, so with
`SOLAR_PLANT` unset the resolved configuration is byte-identical to what
the live Sirmour automation has always used. Every plant-specific knob
added for the new plants carries Sirmour's existing behaviour as its
default for the same reason. When you add a knob, follow that rule.

### Effective time (freeze horizon)

The mentor's *Effective Time Schedule Guide* (2026-08-08): a schedule
generated at 11:15 does **not** take effect at 11:15. The next few blocks
are already declared to the grid operator and stay at whatever the
previous schedule said. Sirmour's horizon is 6 blocks (90 min); the
Kasipet/Kothagudem family's is 3 blocks (45 min).

```
run at 11:15 = engine block 46
  Sirmour   freeze 46-51  ->  new schedule from block 52 (12:45)
  Kasipet   freeze 46-48  ->  new schedule from block 49 (12:00)
```

This is why the same day's final report differs between the three plants:
they revise at different speeds. It lives in
`modules/scheduling/effective_time.py` and is configured per plant as
`schedule_rules.freeze_blocks`.

> **Sirmour is currently set to 0, not 6.** Sirmour has published with no
> freeze horizon since the pipeline was built, and every tuned constant in
> `settings.yaml` was validated against that behaviour. Raising it to 6 is
> a one-line change plus a re-backtest — the machinery is built and already
> live on the other two plants. Decide, then flip it.

Details of the per-plant automation: [`automation/README.md`](automation/README.md).

---

## How the forecast works

The system is a **hybrid** of three independent signals, combined in
"clear-sky index" (kt) space:

```
Latest meter data ──► Preprocessing ──► kt history ─────┐
                                                        │
Windy cloud video ──► Frame extraction ──► Gemini ──────┤──► Hybrid blend ──► Final forecast
                                                        │      (kt space)      (15-min blocks
Plant location ─────► pvlib clear-sky physics ──────────┘                       to 19:00)
                                                                     │
                                              Enercast (validation only, never an input)
```

1. **Clear-sky physics (pvlib)** — computes the theoretical maximum
   generation for any moment from sun position + plant geometry alone.
   Needs zero historical data. This provides the exact sunrise/sunset
   shape, guaranteeing forecasts always go to zero at night.
2. **Clear-sky index (kt)** — actual measured GHI divided by the
   theoretical clear-sky GHI. This is the honest "how much are clouds
   actually costing us" number (per the project brief, cloud cover is
   never treated directly as solar loss).
3. **Chronos (amazon/chronos-bolt-small)** — Amazon's free, pretrained,
   zero-shot time-series model. It forecasts the kt series forward —
   a bounded weather signal — never raw power (it doesn't know sunsets
   exist).
4. **Gemini vision (gemini-3.5-flash, free tier)** — analyzes Windy
   cloud animations: coverage, density, movement direction/speed,
   whether clouds move toward the plant, minutes until arrival, and
   separate 2-hour / 2-4-hour irradiance trends. Applied as a
   **time-phased** per-block adjustment that ramps in at the predicted
   cloud arrival time.
5. **Open-Meteo (weather forecast)** — free, no API key. Supplies a
   *forward-looking* forecasted GHI for every future block (the one
   thing the other signals lack). Converted to a forecasted clear-sky
   index and blended in at weight 0.25 (tuned + leave-one-day-out
   validated; re-tuned down from 0.65 on 2026-08-04). This was the
   single biggest accuracy gain — it closed the gap to Enercast from
   ~2 points to ~0.17.
6. **Enercast** — used strictly as a validation benchmark, never as a
   model input.

The final forecast per block = blended kt × clear-sky generation curve,
clipped to plant capacity.

### Residual correction (LightGBM) — currently DISABLED

A tiny LightGBM model trained on the backtest record of our own past
mistakes (features: block hour, horizon, kt at run time, forecast value
— all known at prediction time).

It is **not a competitor to the weather signal** — it is a correction
layer applied *after* the forecast, whereas Open-Meteo is an input
signal feeding *into* it. The two are different components, and the
finding was that **the weather signal made the correction redundant**:

- *Before* Open-Meteo existed: +1.14 pct points on unseen days
  (leave-one-day-out, 10/13 days helped) — genuinely useful.
- *After* Open-Meteo was added: fell to −0.37 pct points (only 4/13
  days helped), so it was switched off.

The interpretation: the residual corrector had been a crutch for the
missing forward-looking signal. Once real forecasted weather supplied
that information properly, re-correcting for it only added noise.

A stricter walk-forward test (each day corrected only by past days —
`tests/test_walkforward_experiment.py`) separately showed the
correction *hurts* until ~7-8 days of mistake history exist. Re-evaluate
by rerunning `tests/test_residual_experiment.py` if the base model
changes; re-enable only if it helps out-of-sample again.

### Block bias correction (time-of-day shape) — ACTIVE

Added 2026-08-06 on mentor guidance: *"analyze the pattern of the
results for the last 4-5 days … identify the pattern among the blocks
causing higher penalties & convey the same to the model."*

`tests/test_block_penalty_pattern.py` is the analysis. On the last 5
finished days it found:

- **100% of the DSM penalty falls in blocks 40–67 (09:45–16:30).**
  Everywhere else the plant is small enough that even a total miss
  stays inside the free ±0.51 MW (10% of capacity) dead band — those
  blocks cannot cost money, so accuracy there is not worth buying.
- **Half the penalty comes from 7 blocks out of 49.**
- Inside the paying window the schedule carries a repeating
  **time-of-day shape**: the late morning is over-forecast, the
  mid-afternoon under-forecast by ~0.34 MW on 4 of the last 5 days.

`modules/forecasting/block_bias_correction.py` feeds that back: each
block is shifted by **half** the median (actual − scheduled) that block
showed over the last 5 finished days, smoothed over ±6 blocks (±90 min).

Validated recursively walk-forward over 12 days
(`tests/test_block_bias_experiment.py`) — training days are themselves
corrected, which is what deployment actually looks like:

| | |
|---|---|
| Penalty before | Rs 990 / day |
| Penalty after | Rs 950 / day (**−4.0%**) |
| Unseen days cheaper | 5 / 7 |
| Deviation | −0.10 pts |

The **control** matters as much as the result: a flat whole-day shift
(same data, no block shape) made the penalty *worse* at every setting.
The money really is in the time-of-day pattern, not the overall level.
Two other negative results are baked into the settings — unsmoothed
per-block medians hurt (5 days is 5 noisy samples per block), and full
strength hurts (same lesson as the case-based corrector).

The profile is **rebuilt at run time** from the recent day schedules and
refuses to run if the newest is more than 3 days old, so it cannot go
stale the way a saved-once model can.

## Project structure

```
config/            settings.yaml  = the BASE layer, and it is Sirmour
  plants/          sirmour.yaml (nearly empty), kasipet.yaml, bhupalpally.yaml
                   -> deep-merged over settings.yaml, chosen by SOLAR_PLANT
data/                              --- Sirmour ---
  historical/      Daily meter CSVs (15-min: power, GHI, POA, weather)
  enercast/        Enercast schedule CSVs (validation reference; Sirmour only)
  windy/videos/    Windy cloud animation captures (.webm)
  processed/       Combined preprocessed dataset
  plants/          --- the other plants, same layout one level down ---
    kasipet/       historical/ processed/ windy/ weather/ ...
    bhupalpally/   historical/ processed/ windy/ weather/ ...
modules/
  preprocessing/   Validation, outlier clipping, 15-min alignment, features
                   (each plant's vendor schema declared in config data_schema)
  vision/          Frame extraction -> Gemini -> JSON parsing (cached per video)
  fusion/          Vision features -> per-block forecast adjustment profile
  forecasting/     clearsky.py (pvlib), chronos_model.py, predictor.py (hybrid),
                   residual_correction.py (LightGBM error-learning),
                   case_based_correction.py (kNN analogues),
                   block_bias_correction.py (time-of-day penalty shape)
  scheduling/      effective_time.py - the per-plant freeze horizon
  evaluation/      metrics.py, evaluator.py, backtester.py (+ tuning grid)
  orchestrator/    pipeline.py - one full forecast run end to end
  scheduler/       Auto-triggers the orchestrator at that plant's run times
automation/        One launcher + one systemd unit per plant, and the notes
                   on lock ports, staggering and EC2 memory
models/            Sirmour's learned state; models/plants/<key>/ for the rest
utils/             Logger (loguru) + file I/O and per-plant path helpers
tests/             Runnable smoke tests / backtest for every module, plus
                   ingest_plant_history.py and backfill_day_schedules.py
static/            Dashboard frontend (Chart.js)
app.py             FastAPI web app serving the dashboard + JSON API
outputs/           Sirmour's generated files; outputs/plants/<key>/ for the
                   rest (not in git)
```

Note: the empty `__init__.py` files are **required** — they mark folders
as Python packages so imports work. Do not delete them.

## Setup (one time)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file in the project root containing:
GOOGLE_API_KEY=your_gemini_api_key_here
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Optional - a separate Windy key per plant, so one plant's exhausted
# quota cannot take the other two down. Falls back to WINDY_API_KEY,
# and no key at all still works (free public embed).
WINDY_API_KEY_SIRMOUR=...
WINDY_API_KEY_KASIPET=...
WINDY_API_KEY_BHUPALPALLY=...
```

## Running it

All commands from the project root with the venv activated. **Every one of
them takes `SOLAR_PLANT`**; leave it unset and you get Sirmour.

```bash
# Refresh the processed dataset from this plant's historical folder
python -m tests.test_preprocessing

# Run one full forecast cycle (produces the Current Final Schedule)
python -m tests.test_orchestrator

# Run the full backtest + parameter tuning (feeds the dashboard)
python -m tests.test_backtest

# Start the web dashboard, then open http://localhost:8000
python app.py
```

For one of the other plants (PowerShell: `$env:SOLAR_PLANT="kasipet"`):

```bash
SOLAR_PLANT=kasipet python -m tests.test_preprocessing
SOLAR_PLANT=kasipet python -m modules.orchestrator.pipeline
```

The scheduler (`modules/scheduler/scheduler.py`) runs the orchestrator
automatically at that plant's official times — 06:45, 08:15, 09:45, 11:15,
12:45, 14:15, 15:45 IST, plus 17:15 for the two Telangana plants. One
scheduler process per plant; see [`automation/`](automation/README.md).

## Onboarding a new plant

Everything a fourth plant needs is data plus one config file.

```bash
# 1. config/plants/<key>.yaml - copy kasipet.yaml and change the
#    coordinates, capacity, paths, prefixes, lock port and log name.

# 2. Load its meter history. The files are validated against that
#    plant's declared schema first, so a wrong column name or date
#    order is caught here rather than becoming a silently wrong
#    forecast later.
SOLAR_PLANT=<key> python -m tests.ingest_plant_history "/path/to/vendor/folder" --upload

# 3. Reconstruct every day it has data for, oldest first. Until this
#    runs, the block-bias corrector has nothing to learn from and
#    stays switched off.
SOLAR_PLANT=<key> python -m tests.backfill_day_schedules --quiet

# 4. Build its processed dataset, backtest, and case store.
SOLAR_PLANT=<key> python -m tests.test_preprocessing
SOLAR_PLANT=<key> python -m tests.test_backtest
SOLAR_PLANT=<key> python -m tests.test_case_based_experiment

# 5. Add a launcher in automation/ and start it.
```

Step 4's backtest also prints the tuning grid for `chronos_weight` and
`clearsky.performance_ratio`. The values a new plant inherits from
`settings.yaml` were **tuned on Sirmour** and are a documented starting
point, not validated numbers for that site — re-tune them once the plant
has a comparable run of days.

## Building a day's report (start here)

`tests/` holds three kinds of file, and only the first kind is what you
want day to day:

**1. The report pipeline.** Run these two, in this order, for any finished
day. The first rebuilds the schedule, the second turns it into the
workbook that goes to the mentor:

```bash
# Step 1 - reconstruct the day's schedule (needs this plant's meter data
# and the day's Windy clips, local or in S3)
python -m tests.generate_schedule_for_day 2026-07-31

# Step 2 - the mentor-facing workbook: per-block MW, DSM penalty,
# how many blocks fell outside the deviation band, Enercast alongside
python -m tests.build_penalty_report 2026-07-31
```

Both steps are per plant, and the three plants write to three separate
folders — so building all three days is the same two commands run three
times:

```bash
for p in sirmour kasipet bhupalpally; do
  SOLAR_PLANT=$p python -m tests.generate_schedule_for_day 2026-08-06
  SOLAR_PLANT=$p python -m tests.build_penalty_report    2026-08-06
done
```

`build_penalty_report.py` writes formulas, not baked-in numbers, so the
saved file shows blanks until Excel (or LibreOffice) opens and calculates
it once. Open it and save, and the values are cached from then on.

`tests/build_simple_schedule.py` is the plainer variant of step 2 - same
data, no penalty columns - when a bare scheduled-vs-actual sheet is all
that is wanted.

Add a day's Enercast file as `data/enercast/Sirmour_<D>july_enercast.csv`
(columns `Block, Time, Scheduled MW, ...`) and both reports pick it up
automatically; leave it out and they simply omit the Enercast columns.
Enercast is a Sirmour-only feed — the Telangana plants have
`enercast.enabled: false`, so their reports carry no Enercast columns at
all rather than an empty one that reads like a data outage.

**2. Real unit/smoke tests** - `test_preprocessing.py`, `test_predictor.py`,
`test_evaluator.py`, `test_fusion.py`, `test_clearsky.py`,
`test_orchestrator.py`, `test_s3.py`, `test_vision.py`,
`test_block_bias.py`. Run any of them to check a module still works.

**3. Tuning experiments** - `test_weather_bias_experiment.py`,
`test_case_based_experiment.py`, `test_residual_experiment.py`,
`test_walkforward_experiment.py`, `test_block_bias_experiment.py` and
friends. These are the evidence behind the tuned numbers in
`config/settings.yaml`; each setting's comment names the experiment that
produced it. You do not run these to make a report - only to re-validate
a setting after changing the model.

**4. Pattern analysis** - `test_block_penalty_pattern.py`. Reads the last
N finished day schedules and reports which blocks carry the DSM penalty,
whether each block misses in a repeatable direction, and how consistent
that is day to day. Run it after a few new days land to see whether the
pattern the model is being taught still holds:

```bash
python -m tests.test_block_penalty_pattern 5
```

**5. Multi-plant tooling** — `ingest_plant_history.py` (load a plant's
meter CSVs, validated against its own schema, optionally into S3),
`backfill_day_schedules.py` (reconstruct every available day, oldest
first), `tune_plant_clearsky.py` (calibrate that site's clear-sky
performance ratio over a range wide enough to find a real optimum), and
`test_effective_time.py` (pins the freeze horizon to the mentor's guide —
both of its tables, verbatim).

**Every tuned constant lives in `config/settings.yaml`** (Sirmour) or in
`config/plants/<key>.yaml` (the others), with a comment saying what tuned
it and when. Change behaviour there rather than in the modules.

## Dashboard

`python app.py` → http://localhost:8000

- **Current Final Schedule** — actual generation for completed blocks,
  forecast for blocks ahead
- **Our Prediction vs Enercast vs Actual** — pick any day/run-time; see
  per-15-min-block kW from each source, energy totals in MWh, and who
  was closer to reality per block and overall
- **Accuracy history** — average deviation per day, ours vs Enercast

## Current results (honest)

### Sirmour

Backtested point-in-time (no lookahead) across 36 days (Jul 1 – Aug 5) ×
7 run-times = 252 runs, WITH the Open-Meteo weather signal (refreshed
2026-08-06):

| Metric (avg over the 147 runs Enercast also covers) | Ours | Enercast |
|---|---|---|
| Deviation (% of capacity) | 8.11% | 7.89% |
| Runs won | 80 / 147 | 67 / 147 |

Read that honestly: we now **win more runs** than Enercast but sit
slightly worse on average deviation — i.e. we are ahead more often, and
behind by more when we are behind. Our bad days are worse than theirs.

These numbers are worse than the 6.85% / 6.68% this table showed on 13
days (Jul 6–18). Nothing regressed — the window grew to include the
monsoon stretch from Jul 19 on, which is genuinely harder to forecast.
Comparing a 36-day number against a 13-day one is comparing two different
questions.

The backtest measures the **base hybrid only**. It does not apply the
case-based or block bias corrections, so it is a floor, not the shipped
schedule's accuracy. For that, read the daily reports in
`outputs/reports/`.

### Kasipet and Bhupalpally (first pass, 2026-08-08)

Same backtest, over their own 37 days (Jul 1 – Aug 6) × 8 run-times =
296 runs each, after calibrating each site's clear-sky performance ratio:

| | Kasipet (15 MW) | Bhupalpally (10 MW) |
|---|---|---|
| Deviation (% of capacity) | 7.43% | 7.40% |
| MAE | 1.11 MW | 0.74 MW |
| Calibrated performance ratio | 0.975 (from 0.80) | 1.00 (from 0.80) |
| Gain from that calibration | +0.77 pts | +0.75 pts |
| Case-based correction, leave-one-day-out | +0.24 pts, 26/37 days | +0.53 pts, 28/37 days |

Read these honestly too:

- **No vision at all yet.** Neither plant has a single Windy clip — their
  capture starts when their schedulers do. Every number above is the
  no-vision floor.
- **Everything except the performance ratio is Sirmour-tuned.** The blend
  weights, weather weight and correction strengths were inherited and are
  a starting point, not validated for these sites.
- **The performance ratio was the big one.** Inheriting Sirmour's 0.80 set
  the forecast's ceiling too low and under-forecast every clear block:
  the first reconstructions came out 6% and 15% short on daily energy.
  Watch for that signature on any new plant, and note that the default
  grid in `test_backtest.py` stops at 0.85 and would have reported the
  edge of its own range as the answer.

Known limitations, stated plainly:

- **36 days of history** — tuned parameters (chronos_weight 0.2,
  performance_ratio 0.80) keep being re-validated as more data arrives.
  The 2026-08-06 tuning grid preferred chronos_weight 0.6 by 0.10 pts
  (8.35% vs 8.45%) — inside the noise on this much data, so nothing was
  changed on it. Re-check when the gap is decisive.
- **The backtest sees almost no vision** — only 4 of its 252 runs have a
  cloud clip, because `backtester.py` searches the old hand-recorded
  `data/windy/videos` folder only. Live runs and the daily reports do far
  better (6–7 clips a day from S3 since the EC2 capture went live). So the
  backtest is a no-vision floor; do not read it as the vision result.
- **The backtest's penalty column is still the placeholder** (flat rate
  beyond a 15% dead band). The real DSM slabs (0–10% free, then 0.50 /
  0.75 / 1.00 Rs per kWh) are used by the daily penalty reports and by
  `tests/test_block_bias_experiment.py` — those are the rupee numbers to
  quote.
- Plant tilt/azimuth are assumptions (latitude-tilt, south-facing) until
  site-confirmed values arrive.

## Keeping the learned parts fresh

Two components learn from recent days, and they go stale in different
ways. This is the maintenance the model actually needs:

| Component | Refreshes | If it goes stale |
|---|---|---|
| **Block bias profile** | By itself, every run, from the last 5 day schedules | Refuses to run past 3 days old — safe by default |
| **Case store** (`models/case_store.csv`) | Only when you re-run the two commands below | Keeps applying old analogues; the pipeline logs a warning past 7 days |

The case store rotted exactly this way once: built 2026-07-24 on Jul 6–22,
still nudging live forecasts a fortnight later with nothing saying so.
Refresh it after every few new days of meter data:

```bash
python -m tests.test_preprocessing
```

```bash
python -m tests.test_backtest
```

```bash
python -m tests.test_case_based_experiment
```

The last one re-runs the leave-one-day-out verdict and **only saves the
store if case-based reasoning still helps** — so a refresh can also
legitimately come back saying "switch this off".

## Remaining work

- Real DSM penalty formula in the backtest (the daily reports already use
  the real slabs)
- Live end-to-end test of the scheduler across a full day
- Backtester should read the S3/auto-captured clips, not just the old
  `data/windy/videos` folder, so vision is actually measured
