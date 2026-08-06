# Solar Forecasting Project — Sirmour Solar Plant

AI-based intraday solar generation forecasting for the Sirmour Solar Plant
(5.1 MW, Rajasthan, India). The system forecasts plant generation for the
same day up to 19:00 IST at 15-minute resolution, re-running at 7 fixed
times daily, and continuously maintains a single **Current Final Schedule**.

Built by **Team 2** (Vikrant, Abhijit, Sandhyarani, Arpita) using only
free / open-source tooling — no paid APIs.

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
config/            settings.yaml (plant, models, tuned weights) + loader
data/
  historical/      Daily meter CSVs (15-min: power, GHI, POA, weather)
  enercast/        Enercast schedule CSVs (validation reference)
  windy/videos/    Windy cloud animation captures (.webm)
  processed/       Combined preprocessed dataset
modules/
  preprocessing/   Validation, outlier clipping, 15-min alignment, features
  vision/          Frame extraction -> Gemini -> JSON parsing (cached per video)
  fusion/          Vision features -> per-block forecast adjustment profile
  forecasting/     clearsky.py (pvlib), chronos_model.py, predictor.py (hybrid),
                   residual_correction.py (LightGBM error-learning),
                   case_based_correction.py (kNN analogues),
                   block_bias_correction.py (time-of-day penalty shape)
  evaluation/      metrics.py, evaluator.py, backtester.py (+ tuning grid)
  orchestrator/    pipeline.py - one full forecast run end to end
  scheduler/       Auto-triggers the orchestrator at the 7 daily run times
models/            Trained residual-correction model (regenerate via
                   tests/test_residual_experiment.py)
utils/             Logger (loguru) + file I/O helpers
tests/             Runnable smoke tests / backtest for every module
static/            Dashboard frontend (Chart.js)
app.py             FastAPI web app serving the dashboard + JSON API
outputs/           Generated: forecasts, schedules, reports, frames (not in git)
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
```

## Running it

All commands from the project root with the venv activated.

```bash
# Refresh the processed dataset from data/historical/
python -m tests.test_preprocessing

# Run one full forecast cycle (produces the Current Final Schedule)
python -m tests.test_orchestrator

# Run the full 49-run backtest + parameter tuning (feeds the dashboard)
python -m tests.test_backtest

# Start the web dashboard, then open http://localhost:8000
python app.py
```

The scheduler (`modules/scheduler/scheduler.py`) can run the orchestrator
automatically at the 7 official times: 06:45, 08:15, 09:45, 11:15, 12:45,
14:15, 15:45 IST.

## Building a day's report (start here)

`tests/` holds three kinds of file, and only the first kind is what you
want day to day:

**1. The report pipeline.** Run these two, in this order, for any finished
day. The first rebuilds the schedule, the second turns it into the
workbook that goes to the mentor:

```bash
# Step 1 - reconstruct the day's schedule (needs meter data in
# data/historical/ and the day's Windy clips, local or in S3)
python -m tests.generate_schedule_for_day 2026-07-31

# Step 2 - the mentor-facing workbook: per-block MW, DSM penalty,
# how many blocks fell outside the deviation band, Enercast alongside
python -m tests.build_penalty_report 2026-07-31
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

**Every tuned constant lives in `config/settings.yaml`**, with a comment
saying what tuned it and when. Change behaviour there rather than in the
modules.

## Dashboard

`python app.py` → http://localhost:8000

- **Current Final Schedule** — actual generation for completed blocks,
  forecast for blocks ahead
- **Our Prediction vs Enercast vs Actual** — pick any day/run-time; see
  per-15-min-block kW from each source, energy totals in MWh, and who
  was closer to reality per block and overall
- **Accuracy history** — average deviation per day, ours vs Enercast

## Current results (honest)

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
