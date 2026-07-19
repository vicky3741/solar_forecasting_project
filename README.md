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
5. **Enercast** — used strictly as a validation benchmark, never as a
   model input.

The final forecast per block = blended kt × clear-sky generation curve,
clipped to plant capacity.

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
  forecasting/     clearsky.py (pvlib), chronos_model.py, predictor.py (hybrid)
  evaluation/      metrics.py, evaluator.py, backtester.py (+ tuning grid)
  orchestrator/    pipeline.py - one full forecast run end to end
  scheduler/       Auto-triggers the orchestrator at the 7 daily run times
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

## Dashboard

`python app.py` → http://localhost:8000

- **Current Final Schedule** — actual generation for completed blocks,
  forecast for blocks ahead
- **Our Prediction vs Enercast vs Actual** — pick any day/run-time; see
  per-15-min-block kW from each source, energy totals in MWh, and who
  was closer to reality per block and overall
- **Accuracy history** — average deviation per day, ours vs Enercast

## Current results (honest)

Backtested point-in-time (no lookahead) across 13 days (Jul 6–18) × 7
run-times = 91 runs:

| Metric (avg over comparable runs) | Ours | Enercast |
|---|---|---|
| Deviation (% of capacity) | 8.6% | 6.7% |
| Runs won | 24 / 84 | 60 / 84 |

Out-of-sample check: parameters tuned on week 1 (Jul 6–12) were tested
on unseen week 2 (Jul 13–18) and performed *better* there (7.8% vs
10.0% deviation) — the model generalizes rather than memorizing.

Known limitations, stated plainly:

- **13 days of history** — tuned parameters (chronos_weight 0.2,
  performance_ratio 0.80) keep being re-validated as more data arrives.
- **Windy videos exist for only 1 of 13 days** (July 9). On vision-assisted
  runs the deviation improved from 5.16% → 4.82%. Daily video capture at
  every run time is the highest-value data improvement available.
- **The scheduling-penalty metric is a placeholder** (flat rate beyond a
  15% dead band) — the real DSM regulation slabs are pending and will be
  swapped in when provided.
- Plant tilt/azimuth are assumptions (latitude-tilt, south-facing) until
  site-confirmed values arrive.

## Remaining work

- Live data connectors (auto-fetch latest meter data / Windy captures —
  currently files are dropped into `data/` manually)
- Real DSM penalty formula (awaiting regulation details)
- Live end-to-end test of the scheduler across a full day
- Deployment to cloud (Team 3 scope; the FastAPI backend is ready)
