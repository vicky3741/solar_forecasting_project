# EC2 → Lambda: the handoff

Everything the Lambda port has to reproduce, and the four things about
this pipeline that will bite if it does not.

## Six functions, nothing shared

| Function | `SOLAR_PLANT` | `SOLAR_TASK` | Memory | Timeout | Bucket it can touch |
|---|---|---|---|---|---|
| `solar-capture-sirmour` | `sirmour` | `capture` | 2048 MB | 300 s | `sirmour-team2-storage` |
| `solar-forecast-sirmour` | `sirmour` | `forecast` | 3008 MB | 600 s | `sirmour-team2-storage` |
| `solar-capture-kasipet` | `kasipet` | `capture` | 2048 MB | 300 s | `kasipet-team2-storage` |
| `solar-forecast-kasipet` | `kasipet` | `forecast` | 3008 MB | 600 s | `kasipet-team2-storage` |
| `solar-capture-bhupalpally` | `bhupalpally` | `capture` | 2048 MB | 300 s | `bhupalpally-team2-storage` |
| `solar-forecast-bhupalpally` | `bhupalpally` | `forecast` | 3008 MB | 600 s | `bhupalpally-team2-storage` |

All six run the same image and the same [handler.py](handler.py). Two
environment variables are the entire difference. Give each function an IAM
role scoped to **its own bucket only** — that is what makes the plant
separation real rather than a convention.

## The triggers, divided per plant

Run [create_schedules.sh](create_schedules.sh). It creates 12 EventBridge
schedules that produce 46 invocations a day.

| Plant | Runs/day | Capture cron (IST) | Forecast cron (IST) |
|---|---|---|---|
| Sirmour | 7 | `cron(45 6,9,12,15 ? * * *)` + `cron(15 8,11,14 ? * * *)` | `cron(50 6,9,12,15 ? * * *)` + `cron(20 8,11,14 ? * * *)` |
| Kasipet | 8 | `cron(45 6,9,12,15 ? * * *)` + `cron(15 8,11,14,17 ? * * *)` | `cron(50 6,9,12,15 ? * * *)` + `cron(20 8,11,14,17 ? * * *)` |
| Bhupalpally | 8 | `cron(45 6,9,12,15 ? * * *)` + `cron(15 8,11,14,17 ? * * *)` | `cron(50 6,9,12,15 ? * * *)` + `cron(20 8,11,14,17 ? * * *)` |

Twelve rules rather than forty-six because the official run times alternate
`:45` and `:15` every 90 minutes, so two crons cover a plant's whole day.
The single `17` in the Telangana hour lists is the **only** scheduling
difference between the plants.

Use EventBridge **Scheduler** (`aws scheduler`), not classic EventBridge
rules, so the expressions stay readable as IST. If a tool forces classic
rules, these are the same times in UTC:

| | `:45`/`:50` runs | `:15`/`:20` runs |
|---|---|---|
| Sirmour capture | `cron(15 1,4,7,10 ? * * *)` | `cron(45 2,5,8 ? * * *)` |
| Sirmour forecast | `cron(20 1,4,7,10 ? * * *)` | `cron(50 2,5,8 ? * * *)` |
| Telangana capture | `cron(15 1,4,7,10 ? * * *)` | `cron(45 2,5,8,11 ? * * *)` |
| Telangana forecast | `cron(20 1,4,7,10 ? * * *)` | `cron(50 2,5,8,11 ? * * *)` |

### Why forecast is a separate function five minutes later

Capture and forecast ran back to back in one EC2 trigger. Inside a Lambda
that puts a 300 s capture timeout plus a 600 s forecast timeout under a
900 s hard ceiling, with nothing spare — a slow day gets killed
mid-forecast. Split, a hung capture also cannot delay or cancel the
forecast.

Five minutes is safe on both sides. The orchestrator floors its run time to
the 15-minute block, so a forecast firing at 06:50 still publishes the
**06:45** schedule; and 5 minutes is well inside the 20-minute
`video_match_tolerance_minutes`, so it finds the clip that capture just
uploaded.

### What is deliberately NOT ported

`modules/scheduler/scheduler.py` does not run on Lambda. Three of its
mechanisms exist only because of the EC2 box and should not be recreated:

- **Lock ports 49732/3/4.** A guard against Windows Task Scheduler stacking
  duplicate schedulers. EventBridge does not stack.
- **`scheduler.run_offset_seconds` (0 / 5 / 10 min).** Kept three 450 MB
  forecasts from peaking together on a 900 MB t3.micro. Every Lambda
  invocation gets its own container, so all three plants fire at the same
  minute. Leave the config values alone — EC2 still reads them.
- **The capture/forecast subprocess split.** It was about handing memory
  back between runs. Separate functions do that now, and a warm container
  keeping torch loaded is a speedup, not a leak.

## Four things that will bite

**1. The filesystem.** Every path in config is relative
(`data/plants/kasipet/raw`, `models/...`, `outputs/...`) and resolves
against the working directory. On Lambda `/var/task` is read-only and only
`/tmp` is writable. `prepare_workspace()` in [handler.py](handler.py)
copies the read-only inputs into `/tmp/solar`, creates the write
directories and chdirs there. Nothing in the pipeline had to change — do
not "fix" this by rewriting paths across the codebase.

**2. Size.** torch plus the Chronos weights is ~450 MB and Playwright
brings its own Chromium. That is far past the 250 MB unzipped limit for zip
packages, so this **must** be a container image (10 GB limit). Bake the
Chronos weights into the image; if they download at cold start, every cold
start pays for it and the first forecast of the day may time out.

**3. Credentials.** `.env` is gitignored and is not in the image. The keys
it holds become function configuration:

| Variable | Notes |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | **Drop these.** Use the function's IAM role instead — boto3 picks it up automatically. |
| `GOOGLE_API_KEY` | One key **per plant**. The Gemini free tier caps requests per day per model, and 23 vision calls a day across three plants does not fit under one key. |
| `WINDY_API_KEY` | Shared; it only unlocks premium overlays, there is no quota to split. |

**4. Logs.** Each plant writes its own log file (`solar_forecasting.log`,
`kasipet.log`, `bhupalpally.log`) into a `/tmp` that does not survive the
container. Everything printed goes to CloudWatch anyway — one log group per
function — so use that and treat the files as scratch.

## Before the first run

The per-plant buckets must exist and hold each plant's history, or every
pull and push fails:

```bash
bash automation/create_plant_buckets.sh
```

Then verify one plant end to end before cutting the EC2 services off:

```bash
aws lambda invoke --function-name solar-capture-kasipet --payload '{}' out.json
```

Keep the EC2 services running until a full day of Lambda runs has produced
schedules that match. Stop them with
`sudo systemctl disable --now solar-forecast-sirmour solar-forecast-kasipet solar-forecast-bhupalpally`
— not before.
