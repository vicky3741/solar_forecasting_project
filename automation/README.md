# Automation - one independent service per plant

Three plants, three schedulers, one codebase. Each launcher does exactly
one thing that matters: it sets `SOLAR_PLANT` before Python starts. From
there `config/config.py` resolves that plant's overlay and the whole
process - data folders, S3 **bucket** and prefixes, run times, freeze
horizon, log file, lock port - is that plant's and nothing else's.

| Plant | `SOLAR_PLANT` | Runs/day | Freeze | S3 bucket | Lock port | Stagger | Log |
|---|---|---|---|---|---|---|---|
| Sirmour (MP, 5.1 MW) | `sirmour` | 7 | 6 blocks | `sirmour-team2-storage` | 49732 | 0 | `logs/solar_forecasting.log` |
| Kasipet (TG, 15 MW) | `kasipet` | 8 | 3 blocks | `kasipet-team2-storage` | 49733 | +5 min | `logs/kasipet.log` |
| Bhupalpally (TG, 10 MW) | `bhupalpally` | 8 | 3 blocks | `bhupalpally-team2-storage` | 49734 | +10 min | `logs/bhupalpally.log` |

They are independent on purpose: a crashed capture, an exhausted API
quota or a hung forecast on one plant stops that plant only. The lock
ports differ for the same reason - the single-instance guard binds a
fixed port, so three services sharing one port would mean only the first
to start ever runs.

## Moving to Lambda

The EC2/systemd setup below is what runs today. The migration to Lambda -
six functions, twelve EventBridge schedules, and the four things that will
bite - is in [lambda/README.md](lambda/README.md).

## One bucket per plant

Until 2026-08-09 all three plants lived in `sirmour-team2-storage` and were
kept apart by prefix alone. All three captures fire at the same official
run time, so that was one mistyped prefix away from three processes writing
over each other. Each plant now has its own bucket in the same AWS account
and region, with the same prefixes underneath - so every object kept the
key it already had.

Create the two new buckets and copy the Telangana history into them
**before** deploying this config, or Kasipet and Bhupalpally will point at
buckets that do not exist:

```bash
bash automation/create_plant_buckets.sh
```

The script never deletes. The old copies stay under
`sirmour-team2-storage` until you remove them by hand.

## Sirmour's freeze horizon changed

Sirmour ran at `freeze_blocks: 0` until 2026-08-09, when the mentor
confirmed it is on the guide's 6-block / 90-minute effective time. It now
matches the guide like the other two plants.

This makes the reported number worse and that is expected - measured over
20 days, deviation 4.190% -> 4.654% and penalty Rs 178 -> Rs 263 a day. The
freeze withholds fresh meter data from blocks that are already declared, so
it can only ever cost accuracy. The old figure was flattered by revisions
the operator would never have accepted. **Any Sirmour deviation number
quoted from before this date is on the old basis** - re-run
`SOLAR_PLANT=sirmour python -m tests.test_backtest` before comparing.

## Windows

```
automation\run_kasipet.bat
automation\run_bhupalpally.bat
```

Each starts a detached, minimised scheduler. Starting one twice is safe -
the second exits immediately on the lock port.

`run_scheduler.bat` in the project root is Sirmour's original launcher and
still works unchanged - it sets no `SOLAR_PLANT`, and the default is
`sirmour`. `automation/run_sirmour.bat` is the same thing written
explicitly; use either.

## Linux / EC2 (systemd)

```bash
sudo cp automation/solar-forecast-sirmour.service /etc/systemd/system/
sudo cp automation/solar-forecast-kasipet.service /etc/systemd/system/
sudo cp automation/solar-forecast-bhupalpally.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now solar-forecast-sirmour solar-forecast-kasipet solar-forecast-bhupalpally
```

The live box has run Sirmour as `solar-forecast.service` since before there
were three plants. `solar-forecast-sirmour.service` is the same service
under the per-plant name - **replace** the old unit, do not enable both.

Check on them with `systemctl status solar-forecast-kasipet` and
`tail -f logs/kasipet.log`.

**Before enabling all three on the existing box, read this.** The EC2
instance is a ~900 MB t3.micro, and a single forecast run peaks around
450 MB because of torch and the Chronos weights. Three schedulers idle at
about 28 MB each, which is fine - but three forecasts at the same instant
is not, and all three plants share the 06:45 / 08:15 / ... run times.

## Staggering

`scheduler.run_offset_seconds` in each plant's overlay delays when that
plant's job *fires*: Sirmour at 06:45:00, Kasipet at 06:50:00,
Bhupalpally at 06:55:00. One capture plus one forecast takes about two
minutes, so the three peaks queue instead of colliding.

This does not move the schedule. The orchestrator floors its run time to
the 15-minute block, so a run fired at 06:50 still produces the **06:45**
schedule, and the Windy clip it records still falls inside the 20-minute
tolerance for that slot. The offsets stay far below the 90 minutes
between run times. If the box is upgraded, set them to 0.

The stagger is a memory workaround, not a correctness one - separate
buckets already make cross-plant collision impossible. It does not apply on
Lambda; see [lambda/README.md](lambda/README.md).

## One-off runs

```bash
SOLAR_PLANT=kasipet python -m modules.orchestrator.pipeline          # one forecast now
SOLAR_PLANT=kasipet python -m modules.capture.windy_capture          # one Windy clip now
SOLAR_PLANT=kasipet python -m tests.generate_schedule_for_day 2026-08-06
SOLAR_PLANT=kasipet python -m tests.build_simple_schedule 2026-08-06
```

On Windows PowerShell, `$env:SOLAR_PLANT="kasipet"` first.
