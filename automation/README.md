# Automation - one independent service per plant

Three plants, three schedulers, one codebase. Each launcher does exactly
one thing that matters: it sets `SOLAR_PLANT` before Python starts. From
there `config/config.py` resolves that plant's overlay and the whole
process - data folders, S3 prefixes, run times, freeze horizon, log file,
lock port - is that plant's and nothing else's.

| Plant | `SOLAR_PLANT` | Runs/day | Freeze | Lock port | Stagger | Log |
|---|---|---|---|---|---|---|
| Sirmour (MP, 5.1 MW) | `sirmour` | 7 | 0 blocks | 49732 | 0 | `logs/solar_forecasting.log` |
| Kasipet (TG, 15 MW) | `kasipet` | 8 | 3 blocks | 49733 | +5 min | `logs/kasipet.log` |
| Bhupalpally (TG, 10 MW) | `bhupalpally` | 8 | 3 blocks | 49734 | +10 min | `logs/bhupalpally.log` |

They are independent on purpose: a crashed capture, an exhausted API
quota or a hung forecast on one plant stops that plant only. The lock
ports differ for the same reason - the single-instance guard binds a
fixed port, so three services sharing one port would mean only the first
to start ever runs.

## Sirmour has not moved

`run_scheduler.bat` in the project root is Sirmour's original launcher
and still works unchanged - it sets no `SOLAR_PLANT`, and the default is
`sirmour`. `automation/run_sirmour.bat` is the same thing written
explicitly; use either.

## Windows

```
automation\run_kasipet.bat
automation\run_bhupalpally.bat
```

Each starts a detached, minimised scheduler. Starting one twice is safe -
the second exits immediately on the lock port.

## Linux / EC2 (systemd)

```bash
sudo cp automation/solar-forecast-kasipet.service /etc/systemd/system/
sudo cp automation/solar-forecast-bhupalpally.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now solar-forecast-kasipet solar-forecast-bhupalpally
```

Check on them with `systemctl status solar-forecast-kasipet` and
`tail -f logs/kasipet.log`.

**Before enabling both on the existing box, read this.** The EC2 instance
is a ~900 MB t3.micro, and a single forecast run peaks around 450 MB
because of torch and the Chronos weights. Three schedulers idle at about
28 MB each, which is fine - but three forecasts at the same instant is
not, and all three plants share the 06:45 / 08:15 / ... run times.

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

## One-off runs

```bash
SOLAR_PLANT=kasipet python -m modules.orchestrator.pipeline          # one forecast now
SOLAR_PLANT=kasipet python -m modules.capture.windy_capture          # one Windy clip now
SOLAR_PLANT=kasipet python -m tests.generate_schedule_for_day 2026-08-06
SOLAR_PLANT=kasipet python -m tests.build_simple_schedule 2026-08-06
```

On Windows PowerShell, `$env:SOLAR_PLANT="kasipet"` first.
