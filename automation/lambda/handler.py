"""
=========================================================
Solar Forecasting Project
AWS Lambda entry point
=========================================================
Reference handler for the EC2 -> Lambda move. One invocation
does ONE job for ONE plant, decided entirely by two
environment variables set on the function:

    SOLAR_PLANT   sirmour | kasipet | bhupalpally
    SOLAR_TASK    capture | forecast

That gives six functions from this one file. They share no
state: separate containers, separate IAM roles, separate S3
buckets, separate CloudWatch log groups. A hung capture, an
exhausted Gemini quota or a crash on one plant cannot reach
another.

WHAT REPLACED WHAT
------------------
modules/scheduler/scheduler.py is NOT used here. On EC2 it
was a long-lived process that slept until a run time and then
spawned two subprocesses. EventBridge Scheduler does the
waiting now, and Lambda does the isolating, so three parts of
that file become dead weight on Lambda and are deliberately
not ported:

  * the single-instance lock port (49732/3/4) - it existed
    because Windows Task Scheduler stacked duplicate
    schedulers; EventBridge does not;
  * scheduler.run_offset_seconds, the 0/5/10-minute plant
    stagger - it existed because three forecasts peaking near
    450 MB at the same instant would OOM a ~900 MB t3.micro.
    Every Lambda invocation gets its own container, so all
    three plants can fire at the same minute. Leave the config
    values alone (EC2 still reads them); they are simply not
    consulted here;
  * the capture -> forecast subprocess split, which was about
    giving memory back between runs. Two separate FUNCTIONS
    now do that, and a warm container keeping torch loaded
    between forecasts is a speedup rather than a leak.

THE ONE REAL PORTING PROBLEM: THE FILESYSTEM
--------------------------------------------
Every path in config is relative (`data/plants/kasipet/raw`,
`models/plants/kasipet/case_store.csv`, `outputs/...`) and is
resolved against the CURRENT WORKING DIRECTORY. On Lambda the
image is mounted read-only at /var/task and only /tmp is
writable, so running the pipeline as-is fails the first time
it writes anything.

prepare_workspace() below is the fix: on a cold start it copies
the read-only inputs the pipeline expects to find beside it
into /tmp/solar, creates the folders it writes to, and chdirs
there. Warm invocations reuse it. Nothing in the pipeline had
to change for this.

Note what is NOT copied: the historical meter CSVs and the
model files are baked into the image, so /tmp only ever holds
one run's working data. Keep the image's data/ directory to
the history the pipeline actually reads, or cold starts get
slow and /tmp fills up.
=========================================================
"""

import os
import runpy
import shutil
from pathlib import Path


# Read-only image root, and the writable copy the pipeline runs in.
IMAGE_ROOT = Path(os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
WORK_ROOT = Path("/tmp/solar")

# Directories the pipeline READS from disk, relative to cwd. Copied out of
# the image once per cold start. config/ is not here on purpose - config.py
# resolves it from __file__, so it is read straight out of /var/task.
SEED_DIRS = ("data", "models")

# Directories the pipeline WRITES to. Created empty; anything worth keeping
# is pushed to S3 by the pipeline itself, and /tmp does not survive the
# container.
WRITE_DIRS = ("outputs", "logs")

TASK_MODULES = {
    "capture": "modules.capture.windy_capture",
    "forecast": "modules.orchestrator.pipeline",
}


def prepare_workspace():
    """
    Make /tmp/solar look like the project root, and work there.

    Idempotent: on a warm container the seed copy is already done and this
    is just a chdir. `dirs_exist_ok` means a half-finished copy from an
    invocation that timed out mid-cold-start is completed rather than
    raising.
    """

    for name in SEED_DIRS:

        source = IMAGE_ROOT / name

        if source.is_dir():
            shutil.copytree(source, WORK_ROOT / name, dirs_exist_ok=True)

    for name in WRITE_DIRS:
        (WORK_ROOT / name).mkdir(parents=True, exist_ok=True)

    os.chdir(WORK_ROOT)


def handler(event, context):
    """
    Run this function's one task for this function's one plant.

    Both target modules are written as `python -m` entry points ending in
    sys.exit(main()), so runpy reproduces exactly what the EC2 scheduler
    invoked - no separate Lambda-only code path that could drift from the
    thing that has been running for weeks.

    A non-zero exit is re-raised. That is deliberate: a silent failure
    would leave the plant on a stale schedule with nothing to alarm on,
    and Lambda's own error metric is the cheapest alarm available.
    """

    plant = os.environ["SOLAR_PLANT"]
    task = os.environ["SOLAR_TASK"]

    if task not in TASK_MODULES:
        raise ValueError(
            f"SOLAR_TASK must be one of {sorted(TASK_MODULES)}, got '{task}'"
        )

    prepare_workspace()

    print(f"[{plant}] running {task}")

    try:
        runpy.run_module(TASK_MODULES[task], run_name="__main__")

    except SystemExit as exit_signal:

        code = exit_signal.code or 0

        if code != 0:
            raise RuntimeError(
                f"[{plant}] {task} exited {code}"
            ) from exit_signal

    return {"plant": plant, "task": task, "status": "ok"}
