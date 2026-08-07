"""
=========================================================
Solar Forecasting Project
Effective Time / Freeze Horizon
=========================================================
MENTOR GUIDANCE 2026-08-08 ("Simple Effective Time Schedule
Guide", supplied with the two new Telangana plants).

A schedule that is generated at 11:15 does not take effect at
11:15. The next few blocks have already been declared to the
grid operator, so they stay at whatever the PREVIOUS schedule
said; the new numbers only start once the freeze horizon has
passed. How long that horizon is differs per plant:

    Sirmour              6 blocks  = 90 minutes
    Kasipet / Kothagudem 3 blocks  = 45 minutes

Worked through with the guide's own example - a run at 11:15
is "engine block" 46:

    Sirmour   freeze 46,47,48,49,50,51  -> new from block 52 (12:45)
    Kasipet   freeze 46,47,48           -> new from block 49 (12:00)

So the effective start is simply

    engine block + freeze_blocks

and since the engine block is the block the run fires in, that
is `run_time + freeze_blocks x 15 minutes` on the clock.

TWO THINGS THIS MODULE IS CAREFUL ABOUT
---------------------------------------
1. A block is only frozen if the previous schedule ACTUALLY HAS
   a value for it. The guide says so directly ("if this is the
   first schedule of the day and no previous schedule exists,
   there may be nothing to freeze"), and without it the day
   would come out with holes: with a 3-block freeze the 06:45
   run would stop at block 31 while the 08:15 run starts at
   block 37, leaving 32-36 scheduled by nobody.

2. freeze_blocks of 0 or 1 means "no freeze" and reproduces the
   original behaviour exactly - every block after the run time
   is rewritten. That is what Sirmour is set to, deliberately,
   so this module changes nothing for the plant that is already
   live. See schedule_rules in config/settings.yaml.
=========================================================
"""

import pandas as pd


def block_number(timestamp):
    """
    Indian scheduling block number for a single timestamp -
    block 1 is 00:00-00:15, so 06:45 is block 28 and 11:15 is
    block 46, matching the guide's quick-reference table.
    """

    timestamp = pd.Timestamp(timestamp)

    return timestamp.hour * 4 + timestamp.minute // 15 + 1


def effective_start(run_time, freeze_blocks, interval_minutes=15):
    """
    The first timestamp a schedule generated at `run_time` is
    allowed to change.

    freeze_blocks <= 1 gives run_time + one block, i.e. the plain
    "everything after the run time" rule.
    """

    run_time = pd.Timestamp(run_time)

    steps = max(int(freeze_blocks), 1)

    return run_time + pd.Timedelta(minutes=steps * interval_minutes)


def freeze_window(run_time, freeze_blocks, interval_minutes=15):
    """
    (first_block, last_block, effective_block) for logging and for
    the report header - the numbers the guide's table shows.
    Returns (None, None, effective_block) when nothing is frozen.
    """

    start = effective_start(run_time, freeze_blocks, interval_minutes)

    engine = block_number(run_time)
    effective = block_number(start)

    if int(freeze_blocks) <= 1:
        return None, None, effective

    return engine, effective - 1, effective


def apply_freeze(
    new_values,
    previous_values,
    run_time,
    freeze_blocks,
    interval_minutes=15,
):
    """
    Merge one run's fresh numbers with the schedule already
    published, under this plant's freeze horizon.

    new_values      : {timestamp -> value} this run wants to publish
    previous_values : {timestamp -> value} the standing schedule
    returns         : {timestamp -> value} to publish, plus the set
                      of timestamps that were actually held back

    Only timestamps at or after the effective start are taken from
    `new_values`; earlier ones are taken from `previous_values` when
    it has them, and from `new_values` when it does not (nothing to
    freeze).
    """

    start = effective_start(run_time, freeze_blocks, interval_minutes)

    published = {}
    frozen = []

    for timestamp, value in new_values.items():

        if timestamp < start and timestamp in previous_values:
            published[timestamp] = previous_values[timestamp]
            frozen.append(timestamp)
        else:
            published[timestamp] = value

    return published, frozen
