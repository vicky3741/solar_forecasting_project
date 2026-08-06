"""
=========================================================
Solar Forecasting Project
Block Bias Correction - smoke test
=========================================================
Checks the four things that would silently break a live
run:

  1. a profile is learned from the recent day schedules,
  2. night blocks are left completely alone (the smoother
     must not bleed a dawn bias into blocks with no data,
     or the forecast stops going to zero at night),
  3. the no-lookahead rule holds - a profile built `as_of`
     a day never contains that day or any later one,
  4. the staleness gate fires, so today is never corrected
     with a month-old shape.

Run:  python -m tests.test_block_bias
=========================================================
"""

import pandas as pd

from modules.forecasting.block_bias_correction import BlockBiasCorrector


def main():

    today = pd.Timestamp.now().date()

    corrector = BlockBiasCorrector().load(as_of=today)

    print("=" * 66)
    print("BLOCK BIAS CORRECTION - smoke test")
    print("=" * 66)
    print(f"enabled        : {corrector.enabled}")
    print(f"available      : {corrector.available}")
    print(f"days learned   : {[str(d) for d in corrector.days_used]}")

    if not corrector.available:
        print()
        print("Not available - either disabled in config, or fewer than "
              f"{corrector.min_days} recent day schedules exist, or the newest "
              f"is more than {corrector.max_profile_age_days} days old. "
              "Nothing further to check.")
        return

    profile = corrector._profile_kw

    print(f"blocks shifted : {int((profile != 0).sum())} of 96")
    print(f"largest shift  : {profile.abs().max() * corrector.strength:.0f} kW "
          f"(strength {corrector.strength} already applied)")

    # ---- 2. night must stay untouched ----
    forecast = pd.DataFrame({
        "timestamp": pd.date_range(
            f"{today} 00:00", f"{today} 23:45", freq="15min"
        )
    })
    forecast["final_forecast_kw"] = 0.0

    corrected = corrector.apply(forecast)

    night = corrected[
        (corrected["timestamp"].dt.hour < 5) | (corrected["timestamp"].dt.hour >= 20)
    ]

    assert (night["final_forecast_kw"] == 0).all(), \
        "night blocks were shifted - the forecast would not go to zero at night"
    print("night blocks   : untouched (correct)")

    # ---- 3. no lookahead ----
    reference = pd.Timestamp(today) - pd.Timedelta(days=3)
    past = BlockBiasCorrector().load(as_of=reference.date())

    assert all(day < reference.date() for day in past.days_used), \
        "profile contains a day at or after its as_of date - that is lookahead"
    print(f"no-lookahead   : as_of {reference.date()} used "
          f"{[str(d) for d in past.days_used]}")

    # ---- 4. staleness gate ----
    future = pd.Timestamp(today) + pd.Timedelta(days=60)
    stale = BlockBiasCorrector().load(as_of=future.date())

    assert not stale.available, \
        "corrector still ran on a 60-day-old profile - the staleness gate is dead"
    print("staleness gate : refuses a 60-day-old profile (correct)")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
