"""
=========================================================
Solar Forecasting Project
Meter Data Quality Test / Report
=========================================================
Prints a per-day data-quality table for every historical
meter file, and an overall total of how many 15-min
"actual" blocks are real measurements vs reconstructed by
gap-filling. Read-only - changes nothing.

Run:  python -m tests.test_meter_quality
=========================================================
"""

from pathlib import Path

from config.config import settings
from modules.preprocessing.quality_report import MeterQualityReport


def main():

    hist = Path(settings["paths"]["historical_data"])
    files = sorted(hist.glob("*.csv"))

    reporter = MeterQualityReport()

    print("=" * 82)
    print("METER DATA QUALITY - real vs reconstructed 15-min blocks")
    print("=" * 82)
    print(f"{'day':12s} {'cover%':>7s} {'real':>5s} {'recon':>6s} "
          f"{'maxgap':>7s} {'span':>13s} {'peak_kw':>8s}")
    print("-" * 82)

    total_real = 0
    total_expected = 0
    worst = []

    for f in files:

        day = f.stem.replace("_SOLAR_INV", "")

        import pandas as pd
        metrics = reporter.profile(pd.read_csv(f))

        if metrics.get("span") is None:
            print(f"{day:12s}  unusable")
            continue

        total_real += metrics["present_blocks"]
        total_expected += metrics["expected_blocks"]

        if metrics["max_gap_minutes"] >= 60:
            worst.append((day, metrics["max_gap_minutes"]))

        print(f"{day:12s} {metrics['coverage_pct']:7.0f} "
              f"{metrics['present_blocks']:5d} {metrics['reconstructed_blocks']:6d} "
              f"{str(metrics['max_gap_minutes']) + 'm':>7s} "
              f"{metrics['span']:>13s} {metrics['peak_kw']:8.0f}")

    reconstructed = total_expected - total_real
    print("-" * 82)
    print(f"TOTAL: {total_real}/{total_expected} blocks real "
          f"({total_real / total_expected * 100:.1f}%), "
          f"{reconstructed} reconstructed ({reconstructed / total_expected * 100:.1f}%)")

    if worst:
        print()
        print("Days with holes >= 1 hour (accuracy on these is partly vs filled data):")
        for day, gap in sorted(worst, key=lambda x: -x[1]):
            print(f"  {day}: {gap} min ({gap / 60:.1f} h)")


if __name__ == "__main__":
    main()
