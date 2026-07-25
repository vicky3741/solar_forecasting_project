"""
=========================================================
Solar Forecasting Project
Meter Data Quality Report
=========================================================
Profiles a raw daily meter CSV and reports how trustworthy
it is BEFORE any cleaning fills the holes. This is
reporting only - it never modifies the data - so it is
safe to run on every ingest without changing forecast
results.

Why it matters: field meter files have data holes (the
sensor drops out for minutes to hours). The alignment
step fills those holes so the pipeline has a complete
15-min grid, but a filled block is not a measurement.
This report makes the difference visible, so we always
know how much of a day's "actual" generation is real vs
reconstructed - which is exactly what the accuracy
grading depends on.
=========================================================
"""

import pandas as pd


class MeterQualityReport:

    def __init__(self, block_minutes=15):

        self.block = pd.Timedelta(minutes=block_minutes)

    # --------------------------------------------------

    @staticmethod
    def _find_power_column(dataframe):

        for column in dataframe.columns:
            lowered = column.lower()
            if "active" in lowered and "power" in lowered:
                return column

        return None

    # --------------------------------------------------

    def profile(self, dataframe, timestamp_column="TimeStamp"):
        """
        Returns a dict of quality metrics for one day's raw
        meter dataframe. Never raises on messy data - bad
        timestamps are counted, not fatal.
        """

        rows = len(dataframe)

        timestamps = pd.to_datetime(
            dataframe[timestamp_column], errors="coerce"
        )

        bad_timestamps = int(timestamps.isna().sum())

        valid = timestamps.dropna().sort_values()

        metrics = {
            "rows": rows,
            "bad_timestamps": bad_timestamps,
            "duplicate_timestamps": int(valid.duplicated().sum()),
        }

        if len(valid) < 2:
            metrics.update({
                "span": None,
                "expected_blocks": 0,
                "present_blocks": len(valid),
                "gap_count": 0,
                "max_gap_minutes": 0,
                "reconstructed_blocks": 0,
                "coverage_pct": 0.0,
            })
            return metrics

        start, end = valid.iloc[0], valid.iloc[-1]

        # Blocks a complete day at this cadence WOULD have over the
        # observed span, vs how many the file actually carries.
        expected_blocks = int((end - start) / self.block) + 1
        present_blocks = len(valid.drop_duplicates())

        diffs = valid.diff().dropna()
        gaps = diffs[diffs > self.block]

        # Every missing block inside the span has to be reconstructed
        # (interpolated) later; this is the count that is NOT measured.
        reconstructed = max(0, expected_blocks - present_blocks)

        power_column = self._find_power_column(dataframe)
        if power_column is not None:
            power = pd.to_numeric(dataframe[power_column], errors="coerce")
            metrics["negative_power_blocks"] = int((power < -0.5).sum())
            metrics["peak_kw"] = round(float(power.max()), 1)
        else:
            metrics["negative_power_blocks"] = None
            metrics["peak_kw"] = None

        metrics.update({
            "span": f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}",
            "expected_blocks": expected_blocks,
            "present_blocks": present_blocks,
            "gap_count": int(len(gaps)),
            "max_gap_minutes": int(diffs.max().total_seconds() // 60),
            "reconstructed_blocks": reconstructed,
            "coverage_pct": round(present_blocks / expected_blocks * 100, 1),
        })

        return metrics

    # --------------------------------------------------

    def summary_line(self, day_label, metrics):
        """One-line human summary, safe to log per ingest."""

        if metrics.get("span") is None:
            return f"{day_label}: unusable ({metrics['rows']} rows, no valid timestamps)"

        flag = ""
        if metrics["max_gap_minutes"] >= 60:
            flag = f"  <-- {metrics['max_gap_minutes']}min hole reconstructed"

        return (
            f"{day_label}: {metrics['coverage_pct']:.0f}% real "
            f"({metrics['present_blocks']}/{metrics['expected_blocks']} blocks), "
            f"{metrics['reconstructed_blocks']} reconstructed, "
            f"span {metrics['span']}{flag}"
        )
