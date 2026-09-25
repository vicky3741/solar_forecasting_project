"""Build docs/images/forecast_vs_actual.png for the README.

Plots one day's Current Final Schedule against metered output, normalised
to % of plant capacity so no absolute plant figures are published.

    python docs/make_readme_chart.py 2026-07-26
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAPACITY_MW = 5.1  # Sirmour

day = sys.argv[1] if len(sys.argv) > 1 else "2026-07-26"
runs = pd.read_csv(ROOT / "outputs" / "schedules" / f"day_schedule_{day}_per_run.csv")

# Current Final Schedule: each block takes the latest run issued at or before it.
runs = runs[runs["scheduling_time"] <= runs["block_time"]]
final = (runs.sort_values("scheduling_time")
             .groupby("block_time", as_index=False).last()
             .sort_values("block"))
final = final[final["actual_is_real"]]

sched = final["scheduled_mw"] / CAPACITY_MW * 100
actual = final["actual_mw"] / CAPACITY_MW * 100
x = final["block_time"]
dev = (sched - actual).abs().mean()

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
ax.fill_between(range(len(x)), sched - 10, sched + 10, color="#4c78a8", alpha=0.12,
                label="±10% free deviation band")
ax.plot(range(len(x)), sched, color="#4c78a8", lw=2, label="AI forecast (final schedule)")
ax.plot(range(len(x)), actual, color="#e45756", lw=2, label="Actual (meter)")
step = 4
ax.set_xticks(range(0, len(x), step))
ax.set_xticklabels(x.iloc[::step], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% of plant capacity")
ax.set_ylim(bottom=0)
ax.set_title(f"Forecast vs actual, 15-min blocks, {day}  (avg deviation {dev:.1f}% of capacity)",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8, frameon=False)
fig.tight_layout()

out = ROOT / "docs" / "images" / "forecast_vs_actual.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}  avg deviation {dev:.2f}%")
