"""
=========================================================
Solar Forecasting Project
Full Backtest
=========================================================
Runs every official 15-min-resolution forecast time on
every historical day (point-in-time, no lookahead), scores
our hybrid forecast against actual generation and against
Enercast, then runs a coarse parameter tuning pass using
the cached signals from that same run.
=========================================================
"""

from pathlib import Path

import pandas as pd

from modules.evaluation.backtester import Backtester


def main():

    processed_file = Path("data/processed/processed_data.csv")

    dataframe = pd.read_csv(processed_file, parse_dates=["timestamp"])

    backtester = Backtester()

    print("=" * 60)
    print("Running backtest across all days and official run-times")
    print("=" * 60)

    results, raw_signals_cache, details = backtester.run(dataframe)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("\nPer-run results")
    print(results)

    total_runs = len(results)
    vision_runs = int(results["has_vision"].sum())

    print(f"\nTotal runs scored : {total_runs}")
    print(f"Runs with vision signal : {vision_runs}")

    print("\n--- Overall: Our Forecast ---")
    print(f"Average MAE (kW)        : {results['our_mae_kw'].mean():.2f}")
    print(f"Average deviation (%)   : {results['our_avg_deviation_pct'].mean():.2f}")
    print(f"Average penalty         : {results['our_scheduling_penalty'].mean():.2f}")

    enercast_rows = results.dropna(subset=["enercast_mae_kw"]) if "enercast_mae_kw" in results.columns else results.iloc[0:0]

    if not enercast_rows.empty:
        print(f"\n--- Overall: Enercast ({len(enercast_rows)} runs with Enercast data) ---")
        print(f"Average MAE (kW)        : {enercast_rows['enercast_mae_kw'].mean():.2f}")
        print(f"Average deviation (%)   : {enercast_rows['enercast_avg_deviation_pct'].mean():.2f}")
        print(f"Average penalty         : {enercast_rows['enercast_scheduling_penalty'].mean():.2f}")

    print("\n" + "=" * 60)
    print("Coarse parameter tuning (re-using cached signals, no Chronos re-runs)")
    print("=" * 60)

    chronos_weight_candidates = [0.2, 0.4, 0.6, 0.8]
    performance_ratio_candidates = [0.70, 0.75, 0.80, 0.85]
    trend_halflife_candidates = [0, 45, 90, 180]

    grid, best = backtester.tune(
        raw_signals_cache,
        chronos_weight_candidates,
        performance_ratio_candidates,
        trend_halflife_candidates
    )

    print("\nGrid results (top 15 by average deviation %)")
    print(grid.sort_values("avg_deviation_pct").reset_index(drop=True).head(15))

    print("\nBest combo found:")
    print(best)

    current_deviation = results["our_avg_deviation_pct"].mean()

    print(f"\nCurrent settings average deviation %  : {current_deviation:.2f}")
    print(f"Best tuned average deviation %        : {best['avg_deviation_pct']:.2f}")

    results_path = Path("outputs/reports/backtest_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"\nBacktest results saved to {results_path} (used by the web dashboard's history view)")

    detail_path = Path("outputs/reports/backtest_detail.csv")
    details.to_csv(detail_path, index=False)
    print(f"Per-block detail saved to {detail_path} (used by the dashboard's comparison view)")


if __name__ == "__main__":
    main()
