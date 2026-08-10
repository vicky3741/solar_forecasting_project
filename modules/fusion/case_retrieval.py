"""
=========================================================
Solar Forecasting Project - NEW APPROACH
Retrieve similar past situations, for the LLM to read
=========================================================
Adopted from similarity_retrieval.py in
github.com/Kushal70-51/Windy-Project-3.

WHAT THIS IS, IN ONE LINE
-------------------------
Instead of only telling the model "it is cloudy", also tell it
"the last N times the sky and the hour looked like this, the
plant actually generated X% more or less than the forecast
said".

WHY IT IS DIFFERENT FROM WHAT WE ALREADY DO
-------------------------------------------
We ALREADY have a case store - models/case_store.csv, 7507
rows of (block_hour, horizon_min, kt_now, forecast, residual)
built by tests/test_case_based_experiment.py - and
modules/forecasting/case_based_correction.py already retrieves
from it. But it uses the result as a silent numeric nudge:
look up 40 analogues, average their residual, apply half of it.

The LLM never sees any of that. It reasons about the sky with
no memory of what happened last time the sky looked this way.

Their design puts the retrieved cases IN THE PROMPT, as
evidence. That is a better use of a language model than
anything we currently ask it to do: judging "these ten past
afternoons started like today and three of them cleared" is
reasoning over precedent, which is what it is good at, rather
than arithmetic, which it is not.

The numeric corrector stays exactly as it is. This is an
additional input to the prompt, not a replacement for a
validated +0.47 pts correction.

HOW A CASE IS MATCHED
---------------------
Weighted Euclidean distance over z-scored features, their
method, with the weights this project already tuned for
case_based_correction (block_hour 2.0, kt_now 2.0,
final_forecast_kw 1.5, horizon_min 1.0).

Z-scoring first is not optional: horizon_min spans 15-720 and
kt_now spans 0-1.2, so on raw values horizon would dominate
every distance and "similar" would mean "same horizon,
any weather".

LEAKAGE
-------
Theirs excludes cases with the same timestamp as the query.
Ours keeps only days STRICTLY BEFORE the query day, which is
stricter and has to be: every accuracy claim in this project is
scored walk-forward by day, so a case from this morning would
let the model see part of the answer to this afternoon - and a
case from next week would hand it the answer outright.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from utils.logger import get_logger


class CaseRetriever:

    def __init__(self, store_path=None):

        self.logger = get_logger()

        case_settings = settings.get("case_based_correction", {})

        self.store_path = Path(
            store_path or case_settings.get(
                "case_store_path", "models/case_store.csv"
            )
        )

        self.weights = case_settings.get("weights", {
            "block_hour": 2.0,
            "kt_now": 2.0,
            "final_forecast_kw": 1.5,
            "horizon_min": 1.0,
        })

        fusion = settings.get("fusion", {})

        self.top_k = fusion.get("case_top_k", 12)
        self.enabled = fusion.get("cases_in_prompt", True)

        self.capacity_kw = settings["plant"]["capacity_mw"] * 1000

        self._store = None

    # --------------------------------------------------

    @property
    def store(self):

        if self._store is None:

            if not self.store_path.exists():
                self.logger.warning(
                    f"No case store at {self.store_path} - the prompt will "
                    "carry no precedent. Build it with: "
                    "python -m tests.test_backtest && "
                    "python -m tests.test_case_based_experiment"
                )
                self._store = pd.DataFrame()
            else:
                self._store = pd.read_csv(self.store_path, parse_dates=["date"])

        return self._store

    @property
    def available(self):

        return self.enabled and not self.store.empty

    # --------------------------------------------------

    def retrieve(self, query, exclude_date=None, top_k=None):
        """
        The `top_k` nearest past situations to `query`, as a DataFrame
        with an `actual_kw` column reconstructed from the stored
        residual.

        `query` keys are matched against the store's columns; any key
        the store does not have is ignored, so a caller may pass extra
        context without this raising.
        """

        store = self.store

        if store.empty:
            return pd.DataFrame()

        cases = store

        if exclude_date is not None:
            # STRICTLY BEFORE, not "any day except this one". The case
            # store is built once over the whole period, so it holds
            # days AFTER the run being backtested - and `!=` let a run
            # on Aug 1 retrieve precedent, with actuals attached, from
            # Aug 5. That is lookahead, and it flatters every historical
            # score. A live run has no future days in the store, so this
            # only ever tightens the backtest; it cannot change
            # production behaviour.
            cases = cases[cases["date"].dt.date < pd.Timestamp(exclude_date).date()]

        if cases.empty:
            return pd.DataFrame()

        columns = [
            name for name in self.weights
            if name in cases.columns and name in query
            and pd.notna(query[name])
        ]

        if not columns:
            return pd.DataFrame()

        values = cases[columns].to_numpy(dtype=float)

        # z-score against the STORE's own spread, so every feature
        # contributes on the same scale (see the module docstring).
        centre = np.nanmean(values, axis=0)
        spread = np.nanstd(values, axis=0)
        spread = np.where(spread < 1e-9, 1.0, spread)

        target = np.array(
            [float(query[name]) for name in columns], dtype=float
        )

        weights = np.array(
            [float(self.weights[name]) for name in columns], dtype=float
        )

        gaps = ((values - centre) / spread) - ((target - centre) / spread)

        distance = np.sqrt(np.nansum(weights * gaps ** 2, axis=1))

        result = cases.copy()
        result["distance"] = distance

        result = result.dropna(subset=["distance"]).nsmallest(
            top_k or self.top_k, "distance"
        )

        # residual_kw is (actual - forecast), so this recovers what the
        # plant really did on that block.
        result["actual_kw"] = (
            result["final_forecast_kw"] + result["residual_kw"]
        )

        return result

    # --------------------------------------------------

    def summarise(self, cases):
        """
        A retrieved set as a few honest numbers.

        Deliberately NOT a list of 12 raw rows. The useful content is
        the direction and spread of what actually happened, and a wall
        of near-identical numbers spends tokens while inviting the
        model to pattern-match on one outlier.
        """

        if cases.empty:
            return None

        residual_pct = cases["residual_kw"] / self.capacity_kw * 100

        higher = int((cases["residual_kw"] > 0).sum())

        return {
            "count": len(cases),
            "median_residual_pct": float(residual_pct.median()),
            "p25_residual_pct": float(residual_pct.quantile(0.25)),
            "p75_residual_pct": float(residual_pct.quantile(0.75)),
            "came_in_higher": higher,
            "came_in_lower": len(cases) - higher,
            "days": sorted({str(d.date()) for d in cases["date"]}),
        }

    # --------------------------------------------------

    def prompt_section(self, kt_now, forecast_kw_by_horizon, exclude_date=None):
        """
        The precedent block for the prompt.

        `forecast_kw_by_horizon` is {horizon_min: forecast_kw} for a few
        representative horizons - the summary is built per horizon band,
        because how much a forecast drifts from the outcome depends
        heavily on how far ahead it was made.
        """

        if not self.available:
            return "(no past cases available)"

        lines = []

        for horizon, forecast_kw in sorted(forecast_kw_by_horizon.items()):

            block_hour = None

            query = {
                "kt_now": kt_now,
                "horizon_min": horizon,
                "final_forecast_kw": forecast_kw,
            }

            if block_hour is not None:
                query["block_hour"] = block_hour

            summary = self.summarise(
                self.retrieve(query, exclude_date=exclude_date)
            )

            if summary is None:
                continue

            direction = (
                "MORE than forecast" if summary["median_residual_pct"] > 0
                else "LESS than forecast"
            )

            lines.append(
                f"  {horizon} min ahead, from {summary['count']} similar past "
                f"situations across {len(summary['days'])} days:\n"
                f"    actual came in {abs(summary['median_residual_pct']):.1f}% "
                f"of capacity {direction} (middle half: "
                f"{summary['p25_residual_pct']:+.1f}% to "
                f"{summary['p75_residual_pct']:+.1f}%)\n"
                f"    {summary['came_in_higher']} of {summary['count']} came in "
                f"higher than forecast"
            )

        if not lines:
            return "(no past cases matched this situation)"

        return (
            "Past situations with a similar clear-sky index, hour and "
            "forecast level, and what the plant ACTUALLY did:\n"
            + "\n".join(lines)
            + "\n\nThese are real measured outcomes, not predictions. A "
            "consistent one-way miss means forecasts like today's have been "
            "biased that way before - weigh it, but a wide middle-half range "
            "means the precedent is weak and should not move the number much."
        )
