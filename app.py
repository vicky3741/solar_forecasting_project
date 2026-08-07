"""
=========================================================
Solar Forecasting Project
Web App
=========================================================
Serves the dashboard and exposes the forecast results as
a small JSON API:

  GET /api/schedule/current  - the Current Final Schedule
                                (actual generation for
                                completed blocks, forecast
                                for blocks still ahead)
  GET /api/history           - the multi-day backtest
                                comparison (our forecast vs
                                Enercast vs actual)
  GET /api/comparison        - per-15-min-block detail for
                                every backtested run (our kW
                                vs Enercast kW vs actual kW)

Run directly with:  python app.py
Then open:          http://localhost:8000

Both endpoints read from files already produced by the
orchestrator/backtester rather than recomputing anything
live - re-running Chronos on every web request would be
far too slow for a page load.

WHICH PLANT IT SHOWS
--------------------
One plant per server, chosen by SOLAR_PLANT the same way
every other entry point chooses (default sirmour):

    SOLAR_PLANT=kasipet python app.py            # port 8000
    SOLAR_PLANT=bhupalpally PORT=8001 python app.py

Three plants side by side means three servers on three
ports. That is deliberate rather than a plant selector in
the UI: the dashboard reads whatever the configuration
points at, so it can never show one plant's schedule under
another's name.
=========================================================
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.config import settings
from utils import file_manager


def records_with_nulls(dataframe):
    """
    Converts NaN (e.g. Enercast columns on days with no
    Enercast file) to JSON-serializable None - Starlette's
    JSON encoder rejects NaN as non-compliant.
    """

    return dataframe.astype(object).where(
        dataframe.notna(), None
    ).to_dict(orient="records")

PLANT = settings["plant"]

app = FastAPI(title=f"Solar Forecasting Dashboard - {PLANT['name']}")

SCHEDULE_PATH = Path(settings["outputs"]["schedules"]) / "current_final_schedule.csv"
HISTORY_PATH = Path(settings["outputs"]["reports"]) / "backtest_results.csv"
DETAIL_PATH = Path(settings["outputs"]["reports"]) / "backtest_detail.csv"

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/plant")
def get_plant():
    """
    Which plant this server is serving. The page can show it in the
    header so a screenshot of one plant's dashboard is never mistaken
    for another's.
    """

    return {
        "key": PLANT["key"],
        "name": PLANT["name"],
        "state": PLANT.get("state"),
        "capacity_mw": PLANT["capacity_mw"],
        "run_times": settings["forecast"]["run_times"],
        "freeze_blocks": settings.get("schedule_rules", {}).get("freeze_blocks", 0),
    }


@app.get("/api/schedule/current")
def get_current_schedule():

    schedule = file_manager.load_dataframe(
        SCHEDULE_PATH,
        parse_dates=["timestamp", "last_updated_run_time"]
    )

    if schedule is None:
        raise HTTPException(
            status_code=404,
            detail="No Current Final Schedule yet - run the orchestrator first."
        )

    return records_with_nulls(schedule)


@app.get("/api/history")
def get_history():

    history = file_manager.load_dataframe(HISTORY_PATH, parse_dates=["date"])

    if history is None:
        raise HTTPException(
            status_code=404,
            detail="No backtest results yet - run tests/test_backtest.py first."
        )

    return records_with_nulls(history)


@app.get("/api/comparison")
def get_comparison():
    """
    Per-block detail for every backtested run: our forecast
    kW vs Enercast kW vs actual kW. The frontend filters by
    day / run-time client-side.
    """

    detail = file_manager.load_dataframe(
        DETAIL_PATH,
        parse_dates=["timestamp"]
    )

    if detail is None:
        raise HTTPException(
            status_code=404,
            detail="No per-block detail yet - run tests/test_backtest.py first."
        )

    return records_with_nulls(detail)


if __name__ == "__main__":

    import uvicorn

    # PORT so three plants' dashboards can run side by side.
    port = int(os.getenv("PORT", "8000"))

    print(f"Serving {PLANT['name']} on http://127.0.0.1:{port}")

    uvicorn.run(app, host="127.0.0.1", port=port)
