@echo off
REM ============================================================
REM Launches the BHUPALPALLY scheduler (Windy capture + forecast + S3).
REM
REM The one line that matters is SOLAR_PLANT: everything downstream -
REM meter folder, output folder, S3 prefixes, run times, freeze
REM horizon, log file and lock port - is resolved from it by
REM config/config.py. Sirmour's own launcher is untouched and does not
REM set it, so it keeps getting the default.
REM
REM PLAYWRIGHT_BROWSERS_PATH is set explicitly for the same reason it is
REM in run_scheduler.bat: Windows Task Scheduler hands the process a
REM trimmed environment, and leaving Playwright to infer the browser
REM location made every scheduled capture fail with "Executable doesn't
REM exist" while working fine from an interactive shell.
REM ============================================================

cd /d "%~dp0.."
set SOLAR_PLANT=bhupalpally
set PLAYWRIGHT_BROWSERS_PATH=%LOCALAPPDATA%\ms-playwright

start /min "Solar Forecasting Scheduler - Bhupalpally" ".venv\Scripts\python.exe" -m modules.scheduler.scheduler
