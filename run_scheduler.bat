@echo off
REM Launches the Solar Forecasting scheduler (Windy capture + forecast + S3).
REM
REM PLAYWRIGHT_BROWSERS_PATH is set explicitly because Windows Task
REM Scheduler hands the process a trimmed environment - leaving Playwright
REM to infer the browser location previously made it fail with
REM "Executable doesn't exist" on every scheduled run while working fine
REM from an interactive shell.

cd /d "%~dp0"
set PLAYWRIGHT_BROWSERS_PATH=%LOCALAPPDATA%\ms-playwright

start /min "Solar Forecasting Scheduler" ".venv\Scripts\python.exe" -m modules.scheduler.scheduler
