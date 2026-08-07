@echo off
REM ============================================================
REM Launches the SIRMOUR scheduler.
REM
REM Identical in effect to run_scheduler.bat in the project root, which
REM is the original launcher and is still what the live automation uses.
REM This one only states SOLAR_PLANT explicitly, so the three plants
REM read the same way side by side in this folder. Either works;
REM sirmour is the default when the variable is unset.
REM ============================================================

cd /d "%~dp0.."
set SOLAR_PLANT=sirmour
set PLAYWRIGHT_BROWSERS_PATH=%LOCALAPPDATA%\ms-playwright

start /min "Solar Forecasting Scheduler - Sirmour" ".venv\Scripts\python.exe" -m modules.scheduler.scheduler
