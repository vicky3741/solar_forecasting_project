# Stops the local automation scheduler (Windy capture + forecast loop).
# Used to switch our automation OFF once we move to the teammate's EC2
# system. Reversible - re-enable with:  Enable-ScheduledTask -TaskName SolarForecastScheduler
# then run:  .\run_scheduler.bat
$stopped = @()
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*modules.scheduler.scheduler*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $stopped += $_.ProcessId }

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') automation disabled - stopped scheduler PID(s): $($stopped -join ', ')" |
    Out-File -FilePath (Join-Path $logDir "automation_disabled.log") -Append -Encoding utf8
