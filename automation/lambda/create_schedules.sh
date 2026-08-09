#!/usr/bin/env bash
#
# Every trigger, divided per plant, for the Lambda automation.
#
# 12 EventBridge schedules -> 46 invocations a day:
#
#     Sirmour       7 captures +  7 forecasts   (7 run times)
#     Kasipet       8 captures +  8 forecasts   (7 + the 17:15 Telangana run)
#     Bhupalpally   8 captures +  8 forecasts
#
# Twelve rather than forty-six because each plant's run times collapse into
# two cron expressions: the official times alternate :45 and :15 every 90
# minutes, so one rule covers the :45 runs and one covers the :15 runs.
#
# WHY CAPTURE AND FORECAST ARE SEPARATE FUNCTIONS
# -----------------------------------------------
# The EC2 scheduler ran them back to back in one trigger. On Lambda that
# would put a 300 s capture timeout and a 600 s forecast timeout inside a
# 900 s hard ceiling with nothing left over - a slow day would be killed
# mid-forecast. Splitting them also means a hung capture cannot delay or
# cancel the forecast, which is the same reason the EC2 version already ran
# the capture in a throwaway subprocess.
#
# The forecast fires 5 minutes after its capture. That is safe on both
# sides: the orchestrator floors its run time to the 15-minute block, so a
# forecast at 06:50 still publishes the 06:45 schedule, and 5 minutes is
# well inside windy_capture's 20-minute video match tolerance, so it picks
# up the clip that capture just uploaded.
#
# NO STAGGER BETWEEN PLANTS. On EC2, Sirmour/Kasipet/Bhupalpally fired at
# +0/+5/+10 minutes so three 450 MB forecasts never peaked together on a
# 900 MB box. Lambda gives each invocation its own container and each plant
# its own bucket, so all three fire at the same minute here.
#
#     bash automation/lambda/create_schedules.sh
#
# Re-running updates the existing schedules in place rather than failing.

set -euo pipefail

REGION="ap-south-1"
ACCOUNT="491429109178"

# The role EventBridge Scheduler assumes to invoke the functions. Create it
# first, with lambda:InvokeFunction on the six functions and a trust policy
# for scheduler.amazonaws.com.
SCHEDULER_ROLE="arn:aws:iam::${ACCOUNT}:role/solar-forecast-scheduler-invoke"

GROUP="solar-forecast"

# Cron is written in IST directly - EventBridge Scheduler takes a timezone,
# unlike classic EventBridge rules. Do not translate these to UTC by hand;
# the run times in the mentor's brief are IST and should stay readable as
# IST. (The UTC equivalents, if a tool ever forces them, are in README.md.)
TIMEZONE="Asia/Kolkata"

# Official run times, IST:
#   all plants   06:45  08:15  09:45  11:15  12:45  14:15  15:45
#   Telangana                                              + 17:15
#
# HOURS_45 covers 06:45 / 09:45 / 12:45 / 15:45.
# HOURS_15 covers 08:15 / 11:15 / 14:15, plus 17:15 for the two Telangana
# plants - that one extra hour in the list is the ONLY scheduling
# difference between the plants.
HOURS_45="6,9,12,15"
HOURS_15_SIRMOUR="8,11,14"
HOURS_15_TELANGANA="8,11,14,17"

# plant : hours for the :15 runs
PLANTS=(
    "sirmour:${HOURS_15_SIRMOUR}"
    "kasipet:${HOURS_15_TELANGANA}"
    "bhupalpally:${HOURS_15_TELANGANA}"
)

aws scheduler create-schedule-group \
    --name "$GROUP" --region "$REGION" 2>/dev/null \
    || echo "schedule group ${GROUP} already exists"


put_schedule() {

    local name="$1" cron="$2" function_name="$3"

    local target="{\"Arn\":\"arn:aws:lambda:${REGION}:${ACCOUNT}:function:${function_name}\",\"RoleArn\":\"${SCHEDULER_ROLE}\"}"

    local verb="create-schedule"

    if aws scheduler get-schedule \
        --name "$name" --group-name "$GROUP" \
        --region "$REGION" >/dev/null 2>&1
    then
        verb="update-schedule"
    fi

    aws scheduler "$verb" \
        --name "$name" \
        --group-name "$GROUP" \
        --region "$REGION" \
        --schedule-expression "$cron" \
        --schedule-expression-timezone "$TIMEZONE" \
        --flexible-time-window '{"Mode":"OFF"}' \
        --target "$target" >/dev/null

    echo "${verb%%-*}d ${name}  ${cron}  -> ${function_name}"
}


for entry in "${PLANTS[@]}"; do

    IFS=":" read -r plant hours_15 <<< "$entry"

    echo "=== ${plant} ==="

    # Capture, on the run time itself.
    put_schedule "${plant}-capture-45" \
        "cron(45 ${HOURS_45} ? * * *)"  "solar-capture-${plant}"

    put_schedule "${plant}-capture-15" \
        "cron(15 ${hours_15} ? * * *)"  "solar-capture-${plant}"

    # Forecast, five minutes later - same 15-minute block, same clip.
    put_schedule "${plant}-forecast-45" \
        "cron(50 ${HOURS_45} ? * * *)"  "solar-forecast-${plant}"

    put_schedule "${plant}-forecast-15" \
        "cron(20 ${hours_15} ? * * *)"  "solar-forecast-${plant}"

    echo
done

echo "Done. Check what will actually fire with:"
echo
echo "    aws scheduler list-schedules --group-name ${GROUP} --region ${REGION}"
