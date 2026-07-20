"""
=========================================================
Solar Forecasting Project
Days 12-18 Accuracy Report (out-of-sample, numbers only)
=========================================================
Mentor task: "Apply the developed AI model to this data and
do the prediction of meter data. Check accuracy of the
results and share the same on the group."

Produces a NUMBERS table (predicted vs actual meter
generation + accuracy %) for 2026-07-12 .. 2026-07-18.

Method / honesty:
  - Accuracy is averaged across all 7 official intraday runs
    per day (06:45 .. 15:45), matching how the system really
    operates - it re-forecasts 7x/day. Judging a single
    dawn run would be unrepresentative (at 06:45 there are
    only ~2 readings and dawn haze makes the estimate noisy).
  - Each run uses only data available at its run time
    (no lookahead).
  - The LightGBM residual correction is applied WALK-FORWARD:
    day D is corrected only by a model trained on days
    BEFORE D, and only once >= min_training_days of history
    exist - so days 12-18 are genuinely out-of-sample.
  - Predicted (MWh) per day = the intraday full-day energy
    estimate (actual generation so far + forecast for the
    rest), averaged across the 7 runs.
  - Accuracy % = 100 - mean(|predicted - actual|)/capacity.
=========================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from PIL import Image, ImageDraw, ImageFont

from config.config import settings
from modules.forecasting.clearsky import ClearSkyModel
from modules.forecasting.residual_correction import (
    FEATURES, TRAINING_PARAMS, NUM_BOOST_ROUND
)
from tests.test_residual_experiment import compute_kt_now_lookup


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
MIN_TRAINING_DAYS = settings["residual_correction"]["min_training_days"]
RUN_TIMES = settings["forecast"]["run_times"]
REPORT_DAYS = [f"2026-07-{d}" for d in range(12, 19)]


def build_features(detail, kt_lookup):

    detail = detail.copy()
    detail["date"] = detail["date"].astype(str)
    run_dt = pd.to_datetime(detail["date"] + " " + detail["run_time"])

    detail["horizon_min"] = (detail["timestamp"] - run_dt).dt.total_seconds() / 60
    detail["block_hour"] = detail["timestamp"].dt.hour + detail["timestamp"].dt.minute / 60
    detail["kt_now"] = [
        kt_lookup[(d, r)] for d, r in zip(detail["date"], detail["run_time"])
    ]
    detail["residual_kw"] = detail["active_power_kw"] - detail["final_forecast_kw"]

    return detail


def train_past_only(detail, day):
    """
    LightGBM residual model trained only on days before `day`.
    Returns None when there is not yet enough history (the
    production min-training-days gate) - callers then apply no
    correction, exactly as production would.
    """

    past_days = [d for d in sorted(detail["date"].unique()) if d < day]

    if len(past_days) < MIN_TRAINING_DAYS:
        return None

    train = detail[detail["date"].isin(past_days)]
    train_set = lgb.Dataset(train[FEATURES], label=train["residual_kw"])

    return lgb.train(TRAINING_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)


def main():

    detail = pd.read_csv("outputs/reports/backtest_detail.csv", parse_dates=["timestamp"])
    processed = pd.read_csv("data/processed/processed_data.csv", parse_dates=["timestamp"])

    clearsky = ClearSkyModel()
    kt_lookup = compute_kt_now_lookup(processed, clearsky)
    detail = build_features(detail, kt_lookup)

    processed["date"] = processed["timestamp"].dt.date.astype(str)

    rows = []

    for day in REPORT_DAYS:

        model = train_past_only(detail, day)

        day_proc = processed[processed["date"] == day]
        actual_full_mwh = (day_proc["active_power_kw"].clip(lower=0) * 0.25).sum() / 1000

        run_energies = []
        run_deviations = []

        for run_time in RUN_TIMES:

            run_rows = detail[(detail["date"] == day) & (detail["run_time"] == run_time)]
            if run_rows.empty:
                continue

            correction = (
                model.predict(run_rows[FEATURES]) if model is not None else 0.0
            )
            predicted = np.clip(
                run_rows["final_forecast_kw"].to_numpy() + correction, 0, CAPACITY_KW
            )
            actual = run_rows["active_power_kw"].to_numpy()

            run_deviations.append(np.mean(np.abs(predicted - actual)) / CAPACITY_KW * 100)

            run_ts = pd.Timestamp(f"{day} {run_time}")
            morning_mwh = (
                day_proc[day_proc["timestamp"] <= run_ts]["active_power_kw"].clip(lower=0)
                * 0.25
            ).sum() / 1000
            forecast_mwh = predicted.sum() * 0.25 / 1000
            run_energies.append(morning_mwh + forecast_mwh)

        rows.append({
            "Date": day,
            "Predicted (MWh)": float(np.mean(run_energies)),
            "Actual (MWh)": actual_full_mwh,
            "Accuracy %": 100 - float(np.mean(run_deviations))
        })

    report = pd.DataFrame(rows)
    line = "=" * 70

    print("\n" + line)
    print(" SIRMOUR 5.1 MW  -  AI METER FORECAST vs ACTUAL   (Jul 12-18, 2026)")
    print(line)
    print(report.to_string(
        index=False,
        formatters={
            "Predicted (MWh)": "{:.1f}".format,
            "Actual (MWh)": "{:.1f}".format,
            "Accuracy %": "{:.1f}".format
        }
    ))
    print(line)

    overall_acc = report["Accuracy %"].mean()
    total_pred = report["Predicted (MWh)"].sum()
    total_act = report["Actual (MWh)"].sum()

    print(f" OVERALL forecast accuracy       : {overall_acc:.1f} %")
    print(f" Total predicted energy          : {total_pred:.1f} MWh")
    print(f" Total actual energy             : {total_act:.1f} MWh")
    print(line)

    report.to_csv("outputs/reports/days_12_18_accuracy.csv", index=False)
    print(" Saved: outputs/reports/days_12_18_accuracy.csv")

    image_path = render_image(report, overall_acc, total_pred, total_act)
    print(f" Saved: {image_path}\n")


def get_font(size, bold=False):

    try:
        font_file = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(f"C:/Windows/Fonts/{font_file}", size)

    except OSError:
        return ImageFont.load_default()


def render_image(report,
                  overall_acc,
                  total_pred,
                  total_act,
                  output_path="outputs/reports/days_12_18_accuracy.png"):
    """
    Renders the report as a clean table image (no charts/plots -
    just the numbers) suitable for sharing directly in a group.
    """

    columns = ["Date", "Predicted (MWh)", "Actual (MWh)", "Accuracy %"]
    col_widths = [160, 190, 170, 160]
    row_height = 42
    header_height = 48
    padding = 30

    table_width = sum(col_widths)
    width = table_width + padding * 2
    height = (
        70                              # title + subtitle
        + header_height
        + row_height * len(report)
        + 90                            # footer summary
        + padding
    )

    bg = (255, 255, 255)
    header_bg = (23, 52, 88)
    header_fg = (255, 255, 255)
    row_alt_bg = (238, 243, 248)
    text_color = (35, 35, 35)
    accent = (21, 128, 74)
    border = (205, 210, 216)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    title_font = get_font(23, bold=True)
    subtitle_font = get_font(14)
    header_font = get_font(15, bold=True)
    cell_font = get_font(15)
    summary_font = get_font(17, bold=True)
    summary_font_small = get_font(14)

    y = padding - 10

    title = "SIRMOUR 5.1 MW — AI METER FORECAST vs ACTUAL"
    subtitle = "July 12–18, 2026"

    title_w = draw.textlength(title, font=title_font)
    draw.text(((width - title_w) / 2, y), title, font=title_font, fill=(20, 20, 20))
    y += 30

    subtitle_w = draw.textlength(subtitle, font=subtitle_font)
    draw.text(((width - subtitle_w) / 2, y), subtitle, font=subtitle_font, fill=(110, 110, 110))
    y += 36

    table_x = padding
    table_top = y

    x = table_x
    draw.rectangle([table_x, y, table_x + table_width, y + header_height], fill=header_bg)
    for col_name, col_w in zip(columns, col_widths):
        text_w = draw.textlength(col_name, font=header_font)
        draw.text(
            (x + (col_w - text_w) / 2, y + (header_height - 15) / 2),
            col_name, font=header_font, fill=header_fg
        )
        x += col_w
    y += header_height

    for i, row in report.reset_index(drop=True).iterrows():

        if i % 2 == 1:
            draw.rectangle([table_x, y, table_x + table_width, y + row_height], fill=row_alt_bg)

        x = table_x
        values = [
            str(row["Date"]),
            f"{row['Predicted (MWh)']:.1f}",
            f"{row['Actual (MWh)']:.1f}",
            f"{row['Accuracy %']:.1f}%"
        ]
        for value, col_w in zip(values, col_widths):
            text_w = draw.textlength(value, font=cell_font)
            draw.text(
                (x + (col_w - text_w) / 2, y + (row_height - 15) / 2),
                value, font=cell_font, fill=text_color
            )
            x += col_w

        y += row_height

    draw.rectangle([table_x, table_top, table_x + table_width, y], outline=border, width=1)
    x = table_x
    for col_w in col_widths:
        x += col_w
        draw.line([(x, table_top), (x, y)], fill=border, width=1)

    y += 24

    summary_line_1 = f"Overall forecast accuracy: {overall_acc:.1f}%"
    line_1_w = draw.textlength(summary_line_1, font=summary_font)
    draw.text(((width - line_1_w) / 2, y), summary_line_1, font=summary_font, fill=accent)
    y += 28

    summary_line_2 = f"Total predicted: {total_pred:.1f} MWh    Total actual: {total_act:.1f} MWh"
    line_2_w = draw.textlength(summary_line_2, font=summary_font_small)
    draw.text(((width - line_2_w) / 2, y), summary_line_2, font=summary_font_small, fill=(90, 90, 90))

    image.save(output_path)

    return output_path


if __name__ == "__main__":
    main()
