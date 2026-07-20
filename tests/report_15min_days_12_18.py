"""
=========================================================
Solar Forecasting Project
Days 12-18  -  15-minute Interval Prediction Report
=========================================================
Instead of one number per day, this shows the predicted vs
actual meter generation for EVERY 15-minute interval of
each day (2026-07-12 .. 2026-07-18).

Each block uses the "frozen schedule" value - the forecast
from the most recent intraday run before that block (the
same idea the mentor's assessment doc describes), with the
walk-forward residual correction applied (day D corrected
only by days before D, once enough history exists).

No Windy videos exist for these days, so vision contributes
nothing here - these predictions come from the meter data +
clear-sky physics + Chronos + residual correction only.

Outputs:
  - outputs/reports/days_12_18_15min.csv   (all days)
  - outputs/reports/15min_<date>.png       (one per day)
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
from tests.report_days_12_18 import build_features, train_past_only


CAPACITY_KW = settings["plant"]["capacity_mw"] * 1000
INTERVAL_HOURS = settings["forecast"]["interval_minutes"] / 60
REPORT_DAYS = [f"2026-07-{d}" for d in range(12, 19)]


def frozen_schedule_for_day(detail, day, model):
    """
    Builds the day's 15-minute frozen schedule: for each
    block, the corrected forecast from the most recent run
    that predicted it, alongside the actual generation.
    """

    day_rows = detail[detail["date"] == day].copy()

    correction = model.predict(day_rows[FEATURES]) if model is not None else 0.0
    day_rows["predicted_kw"] = np.clip(
        day_rows["final_forecast_kw"].to_numpy() + correction, 0, CAPACITY_KW
    )

    # Most recent run per block = the frozen (final) forecast
    latest_idx = day_rows.groupby("timestamp")["run_time"].idxmax()
    frozen = day_rows.loc[latest_idx].sort_values("timestamp")

    return pd.DataFrame({
        "Time": frozen["timestamp"].dt.strftime("%H:%M"),
        "Predicted (kWh)": frozen["predicted_kw"].to_numpy() * INTERVAL_HOURS,
        "Actual (kWh)": frozen["active_power_kw"].clip(lower=0).to_numpy() * INTERVAL_HOURS
    })


def get_font(size, bold=False):
    try:
        name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def render_day_image(day, table, output_path):

    columns = ["Time", "Predicted (kWh)", "Actual (kWh)"]
    col_widths = [150, 210, 190]
    row_h = 27
    header_h = 42
    pad = 26
    table_w = sum(col_widths)
    width = table_w + pad * 2

    total_pred = table["Predicted (kWh)"].sum()
    total_act = table["Actual (kWh)"].sum()

    height = 78 + header_h + row_h * (len(table) + 1) + 60 + pad

    header_bg = (23, 52, 88)
    alt_bg = (238, 243, 248)
    total_bg = (223, 233, 226)
    border = (205, 210, 216)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)

    title_font = get_font(20, bold=True)
    sub_font = get_font(13)
    head_font = get_font(14, bold=True)
    cell_font = get_font(14)
    total_font = get_font(14, bold=True)

    y = pad - 8
    title = f"SIRMOUR 5.1 MW  —  15-min Prediction vs Actual"
    tw = d.textlength(title, font=title_font)
    d.text(((width - tw) / 2, y), title, font=title_font, fill=(20, 20, 20))
    y += 27
    sub = f"{day}  (energy per 15-minute interval, kWh)"
    sw = d.textlength(sub, font=sub_font)
    d.text(((width - sw) / 2, y), sub, font=sub_font, fill=(110, 110, 110))
    y += 30

    tx = pad
    top = y
    x = tx
    d.rectangle([tx, y, tx + table_w, y + header_h], fill=header_bg)
    for c, w in zip(columns, col_widths):
        cw = d.textlength(c, font=head_font)
        d.text((x + (w - cw) / 2, y + (header_h - 14) / 2), c, font=head_font, fill=(255, 255, 255))
        x += w
    y += header_h

    for i, row in table.reset_index(drop=True).iterrows():
        if i % 2 == 1:
            d.rectangle([tx, y, tx + table_w, y + row_h], fill=alt_bg)
        x = tx
        vals = [row["Time"], f"{row['Predicted (kWh)']:.0f}", f"{row['Actual (kWh)']:.0f}"]
        for v, w in zip(vals, col_widths):
            vw = d.textlength(v, font=cell_font)
            d.text((x + (w - vw) / 2, y + (row_h - 14) / 2), v, font=cell_font, fill=(35, 35, 35))
            x += w
        y += row_h

    # Total row
    d.rectangle([tx, y, tx + table_w, y + row_h], fill=total_bg)
    x = tx
    totals = ["TOTAL", f"{total_pred:.0f}", f"{total_act:.0f}"]
    for v, w in zip(totals, col_widths):
        vw = d.textlength(v, font=total_font)
        d.text((x + (w - vw) / 2, y + (row_h - 14) / 2), v, font=total_font, fill=(21, 90, 52))
        x += w
    y += row_h

    d.rectangle([tx, top, tx + table_w, y], outline=border, width=1)
    x = tx
    for w in col_widths:
        x += w
        d.line([(x, top), (x, y)], fill=border, width=1)

    y += 18
    acc = 100 - abs(total_pred - total_act) / (CAPACITY_KW * INTERVAL_HOURS * len(table)) * 100
    note = f"Day energy — predicted {total_pred/1000:.1f} MWh vs actual {total_act/1000:.1f} MWh"
    nw = d.textlength(note, font=total_font)
    d.text(((width - nw) / 2, y), note, font=total_font, fill=(90, 90, 90))

    img.save(output_path)
    return output_path


def main():

    detail = pd.read_csv("outputs/reports/backtest_detail.csv", parse_dates=["timestamp"])
    processed = pd.read_csv("data/processed/processed_data.csv", parse_dates=["timestamp"])

    clearsky = ClearSkyModel()
    kt_lookup = compute_kt_now_lookup(processed, clearsky)
    detail = build_features(detail, kt_lookup)

    all_rows = []
    saved_images = []

    for day in REPORT_DAYS:

        model = train_past_only(detail, day)
        table = frozen_schedule_for_day(detail, day, model)

        stamped = table.copy()
        stamped.insert(0, "Date", day)
        all_rows.append(stamped)

        img_path = f"outputs/reports/15min_{day}.png"
        render_day_image(day, table, img_path)
        saved_images.append(img_path)

        print(f"{day}: {len(table)} intervals  |  "
              f"predicted {table['Predicted (kWh)'].sum()/1000:.1f} MWh  vs  "
              f"actual {table['Actual (kWh)'].sum()/1000:.1f} MWh")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv("outputs/reports/days_12_18_15min.csv", index=False)

    print("\nSaved CSV : outputs/reports/days_12_18_15min.csv")
    for p in saved_images:
        print(f"Saved img : {p}")


if __name__ == "__main__":
    main()
