"""
=========================================================
Solar Forecasting Project
Preprocessing Test
=========================================================
Runs the preprocessing pipeline on every daily historical
CSV independently (each day has its own gaps/coverage, so
resampling per day avoids interpolating across the
overnight gap between days), then concatenates the results
into one combined processed dataset.

This is the file the backtest and the tuning experiments read,
so it must be rebuilt for a plant before any of them are run:

    SOLAR_PLANT=kasipet python -m tests.test_preprocessing
=========================================================
"""

from pathlib import Path

import pandas as pd

from config.config import settings
from modules.preprocessing.preprocess import DataPreprocessor


def main():

    # -----------------------------
    # Folder containing daily historical CSVs, for the plant
    # SOLAR_PLANT names (default sirmour).
    # -----------------------------

    historical_folder = Path(settings["paths"]["historical_data"])

    csv_files = sorted(historical_folder.glob("*.csv"))

    # Create Preprocessor - the vendor's column names and date order
    # come from this plant's declared data_schema.
    preprocessor = DataPreprocessor()

    processed_days = []

    for csv_file in csv_files:

        print(f"\nProcessing {csv_file.name} ...")

        dataframe = preprocessor.preprocess(file_path=csv_file)

        print("Rows :", len(dataframe))

        processed_days.append(dataframe)

    combined = pd.concat(
        processed_days,
        ignore_index=True
    )

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Print results
    print("\n========== COMBINED DATA ==========\n")
    print(combined.head())

    print("\nTotal Rows :", len(combined))
    print("Days :", len(csv_files))
    print("Columns :", combined.columns.tolist())

    # Save processed data, into this plant's own processed folder
    output_path = Path(
        settings["paths"]["processed_data"]
    ) / "processed_data.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False)

    print("\nProcessed file saved successfully.")
    print(output_path)


if __name__ == "__main__":
    main()
