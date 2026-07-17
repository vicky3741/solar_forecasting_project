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
=========================================================
"""

from pathlib import Path

import pandas as pd

from modules.preprocessing.preprocess import DataPreprocessor


def main():

    # -----------------------------
    # Folder containing daily historical CSVs
    # -----------------------------

    historical_folder = Path("data/historical")

    csv_files = sorted(historical_folder.glob("*.csv"))

    # Required columns
    required_columns = [
        "TimeStamp"
    ]

    # Create Preprocessor
    preprocessor = DataPreprocessor()

    processed_days = []

    for csv_file in csv_files:

        print(f"\nProcessing {csv_file.name} ...")

        dataframe = preprocessor.preprocess(
            file_path=csv_file,
            required_columns=required_columns,
            timestamp_column="TimeStamp"
        )

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

    # Save processed data
    output_path = Path(
        "data/processed/processed_data.csv"
    )

    combined.to_csv(output_path, index=False)

    print("\nProcessed file saved successfully.")
    print(output_path)


if __name__ == "__main__":
    main()
