"""
=========================================================
Solar Forecasting Project
Preprocessing Test
=========================================================
Runs the preprocessing pipeline on a real dataset.
=========================================================
"""

from pathlib import Path

from modules.preprocessing.preprocess import DataPreprocessor


def main():

    # -----------------------------
    # Path to your historical CSV
    # -----------------------------

    csv_file = Path(
        "data/historical/2026_07_12_SOLAR_INV.csv"
    )

    # Required columns
    required_columns = [
        "TimeStamp"
    ]

    # Create Preprocessor
    preprocessor = DataPreprocessor()

    # Run preprocessing
    dataframe = preprocessor.preprocess(
        file_path=csv_file,
        required_columns=required_columns,
        timestamp_column="TimeStamp"
    )

    # Print results
    print("\n========== DATA ==========\n")
    print(dataframe.head())

    print("\nRows :", len(dataframe))
    print("\nColumns :", dataframe.columns.tolist())

    # Save processed data
    output_path = Path(
        "data/processed/processed_data.csv"
    )

    dataframe.to_csv(output_path, index=False)

    print("\nProcessed file saved successfully.")
    print(output_path)
    

if __name__ == "__main__":
    main()