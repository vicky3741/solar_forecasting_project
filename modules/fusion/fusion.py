"""
=========================================
Feature Fusion Module
=========================================
Combines numerical weather data and
vision-based weather features into one
dataset for forecasting.
=========================================
"""

import pandas as pd


class FeatureFusion:

    def __init__(self):
        print("Feature Fusion Module Initialized")
        
    def prepare_vision_features(self, vision_result):

        """
        Extracts only the weather features returned by
            the Vision module.
        """

        return vision_result["weather_features"]
    def vision_to_dataframe(self, vision_features):

        """
        Converts vision features dictionary
        into a single-row DataFrame.
        """

        return pd.DataFrame([vision_features])
    
    def fuse(self, numerical_df, vision_df):

        """
        Combine numerical data with vision data.
        """

        for column in vision_df.columns:

            numerical_df[column] = vision_df.iloc[0][column]

        return numerical_df