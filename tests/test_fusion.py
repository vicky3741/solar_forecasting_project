"""
=========================================
Feature Fusion Test
=========================================
"""

from modules.fusion.fusion import FeatureFusion


def main():

    fusion = FeatureFusion()

    # Sample Vision Output
    vision_result = {
        "weather_features": {
            "cloud_coverage": 55,
            "cloud_density": "low",
            "sun_visibility": 70,
            "rain_probability": 10,
            "weather_condition": "partly cloudy"
        }
    }

    # Step 1
    vision_features = fusion.prepare_vision_features(
        vision_result
    )

    print("\nExtracted Vision Features")
    print(vision_features)

    # Step 2
    vision_df = fusion.vision_to_dataframe(
        vision_features
    )

    print("\nVision DataFrame")
    print(vision_df)
    
    import pandas as pd

    # Sample Numerical Data
    numerical_df = pd.DataFrame({
        "timestamp": ["2026-07-12 14:10"],
        "active_power_kw": [520],
        "ghi_w_m2": [865],
        "poa_w_m2": [840]
    })

    # Fuse both DataFrames
    final_df = fusion.fuse(
        numerical_df,
        vision_df
    )

    print("\nFinal Fused DataFrame")
    print(final_df)


if __name__ == "__main__":
    main()