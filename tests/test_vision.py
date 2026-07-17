"""
=========================================================
Solar Forecasting Project
Vision Pipeline Smoke Test
=========================================================
"""

from modules.vision.vision_module import VisionModule


def main():

    print("=" * 60)
    print("Starting Vision Pipeline Test...")
    print("=" * 60)

    vision = VisionModule()

    result = vision.analyze_video(

        video_path="data/windy/videos/Tab-7-9-2026_14_10.webm",

        output_folder="outputs/extracted_frames",

        target_frames=10

    )

    print("\n")

    print("=" * 60)
    print("Vision Pipeline Output")
    print("=" * 60)

    print(result)

    print("\n")

    print("=" * 60)
    print("Vision Pipeline Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()