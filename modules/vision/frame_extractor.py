"""
=========================================================
Solar Forecasting Project
Frame Extractor
=========================================================
Extracts representative frames from Windy videos
using uniform sampling and saves extraction metadata.
=========================================================
"""

from pathlib import Path
import json
import cv2

from modules.vision.frame_preprocessor import preprocess_frame


class FrameExtractor:

    def __init__(self, target_frames=10, preprocess=False):
        self.target_frames = target_frames
        # OpenCV cleanup (UI crop + contrast enhancement) before a frame
        # is saved - off by default so existing behaviour/caches are
        # unaffected; see modules/vision/frame_preprocessor.py.
        self.preprocess = preprocess

    # --------------------------------------------------

    def get_video_information(self, video_path):

        video = cv2.VideoCapture(str(video_path))

        if not video.isOpened():
            raise FileNotFoundError(
                f"Unable to open video:\n{video_path}"
            )

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        duration = total_frames / fps if fps > 0 else 0

        video.release()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration
        }

    # --------------------------------------------------

    def create_output_folder(self, video_path, output_root):

        video_name = Path(video_path).stem

        output_folder = Path(output_root) / video_name

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        return output_folder

    # --------------------------------------------------
    def count_readable_frames(self, video_path):

        video = cv2.VideoCapture(str(video_path))

        count = 0

        while True:

            success, _ = video.read()

            if not success:
                break

            count += 1

        video.release()

        return count
    
    def extract_frames(
        self,
        video_path,
        output_root,
        target_frames=None,
        start_fraction=0.0,
        end_fraction=1.0,
        preprocess=None
    ):
        """
        start_fraction / end_fraction TRIM the video to a clean
        window before sampling. Windy clips open with a few seconds
        of map loading and later loop the timeline back to the
        start; sampling only, say, 0.15-0.70 skips the loading and
        stops before the loop, so the frames are one clean forward
        sequence. Defaults (0.0-1.0) use the whole clip unchanged.
        """

        info = self.get_video_information(video_path)

        output_folder = self.create_output_folder(
            video_path,
            output_root
        )

        if target_frames is None:
            target_frames = self.target_frames

        if preprocess is None:
            preprocess = self.preprocess

        actual_frames = self.count_readable_frames(video_path)

        start_index = int(actual_frames * start_fraction)
        end_index = int(actual_frames * end_fraction)
        window = max(1, end_index - start_index)

        interval = max(
            1,
            window // target_frames
        )
        video = cv2.VideoCapture(str(video_path))

        frame_index = 0
        saved = 0
        frame_paths = []

        while True:

            success, frame = video.read()

            if not success:
                break

            in_window = start_index <= frame_index < end_index

            if in_window and (frame_index - start_index) % interval == 0:

                filename = (
                    output_folder /
                    f"frame_{saved:03d}.png"
                )

                if preprocess:
                    frame = preprocess_frame(frame)

                cv2.imwrite(str(filename), frame)

                frame_paths.append(filename)

                saved += 1

                if saved >= target_frames:
                    break

            frame_index += 1

        video.release()

        self.save_metadata(
            output_folder,
            video_path,
            info,
            saved,
            interval
        )

        return {
            "frame_folder": output_folder,
            "frame_paths": frame_paths,
            "frame_count": saved,
            "metadata_file": output_folder / "metadata.json"
        }

    # --------------------------------------------------

    def save_metadata(
        self,
        output_folder,
        video_path,
        info,
        saved,
        interval
    ):

        metadata = {
            "video_name": Path(video_path).name,
            "fps": info["fps"],
            "duration_seconds": round(info["duration"], 2),
            "total_frames": info["total_frames"],
            "frames_extracted": saved,
            "sampling_interval": interval,
            "sampling_method": "uniform_sampling"
        }

        metadata_file = (
            output_folder /
            "metadata.json"
        )

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )