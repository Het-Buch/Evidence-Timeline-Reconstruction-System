import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:

    def __init__(self, frames_dir, detections_dir, output_dir):

        self.frames_dir = frames_dir
        self.detections_dir = detections_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_cosine_distance=0.3
        )

    def load_detections(self, detection_file):
        """
        Load YOLO detections from txt file
        Format:
        class confidence x1 y1 x2 y2
        """

        detections = []

        if not os.path.exists(detection_file):
            return detections

        with open(detection_file, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 6:
                continue

            class_id = int(parts[0])
            conf = float(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])

            width = x2 - x1
            height = y2 - y1

            detections.append(([x1, y1, width, height], conf, class_id))

        return detections

    def process_video(self, video_name):

        frame_folder = os.path.join(self.frames_dir, video_name)
        detection_folder = os.path.join(self.detections_dir, video_name)

        output_folder = os.path.join(self.output_dir, video_name)
        os.makedirs(output_folder, exist_ok=True)

        track_file = os.path.join(output_folder, "tracks.txt")

        frame_files = sorted(os.listdir(frame_folder))

        with open(track_file, "w") as out:

            for frame_idx, frame_file in enumerate(tqdm(frame_files)):

                frame_path = os.path.join(frame_folder, frame_file)
                det_file = os.path.join(
                    detection_folder,
                    frame_file.replace(".jpg", ".txt")
                )

                frame = cv2.imread(frame_path)

                detections = self.load_detections(det_file)

                tracks = self.tracker.update_tracks(detections, frame=frame)

                for track in tracks:

                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id

                    l, t, r, b = track.to_ltrb()

                    out.write(
                        f"{frame_idx} {track_id} {l:.2f} {t:.2f} {r:.2f} {b:.2f}\n"
                    )

    def run(self):

        videos = os.listdir(self.frames_dir)

        print(f"Found {len(videos)} video folders")

        for video in videos:

            print(f"\nTracking objects in {video}")

            self.process_video(video)


if __name__ == "__main__":

    frames_dir = "data/processed/frames"
    detections_dir = "data/processed/annotations"
    output_dir = "data/processed/tracks"

    tracker = ObjectTracker(
        frames_dir,
        detections_dir,
        output_dir
    )

    tracker.run()