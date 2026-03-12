import os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

class YOLODetector:

    def __init__(
        self,
        frames_root="data/processed/frames",
        output_root="data/processed/annotations",
        model_name="yolov8s.pt",
        conf_threshold=0.10,
        device="cuda"
    ):

        self.frames_root = frames_root
        self.output_root = output_root
        self.conf_threshold = conf_threshold

        os.makedirs(self.output_root, exist_ok=True)

        print("Loading YOLO model...")
        self.model = YOLO(model_name)

        if device == "cuda":
            self.model.to("cuda")

        print("Model loaded successfully")

    def process_video_frames(self, video_folder):

        video_path = os.path.join(self.frames_root, video_folder)
        output_video_dir = os.path.join(self.output_root, video_folder)

        os.makedirs(output_video_dir, exist_ok=True)

        frame_files = sorted(
            [f for f in os.listdir(video_path) if f.endswith(".jpg")]
        )

        for frame_file in frame_files:

            frame_path = os.path.join(video_path, frame_file)

            results = self.model(
                frame_path,
                conf=self.conf_threshold,
                device="cuda",
                verbose=False
            )

            detections = results[0].boxes

            annotation_file = os.path.join(
                output_video_dir,
                frame_file.replace(".jpg", ".txt")
            )

            with open(annotation_file, "w") as f:

                if detections is None:
                    continue

                for box in detections:

                    cls = int(box.cls)
                    conf = float(box.conf)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    f.write(
                        f"{cls} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n"
                    )

    def run_detection(self):

        video_folders = [
            d for d in os.listdir(self.frames_root)
            if os.path.isdir(os.path.join(self.frames_root, d))
        ]

        print(f"\nFound {len(video_folders)} video folders\n")

        for video in tqdm(video_folders):

            self.process_video_frames(video)

        print("\nDetection complete!")


if __name__ == "__main__":

    detector = YOLODetector(
        frames_root="data/processed/frames",
        output_root="data/processed/annotations",
        model_name="yolov8s.pt",
        conf_threshold=0.25,
        device="cuda"
    )

    detector.run_detection()