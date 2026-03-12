import os
import json
from collections import defaultdict
from tqdm import tqdm


class TrackSerializer:

    def __init__(self, tracks_dir):

        self.tracks_dir = tracks_dir

    def serialize_video_tracks(self, video_folder):

        track_file = os.path.join(self.tracks_dir, video_folder, "tracks.txt")

        if not os.path.exists(track_file):
            print(f"Skipping {video_folder}, tracks.txt not found")
            return

        tracks = defaultdict(list)

        with open(track_file, "r") as f:

            lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) < 6:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])

                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])

                tracks[track_id].append({
                    "frame": frame_id,
                    "bbox": [x1, y1, x2, y2]
                })

        output_file = os.path.join(self.tracks_dir, video_folder, "tracks.json")

        with open(output_file, "w") as f:
            json.dump(tracks, f, indent=4)

        print(f"Serialized tracks for {video_folder}")

    def run(self):

        video_folders = os.listdir(self.tracks_dir)

        print(f"Found {len(video_folders)} video folders")

        for video in tqdm(video_folders):

            video_path = os.path.join(self.tracks_dir, video)

            if not os.path.isdir(video_path):
                continue

            self.serialize_video_tracks(video)


if __name__ == "__main__":

    tracks_dir = "data/processed/tracks"

    serializer = TrackSerializer(tracks_dir)

    serializer.run()