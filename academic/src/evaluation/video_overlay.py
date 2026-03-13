import os
import cv2
import torch
import numpy as np
import random

from src.modeling.transformer_model import TemporalTransformer


FEATURES_DIR = "data/processed/features"
FRAMES_DIR = "data/processed/frames"
CHECKPOINT = "models/checkpoints/best_model.pth"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_features(video):

    path = os.path.join(FEATURES_DIR, video)

    feats = []

    for f in sorted(os.listdir(path)):

        if f.endswith(".npy"):
            feats.append(np.load(os.path.join(path,f)).squeeze())

    feats = np.stack(feats)

    return torch.tensor(feats,dtype=torch.float32).unsqueeze(0)


def run(video):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalTransformer(num_classes=2)
    model.load_state_dict(torch.load(CHECKPOINT,map_location=device))
    model = model.to(device)
    model.eval()

    feats = load_features(video).to(device)

    with torch.no_grad():

        preds = torch.argmax(model(feats),dim=-1).cpu().numpy()[0]

    frame_dir = os.path.join(FRAMES_DIR, video)

    frames = sorted(os.listdir(frame_dir))

    for i,f in enumerate(frames):

        out_path = os.path.join(OUTPUT_DIR, f"{video}_prediction.mp4")

        first_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
        h, w, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 25, (w, h))

        for i,f in enumerate(frames):

            frame = cv2.imread(os.path.join(frame_dir,f))

            clip_index = min(i//16, len(preds)-1)

            label = preds[clip_index]

            text = "ANOMALY" if label==1 else "NORMAL"
            color = (0,0,255) if label==1 else (0,255,0)

            cv2.putText(frame,text,(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,color,2)

            writer.write(frame)

            cv2.imshow("prediction",frame)

            if cv2.waitKey(20)==27:
                break

        writer.release()
        cv2.destroyAllWindows()

        print("Saved output video:", out_path)


if __name__=="__main__":

    videos = [
        v for v in os.listdir(FEATURES_DIR)
        if os.path.isdir(os.path.join(FEATURES_DIR, v))
    ]

    video = random.choice(videos)

    print("\nSelected video:", video)

    run(video)