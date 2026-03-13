import os
import torch
import numpy as np

from src.modeling.transformer_model import TemporalTransformer


FEATURES_DIR = "data/processed/features"
CHECKPOINT_DIR = "models/checkpoints"


def load_video_features(video_name):

    video_path = os.path.join(FEATURES_DIR, video_name)

    clips = sorted(os.listdir(video_path))

    features = []

    for clip in clips:

        if clip.endswith(".npy"):

            f = np.load(os.path.join(video_path, clip))
            features.append(f.squeeze())

    features = np.stack(features)

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def run_inference(model_path, video_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    features = load_video_features(video_name).to(device)

    with torch.no_grad():

        outputs = model(features)

        preds = torch.argmax(outputs, dim=-1)

    print("\nCheckpoint:", model_path)
    print("Video:", video_name)
    print("Predictions shape:", preds.shape)
    print("Predicted classes (first 50 clips):")
    print(preds[0][:50])


if __name__ == "__main__":

    video_name = os.listdir(FEATURES_DIR)[0]

    checkpoints = [
        "models/checkpoints/epoch_5.pth",
        "models/checkpoints/epoch_10.pth",
        "models/checkpoints/best_model.pth"
    ]

    for ckpt in checkpoints:

        if os.path.exists(ckpt):

            run_inference(ckpt, video_name)