import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.modeling.transformer_model import TemporalTransformer

FEATURES_DIR = "data/processed/features"
CHECKPOINT = "models/checkpoints/best_model.pth"


def load_video_features(video):

    video_dir = os.path.join(FEATURES_DIR, video)

    clips = sorted(os.listdir(video_dir))

    features = []

    for clip in clips:

        if clip.endswith(".npy"):
            f = np.load(os.path.join(video_dir, clip))
            features.append(f.squeeze())

    features = np.stack(features)

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def visualize(video):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalTransformer(num_classes=2)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()

    features = load_video_features(video).to(device)

    with torch.no_grad():

        outputs = model(features)
        preds = torch.argmax(outputs, dim=-1).cpu().numpy()[0]

    # plot
    plt.figure(figsize=(12,3))
    plt.plot(preds, linewidth=3)
    plt.title(f"Temporal Predictions: {video}")
    plt.xlabel("Clip index")
    plt.ylabel("Prediction (0=normal, 1=anomaly)")
    plt.ylim(-0.2,1.2)
    plt.grid()

    plt.show()


if __name__ == "__main__":

    video = os.listdir(FEATURES_DIR)[0]
    visualize(video)