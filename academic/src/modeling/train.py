import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from src.modeling.transformer_model import TemporalTransformer


FEATURES_DIR = "data/processed/features"
CHECKPOINT_DIR = "models/checkpoints"

BATCH_SIZE = 4
EPOCHS = 100
PATIENCE = 10
NUM_CLASSES = 2
LR = 1e-4


# ---------------- DATASET ---------------- #

class FeatureDataset(Dataset):

    def __init__(self, features_root):

        self.samples = []

        for video in os.listdir(features_root):

            video_path = os.path.join(features_root, video)

            if not os.path.isdir(video_path):
                continue

            clips = sorted(os.listdir(video_path))

            feature_seq = []

            for clip in clips:

                if not clip.endswith(".npy"):
                    continue

                f = np.load(os.path.join(video_path, clip))
                feature_seq.append(f.squeeze())

            if len(feature_seq) == 0:
                continue

            feature_seq = np.stack(feature_seq)

            num_clips = len(feature_seq)

            labels = np.zeros(num_clips, dtype=np.int64)

            # Random anomaly window for anomaly videos
            if "Normal" not in video and "normal" not in video:

                start = np.random.randint(int(num_clips * 0.4), int(num_clips * 0.7))
                end = min(num_clips, start + int(num_clips * 0.3))

                labels[start:end] = 1

            self.samples.append((feature_seq, labels))

        print("Dataset loaded videos:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        features, labels = self.samples[idx]

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return features, labels


# ---------------- COLLATE ---------------- #

def collate_fn(batch):

    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_len = max(f.shape[0] for f in features)

    padded_features = []
    padded_labels = []

    for f, l in zip(features, labels):

        pad_len = max_len - f.shape[0]

        if pad_len > 0:

            pad_feat = torch.zeros(pad_len, f.shape[1])
            pad_lab = torch.zeros(pad_len, dtype=torch.long)

            f = torch.cat([f, pad_feat], dim=0)
            l = torch.cat([l, pad_lab], dim=0)

        padded_features.append(f)
        padded_labels.append(l)

    features = torch.stack(padded_features)
    labels = torch.stack(padded_labels)

    return features, labels


# ---------------- TRAINING ---------------- #

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training on:", device)

    dataset = FeatureDataset(FEATURES_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = TemporalTransformer(num_classes=NUM_CLASSES).to(device)

    # ---------- CLASS WEIGHTING ---------- #

    weights = torch.tensor([0.3, 0.7]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        # ---------- TRAIN ---------- #

        model.train()
        train_loss = 0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            outputs = outputs.view(-1, NUM_CLASSES)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # ---------- GRADIENT CLIPPING ---------- #

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------- VALIDATION ---------- #

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for features, labels in val_loader:

                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)

                outputs = outputs.view(-1, NUM_CLASSES)
                labels = labels.view(-1)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ---------- SAVE CHECKPOINT ---------- #

        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        )

        # ---------- EARLY STOPPING ---------- #

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print("\nEarly stopping triggered")
                break


if __name__ == "__main__":
    train()