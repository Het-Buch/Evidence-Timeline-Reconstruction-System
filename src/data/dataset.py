"""
Dataset & DataLoader
=====================
THREE datasets:

1. VideoLevelDataset  ← PRIMARY (fixed with consistent video augmentation)
   Each sample = ONE VIDEO (not one frame).
   KEY FIX: all frames in a video now receive the SAME spatial transformation
   (same crop position, same flip). Previously each frame was augmented
   independently, which broke temporal consistency and let the model cheat.

2. SequenceDataset    ← For CNN-LSTM / CNN-Transformer (temporal localization)
3. UCFCrimeDataset    ← Legacy frame-level (evaluate.py / xai.py only)
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── Normalization constants ───────────────────────────────────
# ImageNet: used by all 2D models (EfficientNet, ConvNeXt, Swin, etc.)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Kinetics-400: used by 3D video models pretrained on Kinetics
# (R2Plus1D-18, R3D-18, S3D).  Using ImageNet stats here breaks
# pretrained feature alignment and severely hurts transfer learning.
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145,  0.216989]

def build_group_mapping(grouped_classes: list, original_classes: list) -> dict:
    """
    Returns {original_class_idx: group_idx} mapping.
    grouped_classes: list of {name, classes} dicts from config.
    """
    mapping = {}
    for group_idx, group in enumerate(grouped_classes):
        for cls_name in group["classes"]:
            if cls_name in original_classes:
                orig_idx = original_classes.index(cls_name)
                mapping[orig_idx] = group_idx
    return mapping


def get_group_names(grouped_classes: list) -> list:
    return [g["name"] for g in grouped_classes]


def _get_normalize(norm_type: str) -> T.Normalize:
    if norm_type == "kinetics":
        return T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ════════════════════════════════════════════════════════════
# Consistent Video Transform (KEY FIX)
# ════════════════════════════════════════════════════════════
class ConsistentVideoTransform:
    """
    Applies the SAME spatial transformation to every frame in a video.

    Why this matters:
      The original code applied T.RandomCrop, T.RandomHorizontalFlip etc.
      INDEPENDENTLY to each frame. This meant frame 0 might be left-cropped
      and frame 1 right-cropped — breaking spatial consistency.

      For temporal models (CNN-LSTM, 3D CNNs) this destroys motion signals.
      For VideoLevelModel this creates inconsistent frame sets that the
      model can't compare meaningfully.

    Strategy:
      - Sample crop/flip/rotation params ONCE per video call
      - Apply SAME spatial params to all frames
      - Apply INDEPENDENT color jitter per frame (natural illumination variation)

    norm_type: "imagenet" (default, 2D models) or "kinetics" (3D video models).
      R2Plus1D / S3D were pretrained on Kinetics — wrong normalization breaks
      feature transfer completely.
    """

    def __init__(self, img_size: int, norm_type: str = "imagenet"):
        self.img_size = img_size
        self.pad_size = img_size + 64
        self.normalize    = _get_normalize(norm_type)
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15
        )
        # RandomErasing: masks a random patch per frame — simulates occlusions,
        # prevents the model from relying on any single spatial region.
        self.eraser = T.RandomErasing(p=0.25, scale=(0.02, 0.15),
                                      ratio=(0.3, 3.0), value=0)

    def __call__(self, pil_imgs: list) -> list:
        """
        Args:
            pil_imgs: list of PIL Images from the same video
        Returns:
            list of normalized tensors with consistent spatial transform
        """
        sz  = self.img_size
        pad = self.pad_size

        # Resize all frames to padded size first
        resized = [img.resize((pad, pad), Image.BILINEAR) for img in pil_imgs]

        # Sample consistent spatial params ONCE for the whole video
        i, j, h, w = T.RandomCrop.get_params(resized[0], output_size=(sz, sz))
        do_hflip = random.random() < 0.5
        do_vflip = False          # Surveillance cameras are never upside-down
        angle    = random.uniform(-8, 8)   # Reduced from ±20: cameras are nearly fixed
        do_gray  = random.random() < 0.1

        result = []
        for img in resized:
            # --- Consistent spatial transforms (same for all frames) ---
            img = TF.crop(img, i, j, h, w)
            if do_hflip:
                img = TF.hflip(img)
            if do_vflip:
                img = TF.vflip(img)
            img = TF.rotate(img, angle)

            # --- Per-frame color augmentation (natural video variation) ---
            if do_gray:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)
            img = self.color_jitter(img)
            # Stochastic blur: simulates camera shake / focus issues
            if random.random() < 0.2:
                ks = random.choice([3, 5])
                img = TF.gaussian_blur(img, kernel_size=ks, sigma=(0.1, 2.0))

            img = TF.to_tensor(img)
            img = self.normalize(img)
            img = self.eraser(img)   # random patch masking (per-frame)
            result.append(img)

        return result


def get_val_transform(img_size: int = 224, norm_type: str = "imagenet"):
    """Clean deterministic transform for val/test — no augmentation."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        _get_normalize(norm_type),
    ])


# Keep backwards-compatible name for legacy code
def get_transforms(split: str, img_size: int = 224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=15),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize,
        ])


def get_advanced_augmentation(img_size: int = 224):
    """Legacy: kept for UCFCrimeDataset / evaluate.py."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((img_size + 64, img_size + 64)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        T.RandomRotation(degrees=20),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        normalize,
    ])


# ── Annotation Parser ────────────────────────────────────────
def parse_temporal_annotations(annotation_path: str) -> dict:
    annotations = {}
    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            stem = Path(parts[0]).stem
            cls  = parts[1]
            s1, e1 = int(parts[2]), int(parts[3])
            s2, e2 = int(parts[4]), int(parts[5])
            segments = []
            if s1 != -1: segments.append((s1, e1))
            if s2 != -1: segments.append((s2, e2))
            annotations[stem] = {"class": cls, "segments": segments}
    return annotations


def get_frame_labels(total_frames: int, segments: list,
                     video_fps: float = 30.0, extracted_fps: int = 3) -> np.ndarray:
    labels = np.zeros(total_frames, dtype=np.int64)
    ratio  = video_fps / extracted_fps
    for native_start, native_end in segments:
        ext_start = max(0, int((native_start - 1) / ratio))
        ext_end   = min(total_frames, int(native_end / ratio) + 1)
        labels[ext_start:ext_end] = 1
    return labels


# ════════════════════════════════════════════════════════════
# Dataset 1: VIDEO-LEVEL — PRIMARY TRAINING DATASET
# ════════════════════════════════════════════════════════════
class VideoLevelDataset(Dataset):
    """
    Each sample = one video.
    Returns: (frames_tensor, class_label)
      frames_tensor: (N, 3, H, W) — N consistently augmented frames
      class_label:   int

    Key fix: ConsistentVideoTransform ensures all N frames receive the
    same spatial augmentation (crop/flip/rotation), while still varying
    per epoch (different random params each call). This preserves spatial
    consistency while providing regularization.
    """

    def __init__(self, split: str,
                 splits_path: str = "data/splits/splits.json",
                 img_size: int = 224,
                 frames_per_video: int = 16,
                 input_type: str = "rgb",
                 norm_type: str = "imagenet",
                 sampling: str = "sparse",
                 group_mapping: dict = None):
        """
        sampling:
          "sparse"  (default, 2D models) — frames spread evenly across full video
                    with random jitter. Captures global video context.
          "dense"   (3D video models: R2Plus1D, S3D) — N consecutive frames from
                    one randomly chosen clip within the video.
                    REQUIRED for 3D models: their temporal convolutions detect
                    motion between adjacent frames. Sparse sampling gives them
                    frames seconds apart — no motion signal exists between them.
        group_mapping: {orig_class_idx: group_idx} from build_group_mapping().
          When provided, __getitem__ remaps 14-class labels → group labels.
        """
        self.frames_per_video = frames_per_video
        self.img_size     = img_size
        self.split        = split
        self.sampling     = sampling
        self.group_mapping = group_mapping

        # Train: consistent spatial augmentation across frames
        # Val/Test: clean deterministic transform
        # norm_type: "imagenet" for 2D models, "kinetics" for R2Plus1D/S3D
        if split == "train":
            self.video_transform = ConsistentVideoTransform(img_size, norm_type)
            self.use_consistent  = True
        else:
            self.frame_transform = get_val_transform(img_size, norm_type)
            self.use_consistent  = False

        with open(splits_path) as f:
            entries = json.load(f)[split]

        self.videos = []
        for e in entries:
            frame_dir = Path(e["flow_dir"] if input_type == "optical_flow"
                             else e["frame_dir"])
            ext = "*.png" if input_type == "optical_flow" else "*.jpg"
            if not frame_dir.exists():
                continue
            frames = sorted(frame_dir.glob(ext))
            if len(frames) < 2:
                continue
            self.videos.append({
                "frames":    [str(f) for f in frames],
                "class_idx": e["class_idx"],
                "class":     e["class"],
                "video_id":  e["video_id"],
            })

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int):
        v      = self.videos[idx]
        frames = v["frames"]
        n      = len(frames)

        if n <= self.frames_per_video:
            selected = frames

        elif self.sampling == "dense":
            # DENSE: N consecutive frames from one clip.
            # Required for 3D models — temporal convolutions need adjacent
            # frames with real motion, not frames seconds apart.
            max_start = n - self.frames_per_video
            if self.split == "train":
                # Training: random start = data augmentation (different clip each epoch)
                start = random.randint(0, max_start)
            else:
                # Val/Test: center clip = DETERMINISTIC, so metrics are stable
                # across epochs and comparable across models.
                start = max_start // 2
            selected = frames[start : start + self.frames_per_video]

        else:
            # SPARSE (default): frames spread evenly across the full video.
            # Train: random jitter within each segment (augmentation).
            # Val/Test: deterministic center-of-segment — critical for stable
            # metrics. Random jitter at val causes noisy F1/acc each epoch,
            # making early stopping and best-model saving unreliable.
            step   = n / self.frames_per_video
            chosen = []
            for i in range(self.frames_per_video):
                seg_start = int(i * step)
                seg_end   = min(int((i + 1) * step), n - 1)
                if self.split == "train":
                    chosen.append(frames[random.randint(seg_start, seg_end)])
                else:
                    chosen.append(frames[(seg_start + seg_end) // 2])
            selected = chosen

        # Load all PIL images first
        pil_imgs = []
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for fp in selected:
            try:
                pil_imgs.append(Image.open(fp).convert("RGB"))
            except Exception:
                pil_imgs.append(Image.fromarray(blank))

        # Pad if fewer frames than requested
        while len(pil_imgs) < self.frames_per_video:
            pil_imgs.append(pil_imgs[-1])
        pil_imgs = pil_imgs[:self.frames_per_video]

        # Apply transforms
        if self.use_consistent:
            # ConsistentVideoTransform returns list of tensors
            tensors = self.video_transform(pil_imgs)
        else:
            tensors = [self.frame_transform(img) for img in pil_imgs]

        frames_tensor = torch.stack(tensors)                      # (N, 3, H, W)
        raw_idx = v["class_idx"]
        label_int = (self.group_mapping.get(raw_idx, 0)
                     if self.group_mapping is not None else raw_idx)
        label = torch.tensor(label_int, dtype=torch.long)
        return frames_tensor, label

    def get_class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for v in self.videos:
            idx = v["class_idx"]
            if self.group_mapping is not None:
                idx = self.group_mapping.get(idx, 0)
            counts[idx] += 1
        return counts


# ════════════════════════════════════════════════════════════
# Dataset 2: SEQUENCE-LEVEL — For CNN-LSTM / CNN-Transformer
# ════════════════════════════════════════════════════════════
class SequenceDataset(Dataset):
    """
    Temporal sequence dataset. Each sample = sliding window of T frames.
    Returns per-frame binary labels from temporal annotations.
    """

    def __init__(self, split: str,
                 splits_path: str = "data/splits/splits.json",
                 annotation_path: str = "data/splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
                 seq_len: int = 16, stride: int = 8,
                 img_size: int = 224, extracted_fps: int = 3,
                 group_mapping: dict = None):
        self.seq_len      = seq_len
        self.img_size     = img_size
        self.group_mapping = group_mapping
        self.transform    = get_transforms(split, img_size)

        annotations = {}
        if Path(annotation_path).exists():
            annotations = parse_temporal_annotations(annotation_path)

        with open(splits_path) as f:
            entries = json.load(f)[split]

        self.sequences = []
        for e in entries:
            frame_dir = Path(e["frame_dir"])
            if not frame_dir.exists():
                continue
            frames = sorted(frame_dir.glob("*.jpg"))
            if len(frames) < seq_len:
                continue

            vid_stem = e["video_id"]
            cls_name = e["class"]
            cls_idx  = e["class_idx"]

            if vid_stem in annotations:
                segs = annotations[vid_stem]["segments"]
                frame_labels = get_frame_labels(len(frames), segs, 30.0, extracted_fps)
            else:
                val = 0 if cls_name == "Normal" else 1
                frame_labels = np.full(len(frames), val, dtype=np.int64)

            for start in range(0, len(frames) - seq_len + 1, stride):
                self.sequences.append({
                    "frames":       [str(f) for f in frames[start:start + seq_len]],
                    "frame_labels": frame_labels[start:start + seq_len].copy(),
                    "class_idx":    cls_idx,
                    "video_id":     vid_stem,
                    "start_frame":  start,
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def get_class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for s in self.sequences:
            idx = s["class_idx"]
            if self.group_mapping is not None:
                idx = self.group_mapping.get(idx, 0)
            if idx < num_classes:
                counts[idx] += 1
        return counts

    def __getitem__(self, idx: int):
        item = self.sequences[idx]
        imgs = []
        for fp in item["frames"]:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.fromarray(
                    np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            imgs.append(self.transform(img))
        seq_tensor   = torch.stack(imgs)
        label_tensor = torch.tensor(item["frame_labels"], dtype=torch.long)
        cls_idx = item["class_idx"]
        if self.group_mapping is not None:
            cls_idx = self.group_mapping.get(cls_idx, 0)
        class_tensor = torch.tensor(cls_idx, dtype=torch.long)
        return seq_tensor, label_tensor, class_tensor


# ════════════════════════════════════════════════════════════
# Dataset 3: FRAME-LEVEL — Legacy, used by evaluate.py/xai.py
# ════════════════════════════════════════════════════════════
class UCFCrimeDataset(Dataset):
    """Frame-level dataset. Kept for evaluation and XAI only."""

    def __init__(self, split: str,
                 splits_path: str = "data/splits/splits.json",
                 input_type: str = "rgb", img_size: int = 224,
                 transform=None, frames_per_video: int = None,
                 advanced_aug: bool = False,
                 binary: bool = False,
                 normal_class_idx: int = 13,
                 group_mapping: dict = None):
        """
        binary=True: collapse all crime classes → label=1, Normal → label=0.
        This is the key to achieving 65-75% accuracy on UCF-Crime.
        14-class is too hard (~55 videos/class, weak frame labels).
        Binary is sufficient for timeline reconstruction (anomaly detection).
        """
        self.img_size     = img_size
        self.binary       = binary
        self.normal_class_idx = normal_class_idx
        self.group_mapping = group_mapping  # {orig_idx: group_idx} or None
        with open(splits_path) as f:
            entries = json.load(f)[split]

        self.transform = (transform or
                          (get_advanced_augmentation(img_size)
                           if advanced_aug and split == "train"
                           else get_transforms(split, img_size)))

        self.samples = []
        for e in entries:
            frame_dir = Path(e["flow_dir"] if input_type == "optical_flow"
                             else e["frame_dir"])
            ext = "*.png" if input_type == "optical_flow" else "*.jpg"
            if not frame_dir.exists():
                continue
            frames = sorted(frame_dir.glob(ext))
            if not frames:
                continue
            if frames_per_video and len(frames) > frames_per_video:
                if split == "train":
                    frames = random.sample(frames, frames_per_video)
                else:
                    step = len(frames) / frames_per_video
                    frames = [frames[int(i * step)] for i in range(frames_per_video)]
            label = e["class_idx"]
            if binary:
                label = 0 if label == normal_class_idx else 1
            elif group_mapping is not None:
                label = group_mapping.get(label, 0)
            for fp in frames:
                self.samples.append((str(fp), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            img = Image.fromarray(
                np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        return self.transform(img), label

    def get_class_sample_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, lbl in self.samples:
            counts[lbl] += 1
        return counts


# ── DataLoader factories ─────────────────────────────────────
def get_dataloaders(cfg: dict, model_cfg: dict = None):
    """
    Returns video-level DataLoaders if video_level_training=true (default),
    otherwise falls back to frame-level.
    """
    img_size    = (model_cfg.get("img_size", cfg["frames"]["img_size"])
                   if model_cfg else cfg["frames"]["img_size"])
    input_type  = (model_cfg.get("input_type", "rgb")
                   if model_cfg else "rgb")
    # norm_type: "kinetics" for R2Plus1D/S3D (Kinetics-400 pretrained),
    #            "imagenet" for all 2D models (EfficientNet, ConvNeXt, Swin, etc.)
    norm_type   = (model_cfg.get("normalization", "imagenet")
                   if model_cfg else "imagenet")
    # sampling: "dense" for 3D video models (consecutive frames for motion detection),
    #           "sparse" for 2D models (spread across full video for global context)
    sampling    = (model_cfg.get("sampling", "sparse")
                   if model_cfg else "sparse")
    splits_path = str(Path(cfg["dataset"]["splits_dir"]) / "splits.json")
    num_classes = cfg["dataset"]["num_classes"]
    # Per-model batch_size override
    bs          = (model_cfg.get("batch_size", cfg["training"]["batch_size"])
                   if model_cfg else cfg["training"]["batch_size"])
    nw          = cfg["training"]["num_workers"]
    fpv         = cfg["training"].get("frames_per_video", 16)
    video_level = cfg["training"].get("video_level_training", True)
    # Per-model override: frame_level=true → UCFCrimeDataset (all frames from every
    # video as individual samples). Gives ~50x more training samples (38,500 vs 770)
    # which is the primary lever for accuracy on small video datasets.
    if model_cfg and model_cfg.get("frame_level", False):
        video_level = False

    if video_level:
        # Build group_mapping for video-level models (e.g. R2Plus1D with grouped=true)
        vl_group_mapping = None
        vl_num_classes   = num_classes
        grouped = bool(model_cfg.get("grouped", False)) if model_cfg else False
        if grouped and cfg["dataset"].get("grouped_classes"):
            orig_classes     = cfg["dataset"]["classes"]
            vl_group_mapping = build_group_mapping(
                cfg["dataset"]["grouped_classes"], orig_classes)
            vl_num_classes   = len(cfg["dataset"]["grouped_classes"])

        train_ds = VideoLevelDataset("train", splits_path, img_size, fpv, input_type, norm_type, sampling, vl_group_mapping)
        val_ds   = VideoLevelDataset("val",   splits_path, img_size, fpv, input_type, norm_type, sampling, vl_group_mapping)
        test_ds  = VideoLevelDataset("test",  splits_path, img_size, fpv, input_type, norm_type, sampling, vl_group_mapping)

        counts = train_ds.get_class_counts(vl_num_classes)
        w = np.array([1.0 / (counts[
                          (vl_group_mapping.get(v["class_idx"], 0)
                           if vl_group_mapping else v["class_idx"])] + 1e-8)
                      for v in train_ds.videos], dtype=np.float32)
        sampler = WeightedRandomSampler(torch.from_numpy(w), len(train_ds), True)

        train_dl = DataLoader(train_ds, bs, sampler=sampler,
                              num_workers=nw, pin_memory=True)
        val_dl   = DataLoader(val_ds, bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
        test_dl  = DataLoader(test_ds, bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    else:
        # frame_level mode: treat each frame as an independent sample.
        # fl_frames: how many frames to randomly sample per video per epoch.
        #   None = all frames (~50/video). 15 = fast training, still diverse.
        # Each epoch resamples randomly → model sees different frames → augmentation.
        fl_frames = (model_cfg.get("fl_frames", None) if model_cfg else None)
        fl_transform_train = get_transforms("train", img_size)
        fl_transform_val   = get_val_transform(img_size, norm_type)
        binary       = bool(model_cfg.get("binary", False)) if model_cfg else False
        grouped      = bool(model_cfg.get("grouped", False)) if model_cfg else False
        orig_classes = cfg["dataset"]["classes"]
        normal_idx   = orig_classes.index("Normal")
        group_mapping = None
        if grouped and cfg["dataset"].get("grouped_classes"):
            group_mapping = build_group_mapping(
                cfg["dataset"]["grouped_classes"], orig_classes)

        train_ds = UCFCrimeDataset("train", splits_path, input_type, img_size,
                                   transform=fl_transform_train,
                                   frames_per_video=fl_frames,
                                   binary=binary, normal_class_idx=normal_idx,
                                   group_mapping=group_mapping)
        val_ds   = UCFCrimeDataset("val",   splits_path, input_type, img_size,
                                   transform=fl_transform_val,
                                   frames_per_video=8,
                                   binary=binary, normal_class_idx=normal_idx,
                                   group_mapping=group_mapping)
        test_ds  = UCFCrimeDataset("test",  splits_path, input_type, img_size,
                                   transform=fl_transform_val,
                                   frames_per_video=None,
                                   binary=binary, normal_class_idx=normal_idx,
                                   group_mapping=group_mapping)
        counts   = train_ds.get_class_sample_counts(num_classes)
        w = np.array([1.0 / (counts[lbl] + 1e-8)
                      for _, lbl in train_ds.samples], dtype=np.float32)
        sampler  = WeightedRandomSampler(torch.from_numpy(w), len(train_ds), True)
        train_dl = DataLoader(train_ds, bs, sampler=sampler,
                              num_workers=nw, pin_memory=True)
        val_dl   = DataLoader(val_ds, bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
        test_dl  = DataLoader(test_ds, bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    return train_dl, val_dl, test_dl


def get_sequence_dataloaders(cfg: dict, model_cfg: dict = None):
    """Sequence DataLoaders for CNN-LSTM / CNN-Transformer."""
    splits_path = str(Path(cfg["dataset"]["splits_dir"]) / "splits.json")
    ann_path    = str(Path(cfg["dataset"]["splits_dir"]) /
                      "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    img_size    = cfg["frames"]["img_size"]
    seq_len     = cfg["training"].get("seq_len",       16)
    stride_tr   = cfg["training"].get("stride_train",   8)
    stride_vt   = cfg["training"].get("stride_val",    16)
    bs          = max(1, cfg["training"]["batch_size"] // 4)
    nw          = cfg["training"]["num_workers"]
    pf          = cfg["training"].get("prefetch_factor", 2) if nw > 0 else None
    ext_fps     = cfg["frames"]["fps"]

    # Build group_mapping if model is grouped
    gm = None
    if model_cfg and model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        gm = build_group_mapping(cfg["dataset"]["grouped_classes"],
                                 cfg["dataset"]["classes"])

    train_ds = SequenceDataset("train", splits_path, ann_path,
                               seq_len, stride_tr, img_size, ext_fps, gm)
    val_ds   = SequenceDataset("val",   splits_path, ann_path,
                               seq_len, stride_vt, img_size, ext_fps, gm)
    test_ds  = SequenceDataset("test",  splits_path, ann_path,
                               seq_len, stride_vt, img_size, ext_fps, gm)

    return (
        DataLoader(train_ds, bs, shuffle=True,  num_workers=nw, pin_memory=True,
                   prefetch_factor=pf),
        DataLoader(val_ds,   bs, shuffle=False, num_workers=nw, pin_memory=True,
                   prefetch_factor=pf),
        DataLoader(test_ds,  bs, shuffle=False, num_workers=nw, pin_memory=True,
                   prefetch_factor=pf),
    )
