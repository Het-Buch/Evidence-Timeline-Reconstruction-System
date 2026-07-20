"""
STEP 1 — Data Preprocessing
============================
Run this FIRST before anything else.

What it does:
  1. Scans data/raw/<ClassName>/*.mp4 (or avi/mkv)
  2. Extracts frames at configured FPS → data/processed/frames/
  3. Computes Farneback optical flow → data/processed/optical_flow/
  4. Creates train/val/test splits → data/splits/
  5. Computes class weights for imbalanced training

Expected raw layout:
  data/raw/
    Abuse/          ← place UCF-Crime videos here
    Arrest/
    ...
    Normal/

Usage:
  python src/data/preprocess.py
  python src/data/preprocess.py --skip-flow   (skip optical flow, saves time)
  python src/data/preprocess.py --fps 2
"""

import os
import cv2
import json
import yaml
import argparse
import random
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def extract_frames(video_path: Path, out_dir: Path, fps: int, img_size: int) -> int:
    """Extract frames from a video at given FPS. Returns number of frames saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return 0

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0
    frame_interval = max(1, int(round(native_fps / fps)))

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame = cv2.resize(frame, (img_size, img_size))
            fname = out_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def compute_optical_flow(frames_dir: Path, out_dir: Path) -> int:
    """Compute Farneback optical flow between consecutive frames. Saves as 3-ch PNG (hue+magnitude)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if len(frame_files) < 2:
        return 0

    saved = 0
    prev_gray = cv2.cvtColor(cv2.imread(str(frame_files[0])), cv2.COLOR_BGR2GRAY)

    for i, fpath in enumerate(frame_files[1:], 1):
        curr_gray = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Encode flow as HSV image (hue=direction, value=magnitude)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*curr_gray.shape, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        fname = out_dir / f"flow_{i:05d}.png"
        cv2.imwrite(str(fname), flow_rgb)
        saved += 1
        prev_gray = curr_gray

    return saved


def scan_videos(raw_dir: Path, classes: list, max_per_class):
    """Returns dict: {class_name: [video_path, ...]}"""
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI"}
    class_videos = defaultdict(list)

    for cls in classes:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            log.warning(f"Class directory missing: {cls_dir}")
            continue
        videos = [p for p in cls_dir.iterdir() if p.suffix in video_exts]
        if max_per_class:
            videos = videos[:max_per_class]
        class_videos[cls] = sorted(videos)
        log.info(f"  {cls}: {len(videos)} videos found")

    return class_videos


def build_splits(class_videos: dict, train_r: float, val_r: float, test_r: float, seed: int):
    """Stratified split by class. Returns {split: [(video_id, class_name, frame_dir)]}"""
    splits = {"train": [], "val": [], "test": []}

    for cls, videos in class_videos.items():
        random.shuffle(videos)
        n = len(videos)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        train_v = videos[:n_train]
        val_v = videos[n_train:n_train + n_val]
        test_v = videos[n_train + n_val:]

        for v in train_v:
            splits["train"].append((v.stem, cls, str(v)))
        for v in val_v:
            splits["val"].append((v.stem, cls, str(v)))
        for v in test_v:
            splits["test"].append((v.stem, cls, str(v)))

    return splits


def compute_class_weights(class_videos: dict, classes: list):
    """Inverse-frequency class weights for imbalanced dataset."""
    counts = np.array([len(class_videos.get(c, [])) for c in classes], dtype=np.float32)
    total = counts.sum()
    weights = total / (len(classes) * counts + 1e-8)
    weights = weights / weights.mean()  # normalize around 1
    return {cls: float(w) for cls, w in zip(classes, weights)}


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    raw_dir = Path(cfg["dataset"]["raw_dir"])
    frames_dir = Path(cfg["frames"]["output_dir"])
    flow_dir = Path(cfg["optical_flow"]["output_dir"])
    splits_dir = Path(cfg["dataset"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    classes = cfg["dataset"]["classes"]
    img_size = cfg["frames"]["img_size"]
    fps = args.fps if args.fps else cfg["frames"]["fps"]
    max_per_class = cfg["dataset"]["max_videos_per_class"]

    log.info("=" * 60)
    log.info("STEP 1: DATA PREPROCESSING")
    log.info("=" * 60)

    # ── Scan videos ─────────────────────────────────────────
    log.info(f"\nScanning videos in: {raw_dir}")
    class_videos = scan_videos(raw_dir, classes, max_per_class)

    total = sum(len(v) for v in class_videos.values())
    if total == 0:
        log.error("No videos found! Check data/raw/ layout.")
        return
    log.info(f"Total videos: {total}")

    # ── Extract frames ───────────────────────────────────────
    frame_index = {}   # video_stem → frame_dir (str)
    log.info(f"\nExtracting frames at {fps} FPS ...")
    for cls, videos in class_videos.items():
        for vpath in videos:
            out = frames_dir / cls / vpath.stem
            if out.exists() and any(out.iterdir()):
                n = len(list(out.glob("*.jpg")))
                log.info(f"  [SKIP] {cls}/{vpath.stem} ({n} frames already extracted)")
            else:
                n = extract_frames(vpath, out, fps, img_size)
                log.info(f"  [OK]   {cls}/{vpath.stem} → {n} frames")
            frame_index[f"{cls}/{vpath.stem}"] = str(out)

    # ── Optical flow ─────────────────────────────────────────
    if not args.skip_flow and cfg["optical_flow"]["enabled"]:
        log.info("\nComputing optical flow ...")
        for cls, videos in class_videos.items():
            for vpath in videos:
                src = frames_dir / cls / vpath.stem
                dst = flow_dir / cls / vpath.stem
                if dst.exists() and any(dst.iterdir()):
                    log.info(f"  [SKIP] {cls}/{vpath.stem} flow exists")
                else:
                    n = compute_optical_flow(src, dst)
                    log.info(f"  [OK]   {cls}/{vpath.stem} → {n} flow frames")
    else:
        log.info("\nOptical flow: SKIPPED")

    # ── Splits ───────────────────────────────────────────────
    log.info("\nBuilding train/val/test splits ...")
    splits = build_splits(
        class_videos,
        cfg["dataset"]["train_split"],
        cfg["dataset"]["val_split"],
        cfg["dataset"]["test_split"],
        cfg["project"]["seed"]
    )
    for sp, items in splits.items():
        log.info(f"  {sp}: {len(items)} videos")

    # Save splits as JSON (video_id, class, video_path, frame_dir)
    splits_data = {}
    for sp, items in splits.items():
        splits_data[sp] = []
        for vid_stem, cls, vpath in items:
            fd = str(frames_dir / cls / vid_stem)
            fod = str(flow_dir / cls / vid_stem)
            splits_data[sp].append({
                "video_id": vid_stem,
                "class": cls,
                "class_idx": classes.index(cls),
                "video_path": vpath,
                "frame_dir": fd,
                "flow_dir": fod
            })

    splits_path = splits_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_data, f, indent=2)
    log.info(f"Splits saved → {splits_path}")

    # ── Class weights ────────────────────────────────────────
    weights = compute_class_weights(class_videos, classes)
    weights_path = splits_dir / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    log.info(f"Class weights saved → {weights_path}")
    for cls, w in weights.items():
        log.info(f"  {cls}: {w:.4f}")

    # ── Summary ──────────────────────────────────────────────
    summary = {
        "total_videos": total,
        "classes": classes,
        "splits": {k: len(v) for k, v in splits_data.items()},
        "fps_used": fps,
        "img_size": img_size,
        "optical_flow_computed": not args.skip_flow
    }
    with open(splits_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("STEP 1 COMPLETE. Run next: python src/data/dataset.py (verify)")
    log.info("Then: python src/training/train.py --model ResNet50")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Timeline - Data Preprocessing")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-flow", action="store_true", help="Skip optical flow computation")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS from config")
    args = parser.parse_args()
    main(args)
