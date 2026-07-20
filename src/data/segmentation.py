"""
STEP 1b — Image Segmentation (FIXED for Surveillance Footage)
==============================================================
Previous issue: Watershed was treating entire background as blobs.
Root cause: UCF-Crime is fixed-camera surveillance footage. The correct
approach is BACKGROUND SUBTRACTION (MOG2/KNN), not watershed.

Methods:
  mog2       (default/best) — Mixture of Gaussians background subtraction
  knn                       — K-Nearest Neighbours background subtractor
  frame_diff                — Consecutive frame differencing (fastest)
  saliency                  — Spectral residual saliency (motion-agnostic)

What correct output looks like:
  seg_XXXXX.png  → Original frame + GREEN overlay on moving person/object
                   + CYAN bounding box + YELLOW contour boundary
  mask_XXXXX.png → BLACK background, WHITE where person/motion is

Usage:
  # FIRST: delete old bad segments, then rerun
  python src/data/segmentation.py --clean
  python src/data/segmentation.py                      (MOG2 default)
  python src/data/segmentation.py --method frame_diff  (fastest)
  python src/data/segmentation.py --class Abuse        (one class only)
"""

import sys
import cv2
import json
import yaml
import shutil
import argparse
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Shared: build colored overlay from a binary fg mask
# ─────────────────────────────────────────────────────────────
def _build_overlay(img_bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Visualization:
      - Foreground pixels  → semi-transparent GREEN fill
      - Each detected blob → CYAN bounding box
      - Mask boundary      → YELLOW outline
    """
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    # Green tint on foreground
    fg_colored = np.zeros_like(img_bgr)
    fg_colored[fg_mask == 255] = (0, 200, 80)
    overlay = cv2.addWeighted(img_bgr, 0.60, fg_colored, 0.40, 0)

    # Bounding boxes + contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.002 * h * w  # ignore dust (<0.2% of frame)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 220, 0), 2)
        cv2.drawContours(overlay, [cnt], -1, (255, 255, 255), 1)

    # Yellow boundary
    boundary = cv2.Canny(fg_mask, 100, 200)
    overlay[boundary > 0] = (0, 220, 255)
    return overlay


# ─────────────────────────────────────────────────────────────
# METHOD 1: MOG2  (best for surveillance)
# Learns background from all frames, marks deviations as fg
# ─────────────────────────────────────────────────────────────
def mog2_segment_video(frames: list, out_dir: Path, max_frames: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build background model over all frames
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=min(len(frames), 300),
        varThreshold=40,
        detectShadows=True
    )
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is not None:
            subtractor.apply(img, learningRate=0.05)

    # Select subset for output
    selected = frames[::max(1, len(frames) // max_frames)][:max_frames]
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    processed = 0
    total_fg = 0.0

    for fp in selected:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        raw = subtractor.apply(img, learningRate=0)
        # 255 = definite fg, 127 = shadow → keep only 255
        fg = np.where(raw == 255, 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open,  iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
        fg = cv2.dilate(fg, k_close, iterations=1)

        overlay = _build_overlay(img, fg)
        idx = fp.stem.split("_")[-1]
        cv2.imwrite(str(out_dir / f"seg_{idx}.png"),  overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(out_dir / f"mask_{idx}.png"), fg,      [cv2.IMWRITE_PNG_COMPRESSION, 3])
        total_fg += fg.mean() / 255.0
        processed += 1

    avg_fg = round(total_fg / max(1, processed) * 100, 1)
    return {"processed": processed, "avg_fg_pct": avg_fg, "method": "mog2"}


# ─────────────────────────────────────────────────────────────
# METHOD 2: KNN background subtractor
# ─────────────────────────────────────────────────────────────
def knn_segment_video(frames: list, out_dir: Path, max_frames: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subtractor = cv2.createBackgroundSubtractorKNN(
        history=min(len(frames), 300),
        dist2Threshold=400.0,
        detectShadows=True
    )
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is not None:
            subtractor.apply(img, learningRate=0.05)

    selected = frames[::max(1, len(frames) // max_frames)][:max_frames]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    processed = 0

    for fp in selected:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        raw = subtractor.apply(img, learningRate=0)
        fg = np.where(raw == 255, 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.dilate(fg, k, iterations=1)
        overlay = _build_overlay(img, fg)
        idx = fp.stem.split("_")[-1]
        cv2.imwrite(str(out_dir / f"seg_{idx}.png"),  overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(out_dir / f"mask_{idx}.png"), fg,      [cv2.IMWRITE_PNG_COMPRESSION, 3])
        processed += 1

    return {"processed": processed, "method": "knn"}


# ─────────────────────────────────────────────────────────────
# METHOD 3: Frame differencing (fastest, no model building)
# ─────────────────────────────────────────────────────────────
def frame_diff_segment_video(frames: list, out_dir: Path, max_frames: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = frames[::max(1, len(frames) // max_frames)][:max_frames]
    if len(selected) < 2:
        return {"processed": 0, "method": "frame_diff"}

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    prev_gray = None
    processed = 0

    for fp in selected:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            h, w = gray.shape
            idx = fp.stem.split("_")[-1]
            cv2.imwrite(str(out_dir / f"mask_{idx}.png"), np.zeros((h, w), np.uint8))
            cv2.imwrite(str(out_dir / f"seg_{idx}.png"), img)
            processed += 1
            continue

        diff = cv2.absdiff(prev_gray, gray)
        _, fg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)
        fg = cv2.dilate(fg, k, iterations=2)

        overlay = _build_overlay(img, fg)
        idx = fp.stem.split("_")[-1]
        cv2.imwrite(str(out_dir / f"seg_{idx}.png"),  overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(out_dir / f"mask_{idx}.png"), fg,      [cv2.IMWRITE_PNG_COMPRESSION, 3])
        prev_gray = gray
        processed += 1

    return {"processed": processed, "method": "frame_diff"}


# ─────────────────────────────────────────────────────────────
# METHOD 4: Spectral residual saliency
# ─────────────────────────────────────────────────────────────
def saliency_segment_video(frames: list, out_dir: Path, max_frames: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    except AttributeError:
        log.warning("Saliency unavailable — falling back to MOG2")
        return mog2_segment_video(frames, out_dir, max_frames)

    selected = frames[::max(1, len(frames) // max_frames)][:max_frames]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    processed = 0

    for fp in selected:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        ok, sal_map = sal.computeSaliency(img)
        if not ok:
            continue
        sal_u8 = (sal_map * 255).astype(np.uint8)
        _, fg = cv2.threshold(sal_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.dilate(fg, k, iterations=1)
        overlay = _build_overlay(img, fg)
        idx = fp.stem.split("_")[-1]
        cv2.imwrite(str(out_dir / f"seg_{idx}.png"),  overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(out_dir / f"mask_{idx}.png"), fg,      [cv2.IMWRITE_PNG_COMPRESSION, 3])
        processed += 1

    return {"processed": processed, "method": "saliency"}


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────
def segment_video_frames(frames_dir: Path, out_dir: Path,
                          method: str, max_frames: int) -> dict:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return {"processed": 0, "error": "no frames"}
    dispatch = {
        "mog2":       mog2_segment_video,
        "knn":        knn_segment_video,
        "frame_diff": frame_diff_segment_video,
        "saliency":   saliency_segment_video,
    }
    return dispatch.get(method, mog2_segment_video)(frames, out_dir, max_frames)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seg_dir = Path(cfg["segmentation"]["output_dir"])
    classes = cfg["dataset"]["classes"]

    if args.cls:
        classes = [c for c in classes if c == args.cls]
        if not classes:
            log.error(f"Class '{args.cls}' not found in config.")
            return

    if args.clean:
        log.info(f"Cleaning {seg_dir} ...")
        if seg_dir.exists():
            shutil.rmtree(seg_dir)
        seg_dir.mkdir(parents=True)
        log.info("  Done.")

    splits_path = Path(cfg["dataset"]["splits_dir"]) / "splits.json"
    if not splits_path.exists():
        log.error("splits.json not found. Run preprocess.py first.")
        return

    with open(splits_path) as f:
        splits = json.load(f)

    log.info(f"\n{'='*60}")
    log.info(f"STEP 1b: SEGMENTATION  method={args.method}  max_frames={args.max_frames}")
    log.info(f"{'='*60}")

    all_entries = splits["train"] + splits["val"] + splits["test"]
    all_stats = {}
    total_done = 0

    for entry in all_entries:
        cls = entry["class"]
        if cls not in classes:
            continue
        if args.max_videos and total_done >= args.max_videos:
            break

        vid_stem = entry["video_id"]
        src_dir  = Path(entry["frame_dir"])
        dst_dir  = seg_dir / cls / vid_stem

        if not args.clean and dst_dir.exists() and any(dst_dir.glob("seg_*.png")):
            n = len(list(dst_dir.glob("seg_*.png")))
            log.info(f"  [SKIP] {cls}/{vid_stem} ({n} segs)")
            continue

        if not src_dir.exists() or not any(src_dir.glob("*.jpg")):
            log.warning(f"  [MISS] {cls}/{vid_stem} — no frames, run preprocess.py")
            continue

        stats = segment_video_frames(src_dir, dst_dir, args.method, args.max_frames)
        fg = stats.get("avg_fg_pct", "?")
        log.info(f"  [OK]   {cls}/{vid_stem} → {stats.get('processed',0)} frames, fg={fg}%")
        all_stats[f"{cls}/{vid_stem}"] = stats
        total_done += 1

    seg_dir.mkdir(parents=True, exist_ok=True)
    with open(seg_dir / "segmentation_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    log.info(f"\nProcessed {total_done} videos.")
    log.info("\nExpected output:")
    log.info("  seg_XXXXX.png  → person/object highlighted GREEN + CYAN box")
    log.info("  mask_XXXXX.png → WHITE on person, BLACK on background")
    log.info("\nIf masks still mostly black → try: --method frame_diff")
    log.info("Run next: python src/training/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--method",     default="mog2",
                        choices=["mog2", "knn", "frame_diff", "saliency"])
    parser.add_argument("--class",      dest="cls", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--clean",      action="store_true",
                        help="Delete existing segments before running")
    args = parser.parse_args()
    main(args)
