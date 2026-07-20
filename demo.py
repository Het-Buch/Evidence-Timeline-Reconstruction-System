"""
DEMO — Evidence Timeline Reconstruction from Raw Video
=======================================================
Give it any video file → anomaly detection timeline with:
  - Frame-by-frame crime class confidence
  - Predicted anomaly segments with timestamps
  - Color-coded timeline bar (TP/FP/FN/TN when GT available)
  - HTML report + PNG saved to results/demo/

Usage:
  python demo.py --video path/to/video.mp4
  python demo.py --video path/to/video.mp4 --model ConvNeXtTiny
  python demo.py --video path/to/video.mp4 --fps 5 --threshold 0.4
  python demo.py --video path/to/video.mp4 --slowmo

Requirements:
  opencv-python (pip install opencv-python)
"""

import sys
import json
import yaml
import argparse
import logging
import shutil
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models.model_builder import load_model, is_temporal_model
from src.data.dataset import get_val_transform

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Class palette (14 classes) ───────────────────────────────
CLASS_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c",
    "#3498db", "#9b59b6", "#e91e63", "#ff5722", "#795548",
    "#607d8b", "#00bcd4", "#8bc34a", "#9e9e9e",
]


def extract_frames_opencv(video_path: str, fps: float, out_dir: Path) -> int:
    """Extract frames from video using OpenCV. Returns frame count."""
    try:
        import cv2
    except ImportError:
        log.error("opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    log.info(f"  Video: {Path(video_path).name}")
    log.info(f"  Original FPS: {video_fps:.1f} | Duration: {duration_sec:.1f}s | Frames: {total_frames}")
    log.info(f"  Extracting at {fps} fps...")

    step = max(1, int(video_fps / fps))
    count = 0
    frame_idx = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            path = out_dir / f"frame_{count:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            count += 1
        frame_idx += 1

    cap.release()
    log.info(f"  Extracted {count} frames")
    return count


def write_annotated_video(video_path: str, frame_probs: np.ndarray,
                          classes: list, normal_idx: int,
                          extract_fps: float, threshold: float, smooth_window: int,
                          out_path: Path):
    """
    Write an annotated MP4 at the ORIGINAL video fps (smooth, not choppy).

    Reads every frame from the source video, maps it to the closest inference
    result (obtained at extract_fps), and burns in:
      - ANOMALY / NORMAL banner (red / green)
      - Predicted class + confidence
      - Anomaly score
      - Timestamp (MM:SS)
    """
    try:
        import cv2
    except ImportError:
        log.warning("cv2 not available — skipping annotated video")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Cannot open video for annotation: {video_path}")
        return

    orig_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_inferred  = len(frame_probs)

    smoothed  = smooth_probs(frame_probs, smooth_window)
    anomaly_p = 1.0 - smoothed[:, normal_idx]

    # Hex colors → BGR for OpenCV
    HEX = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
           "#3498db","#9b59b6","#e91e63","#ff5722","#795548"]
    def hex2bgr(hx):
        hx = hx.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return (b, g, r)
    cls_bgr = [hex2bgr(HEX[i % len(HEX)]) for i in range(len(classes))]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), fourcc, orig_fps, (width, height))

    bar_h    = max(44, height // 14)
    panel_w  = 210
    font_big = bar_h / 36
    font_sm  = 0.45

    # ── Hysteresis: stabilise class label across frames ──────
    # Prevents rapid flickering between similar classes (Hazard/Violence).
    # A class must hold dominance for HYST consecutive inference steps
    # before the displayed label switches.
    HYST     = 10
    pred_raw = smoothed.argmax(axis=1)
    stable   = np.empty(n_inferred, dtype=int)
    cur_cls, dwell = int(pred_raw[0]), HYST
    for fi in range(n_inferred):
        if pred_raw[fi] == cur_cls:
            dwell = min(dwell + 1, HYST)
        else:
            dwell -= 1
            if dwell <= 0:
                cur_cls, dwell = int(pred_raw[fi]), HYST
        stable[fi] = cur_cls

    orig_frame_idx = 0
    log.info(f"Writing annotated video at {orig_fps:.1f} fps …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Map original frame → nearest inference bucket
        inf_idx   = min(int(orig_frame_idx / orig_fps * extract_fps), n_inferred - 1)
        probs_i   = smoothed[inf_idx]
        pred_cls  = stable[inf_idx]
        pred_conf = float(probs_i[pred_cls])
        anom_sc   = float(anomaly_p[inf_idx])
        is_anom   = anom_sc > threshold

        # ── Top banner ────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_h),
                      (20, 20, 200) if is_anom else (20, 160, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Left: ANOMALY / NORMAL
        cv2.putText(frame, "ANOMALY" if is_anom else "NORMAL",
                    (12, bar_h - 10), cv2.FONT_HERSHEY_DUPLEX,
                    font_big, (255, 255, 255), 2, cv2.LINE_AA)

        # Centre: stable class + confidence
        cls_label = f"{classes[pred_cls]}  {pred_conf:.0%}"
        (tw, _), _ = cv2.getTextSize(cls_label, cv2.FONT_HERSHEY_DUPLEX,
                                     font_big * 0.85, 2)
        cv2.putText(frame, cls_label,
                    (width // 2 - tw // 2, bar_h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, font_big * 0.85,
                    cls_bgr[pred_cls], 2, cv2.LINE_AA)

        # Right: anomaly score
        sc_txt = f"{anom_sc:.2f}"
        (sw, _), _ = cv2.getTextSize(sc_txt, cv2.FONT_HERSHEY_DUPLEX,
                                     font_big, 2)
        cv2.putText(frame, sc_txt,
                    (width - sw - 14, bar_h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, font_big,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Timestamp bottom-left
        cv2.putText(frame, frames_to_timestamp(orig_frame_idx, orig_fps),
                    (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (210, 210, 210), 1, cv2.LINE_AA)

        # Progress bar at very bottom
        cv2.rectangle(frame, (0, height - 5),
                      (int(width * anom_sc), height),
                      (20, 20, 220) if is_anom else (20, 160, 20), -1)

        # ── Right panel: per-class confidence bars ────────────
        panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
        panel[:] = (28, 28, 28)
        cv2.putText(panel, "Confidence", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        n_cls = len(classes)
        row_h = max(22, (height - 36) // n_cls)
        for ci, cname in enumerate(classes):
            y0   = 32 + ci * row_h
            conf = float(probs_i[ci])
            active = (ci == pred_cls)

            if active:
                cv2.rectangle(panel, (0, y0 - 2),
                              (panel_w, y0 + row_h - 2), (45, 45, 75), -1)

            # Bar
            blen = int((panel_w - 14) * conf)
            bcol = cls_bgr[ci] if ci != normal_idx else (50, 160, 50)
            cv2.rectangle(panel, (7, y0 + 5),
                          (7 + blen, y0 + row_h - 6), bcol, -1)

            # Label
            lbl = f"{cname[:9]}  {conf:.0%}"
            cv2.putText(panel, lbl, (9, y0 + row_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sm,
                        (255, 255, 255) if active else (140, 140, 140),
                        1, cv2.LINE_AA)

        # Combine for display only (save original-width frame)
        display = np.hstack([frame, panel])
        vw.write(frame)

        cv2.imshow("Evidence Timeline Reconstruction", display)
        if cv2.waitKey(max(1, int(1000 / orig_fps))) & 0xFF == ord("q"):
            break

        orig_frame_idx += 1

    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    log.info(f"Annotated video saved: {out_path}")


@torch.no_grad()
def infer_frames(model, frame_dir: Path, transform, device: str,
                 num_classes: int, batch_size: int = 32) -> np.ndarray:
    """
    Run model on every extracted frame.
    Each frame is treated as a 1-frame video: (1,1,3,H,W).
    Returns (N, num_classes) softmax probabilities.
    """
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        log.error(f"No frames found in {frame_dir}")
        sys.exit(1)

    all_probs = []
    for i in range(0, len(frames), batch_size):
        batch_paths = frames[i:i + batch_size]
        imgs = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.fromarray(
                    np.zeros((transform.transforms[0].size[0]
                              if hasattr(transform.transforms[0], 'size')
                              else 224,) * 2 + (3,), dtype=np.uint8))
            imgs.append(transform(img))

        batch = torch.stack(imgs).unsqueeze(1).to(device)  # (B,1,3,H,W)
        out = model(batch)
        if isinstance(out, tuple):
            out = out[1]
        probs = F.softmax(out.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)   # (N, C)


def smooth_probs(probs: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average along the time axis."""
    if window <= 1:
        return probs
    kernel = np.ones(window) / window
    return np.stack([
        np.convolve(probs[:, c], kernel, mode='same')
        for c in range(probs.shape[1])
    ], axis=1)


def extract_segments(binary: np.ndarray, min_len: int = 3):
    """Extract (start, end) pairs from binary anomaly mask."""
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            in_seg, start = True, i
        elif not v and in_seg:
            if i - start >= min_len:
                segs.append((start, i))
            in_seg = False
    if in_seg and len(binary) - start >= min_len:
        segs.append((start, len(binary)))
    return segs


def frames_to_timestamp(frame_idx: int, fps: float) -> str:
    total_sec = frame_idx / fps
    m = int(total_sec // 60)
    s = int(total_sec % 60)
    ms = int((total_sec - int(total_sec)) * 10)
    return f"{m:02d}:{s:02d}.{ms}"


def plot_timeline(frame_probs, classes, normal_idx, fps, out_path,
                  video_name, model_name, threshold, smooth_window):
    """Generate a 4-panel timeline figure."""
    N = len(frame_probs)
    time_axis = np.arange(N) / fps
    num_classes = len(classes)

    smoothed    = smooth_probs(frame_probs, smooth_window)
    anomaly_p   = 1.0 - smoothed[:, normal_idx]
    pred_cls    = smoothed.argmax(axis=1)
    pred_binary = (anomaly_p > threshold).astype(int)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Evidence Timeline — {video_name}\n"
        f"Model: {model_name}  |  Threshold: {threshold}  |  "
        f"Smooth window: {smooth_window}",
        fontsize=13, fontweight="bold"
    )

    gs = fig.add_gridspec(4, 1, hspace=0.45,
                          height_ratios=[2, 1.2, 1.2, 0.8])

    # ── Panel 1: per-class confidence over time ───────────────
    ax1 = fig.add_subplot(gs[0])
    for c in range(num_classes):
        if c == normal_idx:
            continue
        ax1.fill_between(time_axis, smoothed[:, c], alpha=0.35,
                         color=CLASS_COLORS[c % len(CLASS_COLORS)],
                         label=classes[c])
    ax1.set_ylabel("Class Confidence")
    ax1.set_ylim(0, 1)
    ax1.set_title("Per-Class Confidence Over Time", fontsize=10)
    ax1.legend(ncol=4, fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: anomaly probability ─────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(time_axis, anomaly_p, alpha=0.7,
                     color="#e74c3c", label="P(Anomaly)")
    ax2.axhline(threshold, color="black", linestyle="--",
                linewidth=1, label=f"Threshold {threshold}")
    ax2.set_ylabel("P(Anomaly)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Anomaly Confidence", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: predicted class index ───────────────────────
    ax3 = fig.add_subplot(gs[2])
    sc = ax3.scatter(time_axis, pred_cls,
                     c=[CLASS_COLORS[c % len(CLASS_COLORS)] for c in pred_cls],
                     s=4, alpha=0.7)
    ax3.set_yticks(range(num_classes))
    ax3.set_yticklabels(classes, fontsize=7)
    ax3.set_title("Predicted Class Per Frame", fontsize=10)
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: anomaly segments bar ────────────────────────
    ax4 = fig.add_subplot(gs[3])
    for i in range(N - 1):
        color = "#e74c3c" if pred_binary[i] else "#3498db"
        ax4.axvspan(time_axis[i], time_axis[i + 1],
                    color=color, alpha=0.8, linewidth=0)
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Anomaly detected"),
        mpatches.Patch(color="#3498db", label="Normal"),
    ]
    ax4.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax4.set_yticks([])
    ax4.set_xlabel("Time (seconds)")
    ax4.set_title("Anomaly Detection Bar", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()
    log.info(f"Timeline plot saved: {out_path}")


def generate_html(video_name, model_name, segments, fps,
                  classes, normal_idx, frame_probs, threshold,
                  plot_filename, out_path, annotated_video_filename=None):
    """Generate self-contained HTML demo report."""
    N = len(frame_probs)
    smoothed = smooth_probs(frame_probs, 5)
    anomaly_p = 1.0 - smoothed[:, normal_idx]
    pred_cls = smoothed.argmax(axis=1)

    # Dominant anomaly class (ignoring Normal)
    anom_votes = pred_cls[anomaly_p > threshold]
    if len(anom_votes):
        votes = np.bincount(anom_votes, minlength=len(classes))
        votes[normal_idx] = 0
        dominant_cls = classes[votes.argmax()]
        dominant_conf = float(smoothed[anomaly_p > threshold, votes.argmax()].mean())
    else:
        dominant_cls = "Normal"
        dominant_conf = float(smoothed[:, normal_idx].mean())

    total_duration = N / fps
    anomaly_frames = int((anomaly_p > threshold).sum())
    anomaly_pct    = 100 * anomaly_frames / max(1, N)

    seg_rows = ""
    for s, e in segments:
        cls_name = classes[int(pred_cls[s:e].mean().round())]
        conf     = float(anomaly_p[s:e].mean())
        seg_rows += f"""
        <tr>
          <td>{frames_to_timestamp(s, fps)}</td>
          <td>{frames_to_timestamp(e, fps)}</td>
          <td>{(e-s)/fps:.1f}s</td>
          <td><b style="color:#e74c3c">{cls_name}</b></td>
          <td>{conf:.3f}</td>
        </tr>"""
    if not seg_rows:
        seg_rows = "<tr><td colspan='5' style='text-align:center'>No anomaly segments detected</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Demo — {video_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 40px;
         background: #f0f2f5; color: #2c3e50; }}
  h1 {{ color: #1a237e; font-size: 1.8em; }}
  h2 {{ color: #283593; border-left: 5px solid #3f51b5; padding-left: 12px; margin-top: 32px; }}
  .meta {{ background: white; border-radius: 10px; padding: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 20px 0;
           display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }}
  .stat {{ text-align: center; }}
  .stat .val {{ font-size: 2em; font-weight: bold; color: #3f51b5; }}
  .stat .lbl {{ font-size: 0.85em; color: #607d8b; }}
  .anomaly .val {{ color: #e74c3c; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  th {{ background: #3f51b5; color: white; padding: 10px 14px; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eceff1; text-align: center; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #e8eaf6; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 16px 0;
         box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
  .card {{ background: white; border-radius: 10px; padding: 24px;
           box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 20px 0; }}
  .badge {{ display: inline-block; padding: 5px 14px; border-radius: 20px;
            font-size: 0.85em; font-weight: bold; }}
  .badge-red {{ background: #ffebee; color: #c62828; }}
  .badge-blue {{ background: #e3f2fd; color: #1565c0; }}
</style>
</head><body>

<h1>Evidence Timeline Reconstruction Demo</h1>
<p>
  <span class="badge badge-blue">Video: {video_name}</span>&nbsp;
  <span class="badge badge-blue">Model: {model_name}</span>&nbsp;
  <span class="badge {'badge-red' if dominant_cls != 'Normal' else 'badge-blue'}">
    Dominant: {dominant_cls} ({dominant_conf:.2%} confidence)
  </span>
</p>

<div class="meta">
  <div class="stat"><div class="val">{total_duration:.1f}s</div><div class="lbl">Duration</div></div>
  <div class="stat"><div class="val">{N}</div><div class="lbl">Frames analyzed</div></div>
  <div class="stat anomaly"><div class="val">{len(segments)}</div><div class="lbl">Anomaly segments</div></div>
  <div class="stat anomaly"><div class="val">{anomaly_pct:.1f}%</div><div class="lbl">Frames anomalous</div></div>
</div>

{f'''<h2>Annotated Video</h2>
<div class="card" style="text-align:center">
  <video controls width="100%" style="border-radius:8px;max-height:480px">
    <source src="{annotated_video_filename}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>''' if annotated_video_filename else ''}

<div class="card">
  <img src="{plot_filename}" alt="Timeline" />
</div>

<h2>Detected Anomaly Segments</h2>
<div class="card">
<table>
  <tr>
    <th>Start</th><th>End</th><th>Duration</th><th>Predicted Class</th><th>Avg Confidence</th>
  </tr>
  {seg_rows}
</table>
</div>

</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"HTML report saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evidence Timeline Reconstruction — Single Video Demo")
    parser.add_argument("--video",        default=None,
                        help="Path to input video file (.mp4, .avi, .mov, ...)")
    parser.add_argument("--model",        default=None,
                        help="Model name (default: best available)")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--fps",          type=float, default=3.0,
                        help="Frames to extract per second (default: 3)")
    parser.add_argument("--threshold",    type=float, default=0.45,
                        help="Anomaly detection threshold (default: 0.45)")
    parser.add_argument("--smooth",       type=int,   default=7,
                        help="Temporal smoothing window in frames (default: 7)")
    parser.add_argument("--slowmo",       action="store_true",
                        help="Video is slow-motion: use fps=1 automatically")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--keep-frames",  action="store_true",
                        help="Keep extracted frames (default: delete after inference)")
    parser.add_argument("--random",       action="store_true",
                        help="Pick a random video from the raw dataset")
    args = parser.parse_args()

    # If --random or no --video, pick one randomly from raw dataset
    if args.random or not args.video:
        with open(args.config) as _cfg_f:
            _cfg = yaml.safe_load(_cfg_f)
        raw_dir = Path(_cfg["dataset"]["raw_dir"])
        import random as _rnd
        candidates = []
        for cls_dir in sorted(raw_dir.iterdir()):
            if cls_dir.is_dir():
                candidates.extend(sorted(cls_dir.glob("*.mp4")) +
                                  sorted(cls_dir.glob("*.avi")))
        if not candidates:
            log.error(f"No videos found in {raw_dir}")
            sys.exit(1)
        chosen = _rnd.choice(candidates)
        args.video = str(chosen)
        log.info(f"Randomly selected video: {args.video}")

    if not Path(args.video).exists():
        log.error(f"Video not found: {args.video}")
        sys.exit(1)

    if args.slowmo:
        args.fps = 1.0
        log.info("Slow-motion mode: extracting at 1 fps")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device      = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    weights_dir = Path(cfg["paths"]["weights_dir"])

    # ── Pick model ────────────────────────────────────────────
    model_name = args.model
    if not model_name:
        # Use best trained model by F1 score
        metrics_path = Path("results/metrics/all_models_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                mm = json.load(f)
            available = {k: v for k, v in mm.items()
                         if (weights_dir / f"{k}_best.pt").exists()}
            if available:
                model_name = max(available, key=lambda k: available[k].get("f1_macro", 0))
        if not model_name:
            # Fallback: first available checkpoint
            for mc in cfg["models"]:
                if (weights_dir / f"{mc['name']}_best.pt").exists():
                    model_name = mc["name"]
                    break
        if not model_name:
            log.error("No trained model found. Train a model first.")
            sys.exit(1)

    log.info(f"Using model: {model_name}")
    model_cfg = next((m for m in cfg["models"] if m["name"] == model_name), None)
    if not model_cfg:
        log.error(f"Model '{model_name}' not in config.")
        sys.exit(1)

    # Per-model class info: grouped models have 5 classes, others have 14.
    if model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        gc = cfg["dataset"]["grouped_classes"]
        num_classes = len(gc)
        classes = [g["name"] for g in gc]
    else:
        num_classes = cfg["dataset"]["num_classes"]
        classes = cfg["dataset"]["classes"]
    normal_idx = classes.index("Normal") if "Normal" in classes else 0
    log.info(f"Classes ({num_classes}): {classes}")

    best_path = weights_dir / f"{model_name}_best.pt"
    model = load_model(model_cfg, num_classes, str(best_path), device)

    model_img_size = model_cfg.get("img_size", cfg["frames"]["img_size"])
    model_norm     = model_cfg.get("normalization", "imagenet")
    transform      = get_val_transform(model_img_size, model_norm)

    # ── Extract frames ────────────────────────────────────────
    video_stem = Path(args.video).stem
    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"demo_{video_stem}_"))
    try:
        frame_count = extract_frames_opencv(args.video, args.fps, tmp_dir)
        if frame_count == 0:
            log.error("No frames extracted. Check video path and opencv installation.")
            sys.exit(1)

        # ── Inference ─────────────────────────────────────────
        log.info("Running inference...")
        frame_probs = infer_frames(
            model, tmp_dir, transform, device, num_classes, args.batch_size)

        # ── Post-process ──────────────────────────────────────
        smoothed    = smooth_probs(frame_probs, args.smooth)
        anomaly_p   = 1.0 - smoothed[:, normal_idx]
        pred_binary = (anomaly_p > args.threshold).astype(int)
        segments    = extract_segments(pred_binary, min_len=3)

        log.info(f"\n{'='*55}")
        log.info(f"RESULTS — {video_stem}")
        log.info(f"  Frames analyzed : {frame_count}")
        log.info(f"  Anomaly segments: {len(segments)}")
        for i, (s, e) in enumerate(segments, 1):
            cls_name = classes[int(smoothed[s:e].argmax(axis=1).mean().round())]
            log.info(f"  Segment {i}: {frames_to_timestamp(s, args.fps)} "
                     f"-> {frames_to_timestamp(e, args.fps)}  [{cls_name}]")
        log.info(f"{'='*55}")

        # ── Outputs ───────────────────────────────────────────
        out_dir = Path("results/demo") / video_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_file = f"{video_stem}_{model_name}_timeline.png"
        plot_timeline(
            frame_probs, classes, normal_idx,
            args.fps, out_dir / plot_file,
            video_stem, model_name, args.threshold, args.smooth
        )

        vid_filename = f"{video_stem}_{model_name}_annotated.mp4"
        generate_html(
            video_stem, model_name, segments, args.fps,
            classes, normal_idx, frame_probs, args.threshold,
            plot_file, out_dir / f"{video_stem}_{model_name}_report.html",
            annotated_video_filename=vid_filename
        )

        # Save raw segment JSON
        result = {
            "video":    str(args.video),
            "model":    model_name,
            "fps":      args.fps,
            "threshold": args.threshold,
            "duration_sec": frame_count / args.fps,
            "segments": [
                {
                    "start": frames_to_timestamp(s, args.fps),
                    "end":   frames_to_timestamp(e, args.fps),
                    "start_frame": s,
                    "end_frame":   e,
                    "predicted_class": classes[int(
                        smoothed[s:e].argmax(axis=1)
                        [np.bincount(smoothed[s:e].argmax(axis=1),
                                     minlength=num_classes).argmax()]
                    )],
                    "anomaly_confidence": float(anomaly_p[s:e].mean()),
                }
                for s, e in segments
            ],
        }
        with open(out_dir / f"{video_stem}_{model_name}_result.json", "w") as f:
            json.dump(result, f, indent=2)

        # ── Annotated video ───────────────────────────────────
        log.info("Writing annotated video...")
        vid_out = out_dir / f"{video_stem}_{model_name}_annotated.mp4"
        write_annotated_video(
            args.video, frame_probs, classes, normal_idx,
            args.fps, args.threshold, args.smooth, vid_out
        )

        log.info(f"\nAll outputs in: results/demo/{video_stem}/")
        log.info(f"  Timeline PNG : {out_dir / plot_file}")
        log.info(f"  HTML report  : {out_dir / f'{video_stem}_{model_name}_report.html'}")
        log.info(f"  Annotated MP4: {vid_out}")

    finally:
        if not args.keep_frames:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
