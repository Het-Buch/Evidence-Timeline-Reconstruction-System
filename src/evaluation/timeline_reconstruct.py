"""
STEP 3b — Evidence Timeline Reconstruction
==========================================
This is the CORE of the project. It:

1. Loads trained temporal model (CNN-LSTM or CNN-Transformer)
2. Runs inference on every test video frame-by-frame
3. Produces per-frame anomaly predictions with confidence scores
4. Reconstructs temporal event boundaries (start/end timestamps)
5. Compares predicted timeline vs ground truth from annotation file
6. Computes temporal IoU (tIoU) — standard metric for temporal detection
7. Generates a detailed evidence timeline report per video (JSON + HTML)

Output per video:
  Predicted:   [Normal 0-45s] [Robbery 45-72s] [Normal 72-120s]
  Ground Truth:[Normal 0-43s] [Robbery 43-75s] [Normal 75-120s]
  tIoU:        0.82

Usage:
  python src/evaluation/timeline_reconstruct.py
  python src/evaluation/timeline_reconstruct.py --model CNNLSTM
  python src/evaluation/timeline_reconstruct.py --model CNNTransformer
  python src/evaluation/timeline_reconstruct.py --smooth-window 5
"""

import sys
import json
import yaml
import argparse
import logging
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import (parse_temporal_annotations,
                               get_frame_labels, get_transforms,
                               get_val_transform)
from src.models.model_builder import load_model, build_model, is_temporal_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Temporal IoU ─────────────────────────────────────────────
def temporal_iou(pred_seg, gt_seg):
    """
    Compute IoU between two temporal segments.
    pred_seg, gt_seg: (start_frame, end_frame)
    """
    ps, pe = pred_seg
    gs, ge = gt_seg
    inter  = max(0, min(pe, ge) - max(ps, gs))
    union  = max(pe, ge) - min(ps, gs)
    return inter / (union + 1e-8)


def compute_tiou_for_video(pred_segments, gt_segments):
    """
    Match predicted segments to ground truth segments.
    Uses best-match greedy assignment.

    Returns:
        mean_tiou: float
        matched:   list of (pred_seg, gt_seg, iou)
    """
    if not gt_segments:
        # No ground truth anomaly → penalize any prediction
        return (0.0 if pred_segments else 1.0), []

    if not pred_segments:
        return 0.0, []

    matched = []
    used_gt = set()

    for ps in pred_segments:
        best_iou = 0.0
        best_gt  = None
        for i, gs in enumerate(gt_segments):
            if i in used_gt:
                continue
            iou = temporal_iou(ps, gs)
            if iou > best_iou:
                best_iou = iou
                best_gt  = i
        if best_gt is not None and best_iou > 0:
            used_gt.add(best_gt)
            matched.append((ps, gt_segments[best_gt], best_iou))

    mean_tiou = np.mean([m[2] for m in matched]) if matched else 0.0
    return float(mean_tiou), matched


# ── Smooth predictions ───────────────────────────────────────
def smooth_predictions(preds: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Temporal smoothing with a median filter.
    Removes isolated single-frame noise.
    preds: (N,) binary array
    """
    if window <= 1 or len(preds) < window:
        return preds
    from scipy.signal import medfilt
    return medfilt(preds.astype(float), kernel_size=window).astype(np.int64)


# ── Extract predicted segments from binary array ─────────────
def extract_segments(binary_labels: np.ndarray, min_len: int = 2):
    """
    Convert binary frame labels → list of (start, end) segments.
    Only keeps segments longer than min_len frames.

    binary_labels: (N,) array of 0/1
    Returns: [(start_frame, end_frame), ...]
    """
    segments = []
    n = len(binary_labels)
    i = 0
    while i < n:
        if binary_labels[i] == 1:
            start = i
            while i < n and binary_labels[i] == 1:
                i += 1
            if (i - start) >= min_len:
                segments.append((start, i))
        else:
            i += 1
    return segments


# ── Convert frames → timestamps ──────────────────────────────
def frames_to_time(frame_idx: int, extracted_fps: int) -> str:
    """Convert extracted frame index → mm:ss string."""
    total_sec = frame_idx / extracted_fps
    m = int(total_sec // 60)
    s = int(total_sec % 60)
    return f"{m:02d}:{s:02d}"


# ── Inference on one video ───────────────────────────────────
@torch.no_grad()
def infer_video_temporal(model, frame_dir: Path, transform,
                          seq_len: int, stride: int,
                          device: str, num_classes: int,
                          img_size: int = 224):
    """
    Run CNN-LSTM / CNN-Transformer on a video.

    Sliding window over all frames → per-frame anomaly probability.
    Overlapping windows are averaged.

    Returns:
        frame_probs:  (N, 2) — softmax probabilities [P(Normal), P(Anomaly)]
        class_probs:  (C,)   — video-level class probabilities
    """
    frames = sorted(frame_dir.glob("*.jpg"))
    N = len(frames)
    if N == 0:
        return None, None

    # Accumulators for overlapping windows
    frame_prob_sum = np.zeros((N, 2), dtype=np.float64)
    frame_counts   = np.zeros(N, dtype=np.int32)
    class_prob_sum = np.zeros(num_classes, dtype=np.float64)
    window_count   = 0

    for start in range(0, max(1, N - seq_len + 1), stride):
        end = min(start + seq_len, N)
        window_frames = frames[start:end]

        # Pad short windows at end
        imgs = []
        for fp in window_frames:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            imgs.append(transform(img))

        # Pad to seq_len if needed
        while len(imgs) < seq_len:
            imgs.append(imgs[-1])

        seq = torch.stack(imgs).unsqueeze(0).to(device)  # (1, T, 3, H, W)

        frame_logits, class_logits = model(seq)           # (1,T,2), (1,C)

        fp = F.softmax(frame_logits[0], dim=-1).cpu().numpy()   # (T, 2)
        cp = F.softmax(class_logits[0], dim=-1).cpu().numpy()   # (C,)

        # Accumulate
        actual_len = end - start
        frame_prob_sum[start:end] += fp[:actual_len]
        frame_counts[start:end]   += 1
        class_prob_sum            += cp
        window_count              += 1

    # Average overlapping windows
    frame_counts = np.maximum(frame_counts, 1)
    frame_probs  = frame_prob_sum / frame_counts[:, None]
    class_probs  = class_prob_sum / max(1, window_count)

    return frame_probs, class_probs


@torch.no_grad()
def infer_video_frame(model, frame_dir: Path, transform,
                       device: str, num_classes: int, batch_size: int = 16,
                       img_size: int = 224):
    """
    Run VideoLevelModel or VideoModel3D on a video frame-by-frame.
    Returns per-frame class probabilities (N, C).

    Both model types expect (B, N, 3, H, W) — we treat each frame as a
    single-frame "video" by unsqueezing the temporal dimension:
      (B, 3, H, W) → (B, 1, 3, H, W)
    The model sees a 1-frame video and predicts a class for it.
    This gives per-frame predictions suitable for timeline reconstruction.
    """
    frames = sorted(frame_dir.glob("*.jpg"))
    N = len(frames)
    if N == 0:
        return None

    all_probs = []
    for i in range(0, N, batch_size):
        batch_paths = frames[i:i + batch_size]
        imgs = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            imgs.append(transform(img))

        # (B, 3, H, W) → (B, 1, 3, H, W): single-frame video for each sample
        batch = torch.stack(imgs).unsqueeze(1).to(device)

        out = model(batch)
        # Temporal models return (frame_logits, class_logits) — use class_logits
        if isinstance(out, tuple):
            out = out[1]
        probs = F.softmax(out, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)   # (N, C)


# ── Plot timeline for one video ──────────────────────────────
def plot_timeline(video_id, pred_binary, gt_binary, frame_probs_anomaly,
                  class_name, tiou, out_path, extracted_fps):
    N = len(pred_binary)
    time_axis = np.arange(N) / extracted_fps   # seconds

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Evidence Timeline — {video_id}\n"
                 f"Class: {class_name} | tIoU: {tiou:.3f}",
                 fontsize=13, fontweight="bold")

    # Row 1: Anomaly confidence over time
    axes[0].fill_between(time_axis, frame_probs_anomaly, alpha=0.7,
                          color="#e74c3c", label="Anomaly confidence")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Threshold 0.5")
    axes[0].set_ylabel("P(Anomaly)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Row 2: Predicted vs GT binary timeline
    axes[1].fill_between(time_axis, pred_binary, step="post",
                          alpha=0.7, color="#e74c3c", label="Predicted anomaly")
    axes[1].fill_between(time_axis, gt_binary,   step="post",
                          alpha=0.5, color="#27ae60", label="Ground truth", linestyle="--")
    axes[1].set_ylabel("Anomaly (0/1)")
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Row 3: Timeline bar (color-coded)
    colors = []
    for p, g in zip(pred_binary, gt_binary):
        if p == 1 and g == 1:
            colors.append("#27ae60")   # True positive  (green)
        elif p == 1 and g == 0:
            colors.append("#e74c3c")   # False positive (red)
        elif p == 0 and g == 1:
            colors.append("#f39c12")   # False negative (orange)
        else:
            colors.append("#3498db")   # True negative  (blue)

    for i in range(N - 1):
        axes[2].axvspan(time_axis[i], time_axis[i + 1],
                        color=colors[i], alpha=0.8, linewidth=0)

    legend_patches = [
        mpatches.Patch(color="#27ae60", label="True Positive"),
        mpatches.Patch(color="#e74c3c", label="False Positive"),
        mpatches.Patch(color="#f39c12", label="False Negative"),
        mpatches.Patch(color="#3498db", label="True Negative"),
    ]
    axes[2].legend(handles=legend_patches, fontsize=7, loc="upper right")
    axes[2].set_ylabel("Detection")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()


# ── HTML report generator ─────────────────────────────────────
def generate_html_report(all_results, out_path, model_name):
    rows = ""
    for r in sorted(all_results, key=lambda x: x["tiou"], reverse=True):
        gt_str   = " | ".join(
            [f"{frames_to_time(s,3)}-{frames_to_time(e,3)}"
             for s, e in r["gt_segments"]]) or "None (Normal)"
        pred_str = " | ".join(
            [f"{frames_to_time(s,3)}-{frames_to_time(e,3)}"
             for s, e in r["pred_segments"]]) or "None detected"
        color = ("#27ae60" if r["tiou"] > 0.5 else
                 "#f39c12" if r["tiou"] > 0.2 else "#e74c3c")
        rows += f"""
        <tr>
          <td><b>{r['video_id']}</b></td>
          <td>{r['class_name']}</td>
          <td>{gt_str}</td>
          <td>{pred_str}</td>
          <td style="color:{color};font-weight:bold">{r['tiou']:.3f}</td>
          <td>{r['num_frames']}</td>
        </tr>"""

    mean_tiou = np.mean([r["tiou"] for r in all_results]) if all_results else 0.0
    html = f"""<!DOCTYPE html>
<html><head>
<title>Evidence Timeline Reconstruction Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 12px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #ecf0f1; }}
  .stat {{ display:inline-block; background:white; border:1px solid #ddd;
           border-radius:8px; padding:16px 24px; margin:10px; text-align:center; }}
  .stat-val {{ font-size:2rem; font-weight:700; color:#3498db; }}
  .stat-label {{ color:#7f8c8d; font-size:0.85rem; }}
</style>
</head><body>
<h1>Evidence Timeline Reconstruction Report</h1>
<p>Model: <b>{model_name}</b> | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div>
  <div class="stat"><div class="stat-val">{mean_tiou:.3f}</div>
    <div class="stat-label">Mean tIoU</div></div>
  <div class="stat"><div class="stat-val">{len(all_results)}</div>
    <div class="stat-label">Videos Evaluated</div></div>
  <div class="stat"><div class="stat-val">{sum(1 for r in all_results if r['tiou']>0.5)}</div>
    <div class="stat-label">tIoU &gt; 0.5</div></div>
</div>

<h2>Per-Video Timeline Results</h2>
<table>
  <tr>
    <th>Video</th><th>Class</th><th>Ground Truth Segments</th>
    <th>Predicted Segments</th><th>tIoU</th><th>Frames</th>
  </tr>
  {rows}
</table>

<h2>Timeline Plots</h2>
<p>Individual timeline plots are saved in <code>results/timeline/plots/</code></p>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)
    log.info(f"HTML report → {out_path}")


# ── Main ─────────────────────────────────────────────────────
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device      = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    ext_fps     = cfg["frames"]["fps"]
    seq_len     = cfg["training"].get("seq_len",     16)
    stride      = cfg["training"].get("stride_val",  8)
    img_size    = cfg["frames"]["img_size"]

    ann_path    = str(Path(cfg["dataset"]["splits_dir"]) /
                      "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    splits_path = str(Path(cfg["dataset"]["splits_dir"]) / "splits.json")
    weights_dir = Path(cfg["paths"]["weights_dir"])

    # ── Pick model ──────────────────────────────────────────
    model_name = args.model
    if not model_name:
        # Auto-pick best temporal model, fallback to best frame model
        for preferred in ["CNNTransformer", "CNNLSTM"]:
            if (weights_dir / f"{preferred}_best.pt").exists():
                model_name = preferred
                break
        if not model_name:
            metrics_path = Path("results/metrics/all_models_metrics.json")
            if metrics_path.exists():
                with open(metrics_path) as f:
                    mm = json.load(f)
                model_name = max(mm, key=lambda k: mm[k].get("f1_macro", 0))
            else:
                log.error("No trained model found. Run train.py first.")
                return

    model_cfg = next((m for m in cfg["models"] if m["name"] == model_name), None)
    if not model_cfg:
        log.error(f"Model '{model_name}' not in config.")
        return

    best_path = weights_dir / f"{model_name}_best.pt"
    if not best_path.exists():
        log.error(f"No checkpoint: {best_path}")
        return

    out_dir   = Path("results/timeline") / model_name
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-model class info: grouped models have 5 classes, others have 14.
    if model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        gc = cfg["dataset"]["grouped_classes"]
        num_classes = len(gc)
        classes = [g["name"] for g in gc]
    else:
        num_classes = cfg["dataset"]["num_classes"]
        classes = cfg["dataset"]["classes"]
    log.info(f"Classes ({num_classes}): {classes}")

    log.info(f"Loading model: {model_name}")
    model    = load_model(model_cfg, num_classes, str(best_path), device)
    temporal = is_temporal_model(model_cfg)
    # Use per-model img_size and normalization (critical for R2Plus1D: 112px + kinetics norm)
    model_img_size = model_cfg.get("img_size", img_size)
    model_norm     = model_cfg.get("normalization", "imagenet")
    transform = get_val_transform(model_img_size, model_norm)

    # ── Load annotations ────────────────────────────────────
    annotations = parse_temporal_annotations(ann_path)
    log.info(f"Loaded {len(annotations)} temporal annotations")

    # ── Load test split ─────────────────────────────────────
    with open(splits_path) as f:
        test_entries = json.load(f)["test"]

    log.info(f"\n{'='*60}")
    log.info(f"TIMELINE RECONSTRUCTION — {model_name}")
    log.info(f"  Videos to process: {len(test_entries)}")
    log.info(f"  FPS extracted:     {ext_fps}")
    log.info(f"{'='*60}\n")

    all_results = []
    all_tious   = []

    for entry in test_entries:
        vid_stem  = entry["video_id"]
        cls_name  = entry["class"]
        cls_idx   = entry["class_idx"]
        frame_dir = Path(entry["frame_dir"])

        if not frame_dir.exists() or not any(frame_dir.glob("*.jpg")):
            log.warning(f"  [SKIP] {vid_stem} — no frames")
            continue

        frames = sorted(frame_dir.glob("*.jpg"))
        N = len(frames)

        # ── Ground truth labels ──────────────────────────────
        if vid_stem in annotations:
            gt_segs   = annotations[vid_stem]["segments"]
            gt_binary = get_frame_labels(N, gt_segs, 30.0, ext_fps)
        else:
            gt_segs   = []
            gt_binary = np.zeros(N, dtype=np.int64)  # Normal

        # ── Model inference ──────────────────────────────────
        if temporal:
            frame_probs, class_probs = infer_video_temporal(
                model, frame_dir, transform, seq_len, stride, device, num_classes,
                img_size=model_img_size)
            if frame_probs is None:
                continue
            # Binary prediction: anomaly if P(anomaly) > threshold
            anomaly_probs = frame_probs[:, 1]
            pred_binary   = (anomaly_probs > args.threshold).astype(np.int64)
            pred_class    = int(np.argmax(class_probs))
        else:
            frame_probs = infer_video_frame(
                model, frame_dir, transform, device, num_classes,
                img_size=model_img_size)
            if frame_probs is None:
                continue
            # For frame-level / video-level model: anomaly if predicted class != Normal
            # Use classes list (works for both grouped-5 and original-14)
            normal_idx  = classes.index("Normal") if "Normal" in classes else 0
            pred_class  = int(np.argmax(frame_probs, axis=1)[N // 2])  # mid-frame
            # Anomaly confidence = 1 - P(Normal)
            normal_col = min(normal_idx, frame_probs.shape[1] - 1)
            anomaly_probs = 1.0 - frame_probs[:, normal_col]
            pred_binary   = (anomaly_probs > args.threshold).astype(np.int64)
            class_probs   = frame_probs.mean(axis=0)

        # ── Temporal smoothing ───────────────────────────────
        pred_binary = smooth_predictions(pred_binary, args.smooth_window)

        # ── Extract predicted segments ───────────────────────
        pred_segs = extract_segments(pred_binary, min_len=2)

        # ── Compute tIoU ────────────────────────────────────
        tiou, matched = compute_tiou_for_video(pred_segs, gt_segs)
        all_tious.append(tiou)

        log.info(
            f"  {vid_stem:35s} | GT={len(gt_segs)} segs | "
            f"Pred={len(pred_segs)} segs | tIoU={tiou:.3f}"
        )

        # ── Plot timeline ────────────────────────────────────
        plot_path = plots_dir / f"{vid_stem}_timeline.png"
        try:
            plot_timeline(
                vid_stem, pred_binary, gt_binary,
                anomaly_probs, cls_name, tiou, plot_path, ext_fps
            )
        except Exception as e:
            log.debug(f"Plot failed for {vid_stem}: {e}")

        # ── Store result ─────────────────────────────────────
        result = {
            "video_id":      vid_stem,
            "class_name":    cls_name,
            "class_idx":     cls_idx,
            "num_frames":    N,
            "gt_segments":   gt_segs,
            "pred_segments": pred_segs,
            "tiou":          tiou,
            "matched":       [(list(p), list(g), round(iou, 4))
                              for p, g, iou in matched],
            "pred_class":    classes[pred_class],
            "duration_sec":  round(N / ext_fps, 1),
        }
        all_results.append(result)

    # ── Aggregate metrics ────────────────────────────────────
    mean_tiou = float(np.mean(all_tious)) if all_tious else 0.0
    tiou_at_50 = float(np.mean([t > 0.5 for t in all_tious])) if all_tious else 0.0
    tiou_at_25 = float(np.mean([t > 0.25 for t in all_tious])) if all_tious else 0.0

    summary = {
        "model":        model_name,
        "num_videos":   len(all_results),
        "mean_tiou":    round(mean_tiou, 4),
        "tiou@0.5":     round(tiou_at_50, 4),
        "tiou@0.25":    round(tiou_at_25, 4),
        "threshold":    args.threshold,
        "smooth_window": args.smooth_window,
    }

    log.info(f"\n{'='*60}")
    log.info("TIMELINE RECONSTRUCTION RESULTS")
    log.info(f"  Mean tIoU:    {mean_tiou:.4f}")
    log.info(f"  tIoU @ 0.50:  {tiou_at_50:.4f}  (fraction of videos with tIoU > 0.5)")
    log.info(f"  tIoU @ 0.25:  {tiou_at_25:.4f}")
    log.info(f"{'='*60}")

    # Save results
    with open(out_dir / f"{model_name}_timeline_results.json", "w") as f:
        json.dump({"summary": summary, "videos": all_results}, f, indent=2)

    with open(out_dir / f"{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    generate_html_report(
        all_results,
        out_dir / f"{model_name}_timeline_report.html",
        model_name
    )

    log.info(f"\nOutputs saved to: results/timeline/{model_name}/")
    log.info(f"  {model_name}_timeline_report.html  <- open in browser")
    log.info(f"  {model_name}_timeline_results.json")
    log.info(f"  plots/  <- per-video timeline charts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evidence Timeline Reconstruction")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--model",         default=None,
                        help="Model to use (CNNLSTM, CNNTransformer, or any frame model)")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Anomaly detection threshold (default: 0.5)")
    parser.add_argument("--smooth-window", type=int,   default=5,
                        help="Temporal smoothing window size (default: 5)")
    args = parser.parse_args()
    main(args)
