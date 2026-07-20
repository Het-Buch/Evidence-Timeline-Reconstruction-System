"""
STEP 5 — Web UI for Testing & Timeline Tracking
================================================
Flask app that lets you:
  - Upload a video (or paste a path)
  - Run inference on it using your best trained model
  - View frame-by-frame anomaly predictions on a timeline
  - See confidence scores, GradCAM overlays, class probabilities
  - Track multiple video results in a session history

Usage:
  python ui/app.py
  Then open: http://localhost:5000

Requirements: Flask (pip install flask)
"""

import sys
import os
import json
import yaml
import uuid
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

import torch
import torch.nn.functional as F
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import get_transforms
from src.models.model_builder import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit

# ── Global state ─────────────────────────────────────────────
UI_DIR     = Path(__file__).parent
UPLOAD_DIR = UI_DIR / "uploads"
RESULT_DIR = UI_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

CLASSES      = CFG["dataset"]["classes"]
NUM_CLASSES  = CFG["dataset"]["num_classes"]
IMG_SIZE     = CFG["frames"]["img_size"]
DEVICE_STR   = CFG["project"]["device"]
DEVICE       = DEVICE_STR if torch.cuda.is_available() else "cpu"
TRANSFORM    = get_transforms("test", IMG_SIZE)

# Class → color mapping for timeline visualization
CLASS_COLORS = {
    "Normal":        "#27ae60",
    "Abuse":         "#e74c3c",
    "Arrest":        "#e67e22",
    "Arson":         "#c0392b",
    "Assault":       "#e74c3c",
    "Burglary":      "#8e44ad",
    "Explosion":     "#d35400",
    "Fighting":      "#e74c3c",
    "RoadAccidents": "#f39c12",
    "Robbery":       "#c0392b",
    "Shooting":      "#922b21",
    "Shoplifting":   "#7d6608",
    "Stealing":      "#784212",
    "Vandalism":     "#6c3483",
}

# ── Model loader (lazy, cached) ───────────────────────────────
_model_cache = {}


def get_best_model():
    """Load best model by F1 from metrics, or fallback to first available."""
    global _model_cache
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["model_name"]

    weights_dir = Path(__file__).parents[1] / CFG["paths"]["weights_dir"]
    metrics_path = Path(__file__).parents[1] / CFG["evaluation"]["results_dir"] / "metrics" / "all_models_metrics.json"

    model_name = None
    model_cfg  = None

    if metrics_path.exists():
        with open(metrics_path) as f:
            all_m = json.load(f)
        model_name = max(all_m, key=lambda k: all_m[k].get("f1_macro", 0))

    if not model_name:
        # Fallback: first available checkpoint
        for mc in CFG["models"]:
            bp = weights_dir / f"{mc['name']}_best.pt"
            if bp.exists():
                model_name = mc["name"]
                break

    if not model_name:
        return None, None

    model_cfg = next((m for m in CFG["models"] if m["name"] == model_name), None)
    best_path = weights_dir / f"{model_name}_best.pt"

    if not best_path.exists() or model_cfg is None:
        return None, None

    log.info(f"Loading model: {model_name} from {best_path}")
    model = load_model(model_cfg, NUM_CLASSES, str(best_path), DEVICE)
    _model_cache["model"] = model
    _model_cache["model_name"] = model_name
    return model, model_name


# ── Inference on a single frame ───────────────────────────────
@torch.no_grad()
def predict_frame(model, img_pil: Image.Image):
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "class_idx":  pred_idx,
        "class_name": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs":      {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "color":      CLASS_COLORS.get(CLASSES[pred_idx], "#95a5a6"),
    }


# ── Video analysis ────────────────────────────────────────────
def analyze_video(video_path: Path, fps_sample: int = 1) -> dict:
    """
    Extract frames and run inference on each.
    Returns timeline data (one prediction per sampled second).
    """
    model, model_name = get_best_model()
    if model is None:
        return {"error": "No trained model found. Run train.py first."}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path.name}"}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps
    interval = max(1, int(round(native_fps / fps_sample)))

    timeline = []
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / native_fps
            img_pil = Image.fromarray(
                cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
            )
            pred = predict_frame(model, img_pil)
            pred["timestamp_sec"] = round(ts, 2)
            pred["timestamp_fmt"] = f"{int(ts//60):02d}:{int(ts%60):02d}"
            pred["frame_idx"] = frame_idx
            timeline.append(pred)
            sampled += 1
        frame_idx += 1

    cap.release()

    # Aggregate stats
    class_counts = {}
    for t in timeline:
        cn = t["class_name"]
        class_counts[cn] = class_counts.get(cn, 0) + 1

    dominant = max(class_counts, key=class_counts.get) if class_counts else "Unknown"
    anomaly_frames = sum(1 for t in timeline if t["class_name"] != "Normal")
    anomaly_pct = round(anomaly_frames / max(1, len(timeline)) * 100, 1)

    # Detect anomaly segments (consecutive anomaly frames → events)
    events = []
    i = 0
    while i < len(timeline):
        t = timeline[i]
        if t["class_name"] != "Normal":
            start = t["timestamp_fmt"]
            start_sec = t["timestamp_sec"]
            j = i
            while j < len(timeline) and timeline[j]["class_name"] == t["class_name"]:
                j += 1
            end_sec = timeline[j-1]["timestamp_sec"] if j > 0 else start_sec
            events.append({
                "class": t["class_name"],
                "color": t["color"],
                "start": start,
                "end":   timeline[j-1]["timestamp_fmt"] if j <= len(timeline) else start,
                "start_sec": start_sec,
                "end_sec":   end_sec,
                "duration_sec": round(end_sec - start_sec, 1),
            })
            i = j
        else:
            i += 1

    return {
        "video_name":    video_path.name,
        "model_used":    model_name,
        "duration_sec":  round(duration_sec, 1),
        "total_frames":  total_frames,
        "sampled_frames": sampled,
        "timeline":      timeline,
        "class_counts":  class_counts,
        "dominant_class": dominant,
        "anomaly_pct":   anomaly_pct,
        "events":        events,
        "analyzed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           classes=CLASSES,
                           class_colors=CLASS_COLORS,
                           device=DEVICE)


@app.route("/api/status")
def status():
    model, model_name = get_best_model()
    return jsonify({
        "model_loaded": model is not None,
        "model_name":   model_name,
        "device":       DEVICE,
        "classes":      CLASSES,
        "num_classes":  NUM_CLASSES,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files and "video_path" not in request.form:
        return jsonify({"error": "No video provided"}), 400

    fps_sample = int(request.form.get("fps_sample", 1))
    session_id = str(uuid.uuid4())[:8]

    if "video" in request.files:
        f = request.files["video"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        suffix = Path(f.filename).suffix
        video_path = UPLOAD_DIR / f"{session_id}{suffix}"
        f.save(str(video_path))
        cleanup = True
    else:
        video_path = Path(request.form["video_path"])
        if not video_path.exists():
            return jsonify({"error": f"File not found: {video_path}"}), 400
        cleanup = False

    try:
        result = analyze_video(video_path, fps_sample=fps_sample)
        # Save result
        result_path = RESULT_DIR / f"{session_id}_result.json"
        with open(result_path, "w") as rp:
            json.dump(result, rp, indent=2)
        result["session_id"] = session_id
        return jsonify(result)
    except Exception as e:
        log.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500
    finally:
        if cleanup and video_path.exists():
            try:
                video_path.unlink()
            except Exception:
                pass


@app.route("/api/history")
def history():
    results = []
    for rp in sorted(RESULT_DIR.glob("*_result.json"), reverse=True)[:20]:
        with open(rp) as f:
            r = json.load(f)
        results.append({
            "session_id":    rp.stem.replace("_result", ""),
            "video_name":    r.get("video_name"),
            "dominant_class": r.get("dominant_class"),
            "anomaly_pct":   r.get("anomaly_pct"),
            "duration_sec":  r.get("duration_sec"),
            "analyzed_at":   r.get("analyzed_at"),
        })
    return jsonify(results)


@app.route("/api/result/<session_id>")
def get_result(session_id):
    rp = RESULT_DIR / f"{session_id}_result.json"
    if not rp.exists():
        return jsonify({"error": "Result not found"}), 404
    with open(rp) as f:
        return jsonify(json.load(f))


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    log.info("Starting Evidence Timeline UI...")
    log.info(f"Device: {DEVICE}")
    model, name = get_best_model()
    if model:
        log.info(f"Model loaded: {name}")
    else:
        log.warning("No trained model found — upload videos but train first!")
    app.run(debug=False, host="0.0.0.0", port=5000)
