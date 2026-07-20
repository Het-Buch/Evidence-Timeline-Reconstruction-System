"""
STEP 4 — Explainable AI (XAI)
==============================
Generates GradCAM and LIME explanations for the best/specified model.
Also produces per-class pixel intensity histograms.

Supports both 2D models (VideoLevelModel) and 3D models (VideoModel3D).
For 3D models: a single representative frame is replicated N times to form
a valid video input (B, N, 3, H, W), and GradCAM hooks target Conv3d layers.

Outputs:
  results/xai/
    gradcam/    ← GradCAM overlays per class
    lime/       ← LIME superpixel explanations
    histograms/ ← pixel intensity histograms per class

Usage:
  python src/evaluation/xai.py
  python src/evaluation/xai.py --model EfficientNetB3 --num-samples 20
"""

import sys
import json
import yaml
import argparse
import logging
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import get_transforms, get_val_transform, build_group_mapping
from src.models.model_builder import load_model, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# Input preparation — handles both 2D and 3D model formats
# ════════════════════════════════════════════════════════════
def prepare_input(img_tensor: torch.Tensor, model_cfg: dict,
                  frames_per_video: int = 16) -> torch.Tensor:
    """
    Convert a single-frame tensor (1, 3, H, W) into the correct
    input format for the target model.

    VideoLevelModel / VideoModel3D both expect (B, N, 3, H, W).
    We replicate the single frame N times — the model sees a static
    "video" but GradCAM gradients still flow to the spatial features.

    CNNLSTM / CNNTransformer also expect (B, T, 3, H, W) — same fix.
    """
    if img_tensor.dim() == 4:
        # (1, 3, H, W) → (1, N, 3, H, W)
        img_tensor = img_tensor.unsqueeze(1).expand(-1, frames_per_video, -1, -1, -1)
    return img_tensor


def model_forward(model, img_tensor: torch.Tensor):
    """
    Forward pass that normalises the output to (B, num_classes) logits
    regardless of model type (temporal vs video-level).
    """
    out = model(img_tensor)
    if isinstance(out, tuple):
        return out[1]   # CNN-LSTM / CNN-Transformer: use class logits
    return out


# ════════════════════════════════════════════════════════════
# GradCAM — works with Conv2d (2D) and Conv3d (3D) backbones
# ════════════════════════════════════════════════════════════
class GradCAM:
    """
    Gradient-weighted Class Activation Map.

    For 2D models: hooks the last Conv2d, produces (H', W') CAM.
    For 3D models: hooks the last Conv3d, mean-pools the temporal
    dimension, then produces (H', W') CAM.
    """

    def __init__(self, model, target_layer_name: str = None):
        self.model = model
        self.gradients  = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks(target_layer_name)

    def _find_last_conv(self, model):
        """Find last Conv2d or Conv3d in the model."""
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                last_conv = (name, module)
        return last_conv

    def _register_hooks(self, target_layer_name):
        target = None
        if target_layer_name:
            for name, module in self.model.named_modules():
                if name == target_layer_name:
                    target = module
                    break
        if target is None:
            result = self._find_last_conv(self.model)
            if result:
                target = result[1]
                log.debug(f"GradCAM target layer: {result[0]}")

        if target is None:
            log.warning("No Conv2d/Conv3d found for GradCAM; will use fallback")
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(target.register_forward_hook(forward_hook))
        self._hook_handles.append(target.register_full_backward_hook(backward_hook))

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None,
                 frames_per_video: int = 16):
        """
        Args:
            img_tensor: (1, 3, H, W) — single preprocessed frame
            class_idx:  class to explain (None = predicted class)
        Returns:
            cam:        np.ndarray (H, W) normalized 0-1
            pred_class: int
        """
        self.model.zero_grad()
        self.model.eval()
        # cuDNN LSTM/GRU backward requires the RNN modules to be in train mode.
        # Keep everything else in eval (BN uses running stats) but flip RNN layers.
        for m in self.model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                m.train()

        x = prepare_input(img_tensor, {}, frames_per_video).requires_grad_(True)
        logits = model_forward(self.model, x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            h = img_tensor.shape[-2]
            return np.ones((h, h), dtype=np.float32) * 0.5, class_idx

        acts = self.activations   # (1, C, H', W') or (1, C, T, H', W')
        grads = self.gradients    # same shape

        # For Conv3d: pool temporal dim first → (1, C, H', W')
        if acts.dim() == 5:
            acts  = acts.mean(dim=2)
            grads = grads.mean(dim=2)

        weights = grads.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1)                 # (B, H', W')
        # B may be B*N for frame-level models (EfficientNet etc.) — average over frames
        cam = cam.mean(dim=0)                             # (H', W')
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def overlay(self, original_img: np.ndarray, cam: np.ndarray) -> np.ndarray:
        """Overlay heatmap on original image. Returns BGR uint8."""
        h, w = original_img.shape[:2]
        cam_2d = np.squeeze(cam)           # ensure exactly (H', W')
        if cam_2d.ndim != 2:
            cam_2d = cam_2d.mean(axis=0) if cam_2d.ndim == 3 else cam_2d.ravel()[:h*w].reshape(h, w)
        cam_f = cv2.resize(cam_2d.astype(np.float32), (w, h))
        lo, hi = cam_f.min(), cam_f.max()
        cam_f = (cam_f - lo) / (hi - lo + 1e-8)          # normalise to [0,1]
        cam_u8 = (cam_f * 255).clip(0, 255).astype(np.uint8)  # must be CV_8UC1
        heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        orig_u8 = np.array(original_img, dtype=np.uint8)
        original_bgr = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


# ════════════════════════════════════════════════════════════
# LIME — lightweight superpixel perturbation
# ════════════════════════════════════════════════════════════
class SimpleLIME:
    """
    Lightweight LIME using grid-based superpixels.
    Works with both 2D and 3D models by wrapping each perturbed
    frame into the correct (1, N, 3, H, W) video format.
    """

    def __init__(self, model, transform, device,
                 num_superpixels: int = 50, num_samples: int = 80,
                 frames_per_video: int = 16):
        self.model    = model
        self.transform = transform
        self.device   = device
        self.n_sp     = num_superpixels
        self.n_samples = num_samples
        self.fpv      = frames_per_video

    def _slic_segments(self, img: np.ndarray):
        h, w = img.shape[:2]
        rows = int(np.sqrt(self.n_sp))
        cols = int(np.ceil(self.n_sp / rows))
        segments = np.zeros((h, w), dtype=np.int32)
        rh = h // rows
        rw = w // cols
        idx = 0
        for r in range(rows):
            for c in range(cols):
                y1 = r * rh
                y2 = (r + 1) * rh if r < rows - 1 else h
                x1 = c * rw
                x2 = (c + 1) * rw if c < cols - 1 else w
                segments[y1:y2, x1:x2] = idx
                idx += 1
        return segments, idx

    @torch.no_grad()
    def _predict(self, imgs_np: list, chunk: int = 4) -> np.ndarray:
        """imgs_np: list of (H,W,3) uint8 → class probabilities (N, C).
        Runs in small chunks to avoid OOM on 11 GB GPU."""
        all_probs = []
        for i in range(0, len(imgs_np), chunk):
            tensors = []
            for im in imgs_np[i:i + chunk]:
                t = self.transform(Image.fromarray(im))   # (3, H, W)
                t = t.unsqueeze(0).unsqueeze(0)           # (1, 1, 3, H, W)
                t = t.expand(-1, self.fpv, -1, -1, -1)   # (1, N, 3, H, W)
                tensors.append(t)
            batch = torch.cat(tensors, dim=0).to(self.device)
            logits = model_forward(self.model, batch)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            del batch, logits
            torch.cuda.empty_cache()
        return np.concatenate(all_probs, axis=0)

    def explain(self, img: np.ndarray, class_idx: int, img_size: int = 224):
        img_r = cv2.resize(img, (img_size, img_size))
        segments, n_sp = self._slic_segments(img_r)
        masks = np.random.randint(0, 2, (self.n_samples, n_sp))
        gray_fill = img_r.mean(axis=(0, 1)).astype(np.uint8)

        perturbed = []
        for mask in masks:
            p = img_r.copy()
            for sp_id in range(n_sp):
                if mask[sp_id] == 0:
                    p[segments == sp_id] = gray_fill
            perturbed.append(p)

        probs   = self._predict(perturbed)
        targets = probs[:, class_idx]

        importance_map = np.zeros((img_size, img_size), dtype=np.float32)
        for sp_id in range(n_sp):
            col = masks[:, sp_id].astype(float)
            corr = np.corrcoef(col, targets)[0, 1] if col.std() > 0 else 0.0
            importance_map[segments == sp_id] = corr

        if importance_map.max() > importance_map.min():
            importance_map = (importance_map - importance_map.min()) / (
                importance_map.max() - importance_map.min() + 1e-8)
        return importance_map


# ════════════════════════════════════════════════════════════
# Histogram
# ════════════════════════════════════════════════════════════
def plot_class_histogram(frame_dir: Path, class_name: str, out_path: Path):
    frames = sorted(frame_dir.glob("*.jpg"))[:10]
    if not frames:
        return
    colors = ("red", "green", "blue")
    fig, ax = plt.subplots(figsize=(8, 4))
    for fpath in frames[:5]:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_rgb], [i], None, [64], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            ax.plot(hist, color=col, alpha=0.3)
    ax.set_title(f"Pixel Intensity Histogram — {class_name}")
    ax.set_xlabel("Pixel Value (bin)")
    ax.set_ylabel("Normalized Frequency")
    ax.legend(colors)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════
# Sample collector
# ════════════════════════════════════════════════════════════
def collect_samples(splits_path: str, num_samples: int, split: str = "test"):
    with open(splits_path) as f:
        data = json.load(f)
    entries = data[split]
    random.shuffle(entries)
    samples = []
    for entry in entries:
        frame_dir = Path(entry["frame_dir"])
        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            continue
        samples.append({
            "frame":      frames[len(frames) // 2],   # center frame
            "class_idx":  entry["class_idx"],
            "class_name": entry["class"],
            "frame_dir":  frame_dir,
        })
        if len(samples) >= num_samples:
            break
    return samples


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    fpv        = cfg["training"].get("frames_per_video", 16)
    num_samples = args.num_samples or cfg["xai"]["num_samples"]

    xai_base    = Path(cfg["xai"]["output_dir"])
    weights_dir = Path(cfg["paths"]["weights_dir"])
    splits_path = str(Path(cfg["dataset"]["splits_dir"]) / "splits.json")

    # Pick model — explicit arg > best by F1 > first available
    model_name  = args.model
    metrics_path = Path(cfg["evaluation"]["results_dir"]) / "metrics" / "all_models_metrics.json"
    if not model_name and metrics_path.exists():
        with open(metrics_path) as f:
            all_m = json.load(f)
        model_name = max(all_m, key=lambda k: all_m[k].get("f1_macro", 0))
        log.info(f"Best model selected for XAI: {model_name}")

    enabled_models = [m for m in cfg["models"] if m.get("enabled", True)]
    model_cfg = next((m for m in enabled_models if m["name"] == model_name),
                     enabled_models[0])
    best_path = weights_dir / f"{model_cfg['name']}_best.pt"

    if not best_path.exists():
        log.error(f"No checkpoint at {best_path}. Run evaluate.py first.")
        return

    # Per-model class info: grouped models have 5 classes, others have 14.
    if model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        gc = cfg["dataset"]["grouped_classes"]
        num_classes = len(gc)
        classes = [g["name"] for g in gc]
    else:
        num_classes = cfg["dataset"]["num_classes"]
        classes = cfg["dataset"]["classes"]
    log.info(f"Classes ({num_classes}): {classes}")

    # Per-model output dirs: results/xai/{model_name}/gradcam|lime|histograms
    # Each model gets its own folder — running XAI on a new model never
    # overwrites another model's results.
    model_xai_dir = xai_base / model_cfg["name"]
    gradcam_dir   = model_xai_dir / "gradcam"
    lime_dir      = model_xai_dir / "lime"
    hist_dir      = model_xai_dir / "histograms"
    for d in [gradcam_dir, lime_dir, hist_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log.info(f"Running XAI on: {model_cfg['name']}")
    log.info(f"Output dir   : {model_xai_dir}")
    model    = load_model(model_cfg, num_classes, str(best_path), device)
    img_size = model_cfg.get("img_size", cfg["frames"]["img_size"])
    norm     = model_cfg.get("normalization", "imagenet")
    transform = get_val_transform(img_size, norm)

    samples = collect_samples(splits_path, num_samples)

    # Remap raw class_idx → group_idx for grouped models
    xai_group_mapping = None
    if model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        orig_classes = cfg["dataset"]["classes"]
        xai_group_mapping = build_group_mapping(cfg["dataset"]["grouped_classes"], orig_classes)
    for s in samples:
        if xai_group_mapping is not None:
            s["class_idx"] = xai_group_mapping.get(s["class_idx"], 0)

    # ── GradCAM ──────────────────────────────────────────────
    log.info("\nGenerating GradCAM explanations ...")
    gcam = GradCAM(model)

    for i, s in enumerate(samples):
        try:
            img_pil = Image.open(s["frame"]).convert("RGB").resize((img_size, img_size))
            img_np  = np.array(img_pil)
            img_t   = transform(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)

            cam, pred_cls = gcam.generate(img_t, s["class_idx"], fpv)
            overlay = gcam.overlay(img_np, cam)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Original Frame")
            axes[0].axis("off")

            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title(
                f"Overlay\nTrue: {s['class_name']} | Pred: {classes[pred_cls]}")
            axes[2].axis("off")

            plt.suptitle(f"GradCAM - {model_cfg['name']}", fontsize=11)
            plt.tight_layout()
            plt.savefig(gradcam_dir / f"gradcam_{i:03d}_{s['class_name']}.png",
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            log.warning(f"GradCAM failed for sample {i}: {e}")

    gcam.remove_hooks()
    log.info(f"GradCAM done -> {gradcam_dir}")

    # ── LIME ─────────────────────────────────────────────────
    log.info("\nGenerating LIME explanations ...")
    # 3D models (VideoModel3D) are far slower per forward pass than 2D models.
    # Reduce perturbations and sample count so LIME finishes in reasonable time.
    from src.models.model_builder import VideoModel3D
    is_3d = isinstance(model, VideoModel3D)
    lime_n_samples = 30 if is_3d else 80      # perturbations per image
    lime_n_images  = 5  if is_3d else 10      # how many images to explain
    lime_explainer = SimpleLIME(model, transform, device,
                                num_superpixels=50, num_samples=lime_n_samples,
                                frames_per_video=fpv)
    lime_samples = samples[:lime_n_images]
    log.info(f"  LIME: {lime_n_images} images × {lime_n_samples} perturbations each"
             + (" (reduced for 3D model)" if is_3d else ""))

    for i, s in enumerate(lime_samples):
        try:
            img_pil = Image.open(s["frame"]).convert("RGB").resize((img_size, img_size))
            img_np  = np.array(img_pil)
            importance = lime_explainer.explain(img_np, s["class_idx"], img_size)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Original Frame")
            axes[0].axis("off")

            axes[1].imshow(img_np)
            axes[1].imshow(importance, cmap="RdYlGn", alpha=0.55)
            axes[1].set_title(f"LIME Importance\nClass: {s['class_name']}")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(lime_dir / f"lime_{i:03d}_{s['class_name']}.png",
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            log.warning(f"LIME failed for sample {i}: {e}")
    log.info(f"LIME done -> {lime_dir}")

    # ── Histograms ───────────────────────────────────────────
    log.info("\nGenerating per-class histograms ...")
    with open(splits_path) as f:
        split_data = json.load(f)

    seen = set()
    for entry in split_data["test"]:
        cls = entry["class"]
        if cls in seen:
            continue
        seen.add(cls)
        plot_class_histogram(
            Path(entry["frame_dir"]), cls,
            hist_dir / f"histogram_{cls}.png")

    log.info(f"Histograms done -> {hist_dir}")
    log.info("\n" + "=" * 60)
    log.info("XAI COMPLETE")
    log.info(f"All outputs in: {str(model_xai_dir)}")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Timeline — XAI")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--model",       default=None,
                        help="Model name for XAI (default: best by F1)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--all-models",  action="store_true",
                        help="Run XAI on ALL trained models, not just the best")
    args = parser.parse_args()

    if args.all_models:
        import yaml as _yaml
        with open(args.config) as _f:
            _cfg = _yaml.safe_load(_f)
        _weights_dir = Path(_cfg["paths"]["weights_dir"])
        for _mc in _cfg["models"]:
            if not _mc.get("enabled", True):
                continue
            _ckpt = _weights_dir / f"{_mc['name']}_best.pt"
            if not _ckpt.exists():
                log.warning(f"Skipping {_mc['name']} — no checkpoint")
                continue
            import copy
            _args = copy.copy(args)
            _args.model = _mc["name"]
            _args.all_models = False
            log.info(f"\n{'='*60}\nXAI: {_mc['name']}\n{'='*60}")
            main(_args)
    else:
        main(args)
