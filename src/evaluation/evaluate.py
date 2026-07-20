"""
STEP 3 — Evaluation & Comparative Analysis
============================================
Runs all 7 models on the test set and generates a full comparative report:

Per-model outputs:
  - Confusion matrix (raw counts + normalized)
  - Class-wise metrics bar chart (F1 / Precision / Recall per class)
  - Confidence score distribution (correct vs incorrect)
  - Learning curves (loss, accuracy, val F1)

Comparative outputs (all models together):
  - Metric bar charts (Accuracy, F1, Precision, Recall, ROC-AUC)
  - Radar / spider chart (5 metrics per model on one polar plot)
  - Per-class F1 heatmap (rows=models, cols=crime classes)
  - Training convergence overlay (val accuracy over epochs)
  - ROC curve overlay (macro-averaged OvR per model)
  - Leaderboard table (ranked by F1 Macro)
  - Full HTML report with all charts embedded

Usage:
  python src/evaluation/evaluate.py
  python src/evaluation/evaluate.py --model R2Plus1D
"""

import sys
import json
import yaml
import argparse
import logging
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from math import pi

import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import get_dataloaders
from src.models.model_builder import load_model, build_model, is_temporal_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Consistent color palette across all plots
MODEL_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722",
]


# # ════════════════════════════════════════════════════════════
# # Inference
# # ════════════════════════════════════════════════════════════
# def _forward_probs(model, imgs: torch.Tensor) -> torch.Tensor:
#     """Forward pass → softmax probabilities. Handles temporal model tuple output."""
#     output = model(imgs)
#     if isinstance(output, tuple):
#         logits = output[1]   # CNN-LSTM / CNN-Transformer: use class_logits
#     else:
#         logits = output
#     return torch.softmax(logits, dim=1)


# @torch.no_grad()
# def run_inference(model, loader, device, num_classes,
#                   temporal: bool = False, tta_flips: bool = False):
#     """
#     Run model on test loader.

#     tta_flips: Test-Time Augmentation — averages predictions from:
#       1. Original video frames
#       2. Horizontally flipped video frames
#     Costs 2× inference time, gains +3–5% accuracy for free.
#     Enabled via config: training.tta_flips: true
#     """
#     model.eval()
#     all_preds, all_labels, all_probs = [], [], []

#     for batch in loader:
#         imgs, labels = batch[0], batch[1]
#         imgs = imgs.to(device, non_blocking=True)
#         # Frame-level dataloaders return (B,3,H,W); model expects (B,N,3,H,W)
#         if imgs.dim() == 4:
#             imgs = imgs.unsqueeze(1)

#         with autocast("cuda", enabled=torch.cuda.is_available()):
#             probs = _forward_probs(model, imgs)

#             if tta_flips:
#                 imgs_flipped = torch.flip(imgs, dims=[-1])
#                 probs_flip   = _forward_probs(model, imgs_flipped)
#                 probs        = (probs + probs_flip) * 0.5

#         probs_np = probs.cpu().numpy()
#         all_probs.extend(probs_np)
#         all_preds.extend(np.argmax(probs_np, axis=1))
#         all_labels.extend(labels.numpy())

#     return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_prob, classes):
    num_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        if num_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        metrics["roc_auc"] = None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # Store raw arrays for ROC curve plotting
    metrics["_y_true"] = y_true.tolist()
    metrics["_y_prob"] = y_prob.tolist()

    return metrics


# ════════════════════════════════════════════════════════════
# Per-model plots
# ════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, classes, model_name, out_dir):
    cm_arr  = np.array(cm)
    cm_norm = cm_arr.astype(float) / (cm_arr.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, data, title, fmt in zip(
        axes,
        [cm_arr, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        [".0f", ".2f"]
    ):
        cmap = "Blues" if not HAS_SEABORN else "YlOrRd"
        im = ax.imshow(data, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{model_name} — {title}", pad=10)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=12,
                        color="white" if data[i, j] > thresh else "black")

    plt.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrix → {path.name}")


# def plot_learning_curves(history, model_name, out_dir):
#     if not history or "train_loss" not in history:
#         return
#     n = len(history["train_loss"])
#     epochs = range(1, n + 1)

#     # Detect temporal vs video-level history format
#     is_temporal = "train_frame_acc" in history

#     if is_temporal:
#         fig, axes = plt.subplots(1, 3, figsize=(16, 4))
#         axes[0].plot(epochs, history["train_loss"],     "b-o", label="Train", ms=3)
#         axes[0].plot(epochs, history["val_loss"],       "r-o", label="Val",   ms=3)
#         axes[0].set_title("Loss")

#         axes[1].plot(epochs, history["train_frame_acc"], "b-o", label="Train Frame", ms=3)
#         axes[1].plot(epochs, history["val_frame_acc"],   "r-o", label="Val Frame",   ms=3)
#         axes[1].set_title("Frame Accuracy")

#         axes[2].plot(epochs, history["val_frame_f1"],  "g-o", label="Val Frame F1",  ms=3)
#         axes[2].plot(epochs, history["val_class_f1"],  "m-o", label="Val Class F1",  ms=3)
#         axes[2].set_title("Val F1 Scores")
#     else:
#         fig, axes = plt.subplots(1, 3, figsize=(16, 4))
#         axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", ms=3)
#         axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   ms=3)
#         axes[0].set_title("Loss")

#         axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", ms=3)
#         axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val",   ms=3)
#         axes[1].set_title("Accuracy")

#         axes[2].plot(epochs, history["val_f1"], "g-o", label="Val F1 Macro", ms=3)
#         axes[2].set_title("Val Macro F1")

#     for ax in axes:
#         ax.set_xlabel("Epoch")
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)

#     plt.suptitle(f"Learning Curves — {model_name}", fontsize=13, fontweight="bold")
#     plt.tight_layout()
#     path = out_dir / f"{model_name}_learning_curves.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()


# def plot_class_wise_metrics(report, model_name, classes, out_dir):
#     cls_data = {c: report.get(c, {}) for c in classes}
#     f1s   = [cls_data[c].get("f1-score",  0) for c in classes]
#     precs = [cls_data[c].get("precision", 0) for c in classes]
#     recs  = [cls_data[c].get("recall",    0) for c in classes]

#     x = np.arange(len(classes))
#     w = 0.25
#     fig, ax = plt.subplots(figsize=(17, 6))
#     ax.bar(x - w, f1s,   w, label="F1",        color="#2196F3", alpha=0.85)
#     ax.bar(x,     precs,  w, label="Precision", color="#FF9800", alpha=0.85)
#     ax.bar(x + w, recs,   w, label="Recall",    color="#4CAF50", alpha=0.85)
#     ax.set_xticks(x)
#     ax.set_xticklabels(classes, rotation=45, ha="right")
#     ax.set_ylabel("Score")
#     ax.set_ylim(0, 1.1)
#     ax.set_title(f"{model_name} — Per-Class Metrics (F1 / Precision / Recall)")
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis="y")
#     for i, (f, p, r) in enumerate(zip(f1s, precs, recs)):
#         ax.text(i - w, f + 0.02, f"{f:.2f}", ha="center", fontsize=6, color="#1565C0")
#     plt.tight_layout()
#     path = out_dir / f"{model_name}_classwise_metrics.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()


# def plot_confidence_histogram(y_true, y_pred, y_prob, model_name, out_dir):
#     confidences = y_prob.max(axis=1)
#     correct = (y_pred == y_true)
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.hist(confidences[correct],  bins=30, alpha=0.65, label="Correct",   color="#4CAF50")
#     ax.hist(confidences[~correct], bins=30, alpha=0.65, label="Incorrect", color="#F44336")
#     ax.set_xlabel("Max Softmax Confidence")
#     ax.set_ylabel("Sample Count")
#     ax.set_title(f"{model_name} — Confidence Distribution")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     path = out_dir / f"{model_name}_confidence_hist.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()


# # ════════════════════════════════════════════════════════════
# # Comparative analysis plots (all models together)
# # ════════════════════════════════════════════════════════════
# def plot_comparative_bar(all_metrics, out_dir):
#     """Side-by-side grouped bar chart for 5 key metrics."""
#     names = list(all_metrics.keys())
#     metric_keys   = ["accuracy", "f1_macro", "precision", "recall", "roc_auc"]
#     metric_labels = ["Accuracy", "F1 Macro", "Precision", "Recall", "ROC-AUC"]
#     colors = MODEL_COLORS[:len(names)]

#     fig, axes = plt.subplots(1, len(metric_keys), figsize=(24, 6))
#     for ax, mk, ml in zip(axes, metric_keys, metric_labels):
#         vals = [all_metrics[n].get(mk) or 0 for n in names]
#         bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.8)
#         ax.set_xticks(range(len(names)))
#         ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
#         ax.set_title(ml, fontweight="bold")
#         ax.set_ylim(0, 1.12)
#         ax.grid(True, alpha=0.25, axis="y")
#         for bar, val in zip(bars, vals):
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + 0.015, f"{val:.3f}",
#                     ha="center", va="bottom", fontsize=7, fontweight="bold")
#         # Highlight best
#         best_idx = int(np.argmax(vals))
#         bars[best_idx].set_edgecolor("gold")
#         bars[best_idx].set_linewidth(2.5)

#     plt.suptitle("7-Model Comparative Analysis — Performance Metrics",
#                  fontsize=15, fontweight="bold", y=1.01)
#     plt.tight_layout()
#     path = out_dir / "comparative_bar_chart.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"Comparative bar chart → {path.name}")


# def plot_radar_chart(all_metrics, out_dir):
#     """
#     Spider / radar chart: each model is a polygon over 5 metric axes.
#     Gives an at-a-glance comparison of model profiles.
#     """
#     categories = ["Accuracy", "F1 Macro", "Precision", "Recall", "ROC-AUC"]
#     metric_keys = ["accuracy", "f1_macro", "precision", "recall", "roc_auc"]
#     N = len(categories)
#     angles = [n / float(N) * 2 * pi for n in range(N)]
#     angles += angles[:1]   # close the polygon

#     fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
#     ax.set_theta_offset(pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories, size=11)
#     ax.set_rlabel_position(30)
#     ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#     ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="grey")
#     ax.set_ylim(0, 1)

#     names = list(all_metrics.keys())
#     colors = MODEL_COLORS[:len(names)]

#     for name, color in zip(names, colors):
#         vals = [all_metrics[name].get(mk) or 0 for mk in metric_keys]
#         vals += vals[:1]
#         ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
#         ax.fill(angles, vals, alpha=0.08, color=color)

#     ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
#     ax.set_title("7-Model Radar Chart\n(higher = better on all axes)",
#                  fontsize=13, fontweight="bold", pad=20)

#     plt.tight_layout()
#     path = out_dir / "comparative_radar_chart.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"Radar chart → {path.name}")


# def plot_per_class_heatmap(all_metrics, classes, out_dir):
#     """
#     Heatmap: rows = models, columns = crime classes, values = F1 score.
#     Reveals which architecture excels at which type of anomaly.
#     """
#     names = list(all_metrics.keys())
#     data = np.zeros((len(names), len(classes)))

#     for i, name in enumerate(names):
#         report = all_metrics[name].get("classification_report", {})
#         for j, cls in enumerate(classes):
#             data[i, j] = report.get(cls, {}).get("f1-score", 0.0)

#     fig, ax = plt.subplots(figsize=(20, max(5, len(names) * 0.9)))

#     if HAS_SEABORN:
#         import seaborn as sns
#         sns.heatmap(
#             data, annot=True, fmt=".2f", cmap="RdYlGn",
#             xticklabels=classes, yticklabels=names,
#             linewidths=0.5, linecolor="white",
#             vmin=0, vmax=1, ax=ax,
#             annot_kws={"size": 8}
#         )
#     else:
#         im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
#         plt.colorbar(im, ax=ax, label="F1 Score")
#         ax.set_xticks(range(len(classes)))
#         ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
#         ax.set_yticks(range(len(names)))
#         ax.set_yticklabels(names, fontsize=9)
#         for i in range(len(names)):
#             for j in range(len(classes)):
#                 ax.text(j, i, f"{data[i,j]:.2f}",
#                         ha="center", va="center", fontsize=7,
#                         color="black" if 0.3 < data[i, j] < 0.8 else "white")

#     ax.set_title("Per-Class F1 Score Heatmap — All Models vs All Crime Classes",
#                  fontsize=13, fontweight="bold", pad=12)
#     ax.set_xlabel("Crime Class", labelpad=8)
#     ax.set_ylabel("Model", labelpad=8)

#     plt.tight_layout()
#     path = out_dir / "comparative_per_class_heatmap.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"Per-class heatmap → {path.name}")


# def plot_training_convergence(all_histories, out_dir):
#     """
#     Overlay val accuracy (or val F1) curves for all models on one plot.
#     Shows which model converges fastest and overfits least.
#     """
#     if not all_histories:
#         return

#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     colors = MODEL_COLORS

#     for idx, (name, hist) in enumerate(all_histories.items()):
#         color = colors[idx % len(colors)]
#         is_temporal = "val_frame_f1" in hist

#         epochs = range(1, len(hist["train_loss"]) + 1)

#         # Val loss
#         axes[0].plot(epochs, hist["val_loss"], "-", color=color,
#                      label=name, linewidth=1.8, alpha=0.85)

#         # Val accuracy or val F1
#         if is_temporal:
#             val_metric = hist.get("val_class_f1", hist.get("val_frame_f1", []))
#             ylabel = "Val F1 Score"
#         else:
#             val_metric = hist.get("val_acc", hist.get("val_f1", []))
#             ylabel = "Val Accuracy"

#         if val_metric:
#             axes[1].plot(range(1, len(val_metric) + 1), val_metric,
#                          "-", color=color, label=name, linewidth=1.8, alpha=0.85)

#     axes[0].set_title("Validation Loss — All Models", fontweight="bold")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend(fontsize=8)
#     axes[0].grid(True, alpha=0.3)

#     axes[1].set_title("Validation Accuracy / F1 — All Models", fontweight="bold")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel(ylabel)
#     axes[1].legend(fontsize=8)
#     axes[1].grid(True, alpha=0.3)

#     plt.suptitle("Training Convergence Comparison", fontsize=13, fontweight="bold")
#     plt.tight_layout()
#     path = out_dir / "comparative_convergence.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"Convergence plot → {path.name}")


# def plot_roc_comparison(all_metrics, classes, out_dir):
#     """
#     Overlaid macro-averaged OvR ROC curves for all models.
#     AUC score shown in legend for quick comparison.
#     """
#     num_classes = len(classes)
#     fig, ax = plt.subplots(figsize=(9, 7))
#     colors = MODEL_COLORS

#     for idx, (name, metrics) in enumerate(all_metrics.items()):
#         y_true = np.array(metrics.get("_y_true", []))
#         y_prob = np.array(metrics.get("_y_prob", []))
#         if len(y_true) == 0:
#             continue

#         y_bin = label_binarize(y_true, classes=list(range(num_classes)))
#         fpr_all, tpr_all = [], []
#         for c in range(num_classes):
#             if y_bin[:, c].sum() == 0:
#                 continue
#             fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
#             fpr_all.append(fpr)
#             tpr_all.append(tpr)

#         if not fpr_all:
#             continue

#         # Interpolate to common FPR axis for macro average
#         mean_fpr = np.linspace(0, 1, 200)
#         mean_tpr = np.mean([np.interp(mean_fpr, f, t)
#                             for f, t in zip(fpr_all, tpr_all)], axis=0)
#         roc_auc = metrics.get("roc_auc") or auc(mean_fpr, mean_tpr)

#         color = colors[idx % len(colors)]
#         ax.plot(mean_fpr, mean_tpr, "-", color=color, linewidth=2.0,
#                 label=f"{name}  (AUC={roc_auc:.3f})", alpha=0.85)

#     ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title("ROC Curves — Macro-Averaged OvR (All Models)",
#                  fontweight="bold")
#     ax.legend(fontsize=9, loc="lower right")
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1.02])

#     plt.tight_layout()
#     path = out_dir / "comparative_roc_curves.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"ROC comparison → {path.name}")


# def plot_per_class_bestmodel(all_metrics, classes, out_dir):
#     """
#     Three heatmaps side-by-side: Precision / Recall / F1 per class per model.
#     Plus a 'Best Model' summary table showing which model leads for each class.
#     """
#     names = list(all_metrics.keys())
#     metrics_keys = [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1")]
#     n_models  = len(names)
#     n_classes = len(classes)

#     # Build (3, n_models, n_classes) array
#     data = np.zeros((3, n_models, n_classes))
#     for mi, name in enumerate(names):
#         report = all_metrics[name].get("classification_report", {})
#         for ci, cls in enumerate(classes):
#             for ki, (key, _) in enumerate(metrics_keys):
#                 data[ki, mi, ci] = report.get(cls, {}).get(key, 0.0)

#     # ── Heatmaps ─────────────────────────────────────────────
#     fig, axes = plt.subplots(3, 1, figsize=(22, 4 + n_models * 1.2))
#     fig.suptitle("Per-Class Metrics — All Models vs All Crime Classes",
#                  fontsize=14, fontweight="bold")

#     for ki, (_, label) in enumerate(metrics_keys):
#         ax = axes[ki]
#         if HAS_SEABORN:
#             import seaborn as sns
#             sns.heatmap(data[ki], annot=True, fmt=".2f", cmap="RdYlGn",
#                         xticklabels=classes, yticklabels=names,
#                         linewidths=0.4, linecolor="white",
#                         vmin=0, vmax=1, ax=ax, annot_kws={"size": 7})
#         else:
#             im = ax.imshow(data[ki], cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
#             plt.colorbar(im, ax=ax)
#             ax.set_xticks(range(n_classes))
#             ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
#             ax.set_yticks(range(n_models))
#             ax.set_yticklabels(names, fontsize=8)
#             for mi in range(n_models):
#                 for ci in range(n_classes):
#                     ax.text(ci, mi, f"{data[ki,mi,ci]:.2f}",
#                             ha="center", va="center", fontsize=6,
#                             color="black" if 0.3 < data[ki,mi,ci] < 0.8 else "white")
#         ax.set_title(f"{label} per Class", fontsize=10, fontweight="bold")
#         ax.set_ylabel("Model")
#         if ki == 2:
#             ax.set_xlabel("Crime Class")

#     plt.tight_layout()
#     path = out_dir / "comparative_per_class_all_metrics.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"Per-class precision/recall/F1 heatmap -> {path.name}")

#     # ── Best Model Per Class table ────────────────────────────
#     fig2, ax2 = plt.subplots(figsize=(22, max(3, n_classes * 0.5 + 2)))
#     ax2.axis("off")

#     rows = []
#     cell_colors = []
#     for ci, cls in enumerate(classes):
#         row = [cls]
#         row_c = ["#f5f5f5"]
#         for ki, (_, label) in enumerate(metrics_keys):
#             best_mi = int(data[ki, :, ci].argmax())
#             best_val = data[ki, best_mi, ci]
#             row.append(f"{names[best_mi]}  ({best_val:.2f})")
#             row_c.append("#C8E6C9" if best_val >= 0.5 else
#                          "#FFF9C4" if best_val >= 0.25 else "#FFCDD2")
#         rows.append(row)
#         cell_colors.append(row_c)

#     table = ax2.table(
#         cellText=rows,
#         colLabels=["Class", "Best Precision", "Best Recall", "Best F1"],
#         cellColours=cell_colors,
#         colColours=["#1565C0"] * 4,
#         cellLoc="center",
#         loc="center",
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(8)
#     table.scale(1.2, 1.5)
#     for j in range(4):
#         table[0, j].set_text_props(color="white", fontweight="bold")

#     ax2.set_title(
#         "Best Model Per Class  (green >= 0.50 | yellow >= 0.25 | red < 0.25)",
#         fontsize=11, fontweight="bold", pad=12)
#     plt.tight_layout()
#     path2 = out_dir / "comparative_best_model_per_class.png"
#     plt.savefig(path2, bbox_inches="tight")
#     plt.close()
#     log.info(f"Best model per class table -> {path2.name}")


# def plot_leaderboard_table(all_metrics, out_dir):
#     """
#     Styled table image showing ranked models with color-coded scores.
#     Best value per column highlighted in green.
#     """
#     names = list(all_metrics.keys())
#     cols  = ["Accuracy", "F1 Macro", "F1 Weighted", "Precision", "Recall", "ROC-AUC"]
#     keys  = ["accuracy", "f1_macro", "f1_weighted",  "precision", "recall", "roc_auc"]

#     # Build data matrix
#     data = []
#     for name in names:
#         row = [all_metrics[name].get(k) or 0 for k in keys]
#         data.append(row)
#     data = np.array(data, dtype=float)

#     # Sort by F1 Macro
#     order = np.argsort(data[:, 1])[::-1]
#     sorted_names = [f"#{i+1} {names[o]}" for i, o in enumerate(order)]
#     sorted_data  = data[order]

#     fig, ax = plt.subplots(figsize=(14, max(3, len(names) * 0.6 + 1.5)))
#     ax.axis("off")

#     cell_text  = [[f"{v:.4f}" for v in row] for row in sorted_data]
#     col_colors = ["#1565C0"] * len(cols)
#     row_colors = [["#f5f5f5" if i % 2 == 0 else "#ffffff"] * len(cols)
#                   for i in range(len(sorted_names))]

#     # Highlight best per column
#     for j in range(len(cols)):
#         best_i = int(np.argmax(sorted_data[:, j]))
#         row_colors[best_i][j] = "#C8E6C9"

#     table = ax.table(
#         cellText=cell_text,
#         rowLabels=sorted_names,
#         colLabels=cols,
#         cellColours=row_colors,
#         colColours=col_colors,
#         cellLoc="center",
#         loc="center",
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(9)
#     table.scale(1.2, 1.6)

#     # Style header
#     for j in range(len(cols)):
#         table[0, j].set_text_props(color="white", fontweight="bold")

#     ax.set_title("Model Leaderboard — Ranked by F1 Macro  (green = best per metric)",
#                  fontsize=11, fontweight="bold", pad=12)
#     plt.tight_layout()
#     path = out_dir / "comparative_leaderboard.png"
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     log.info(f"Leaderboard table → {path.name}")


# # ════════════════════════════════════════════════════════════
# # HTML Report
# # ════════════════════════════════════════════════════════════
# def generate_html_report(all_metrics, classes, out_dir, plots_dir):
#     names  = list(all_metrics.keys())
#     ranked = sorted(all_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True)

#     # Build leaderboard rows
#     table_rows = ""
#     for rank, (name, m) in enumerate(ranked, 1):
#         roc     = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"
#         is_best = rank == 1
#         cls_str = "class='best'" if is_best else ""
#         table_rows += f"""
#         <tr {cls_str}>
#           <td><b>#{rank}</b></td>
#           <td><b>{name}</b></td>
#           <td>{m['accuracy']:.4f}</td>
#           <td>{m['precision']:.4f}</td>
#           <td>{m['recall']:.4f}</td>
#           <td>{m['f1_macro']:.4f}</td>
#           <td>{m['f1_weighted']:.4f}</td>
#           <td>{roc}</td>
#         </tr>"""

#     # Per-model sections
#     model_sections = ""
#     for name, m in all_metrics.items():
#         plots_rel = "plots"
#         model_sections += f"""
#         <div class="model-section">
#           <h3>{name}</h3>
#           <div class="metric-pills">
#             <span class="pill">Acc: {m['accuracy']:.4f}</span>
#             <span class="pill">F1: {m['f1_macro']:.4f}</span>
#             <span class="pill">ROC-AUC: {m.get('roc_auc') and f"{m['roc_auc']:.4f}" or 'N/A'}</span>
#           </div>
#           <div class="img-row">
#             <img src="{plots_rel}/{name}_confusion_matrix.png" alt="CM" />
#           </div>
#           <div class="img-row">
#             <img src="{plots_rel}/{name}_classwise_metrics.png" alt="Class Metrics" />
#           </div>
#           <div class="img-row-half">
#             <img src="{plots_rel}/{name}_confidence_hist.png" alt="Confidence" />
#             <img src="{plots_rel}/{name}_learning_curves.png" alt="Curves" />
#           </div>
#         </div>"""

#     html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8" />
# <title>Evidence Timeline Reconstruction — 7-Model Evaluation Report</title>
# <style>
#   *, *::before, *::after {{ box-sizing: border-box; }}
#   body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 40px;
#          background: #f0f2f5; color: #2c3e50; }}
#   h1 {{ color: #1a237e; font-size: 2em; margin-bottom: 4px; }}
#   h2 {{ color: #283593; border-left: 5px solid #3f51b5;
#          padding-left: 12px; margin-top: 40px; }}
#   h3 {{ color: #37474f; margin: 12px 0 6px; }}
#   .subtitle {{ color: #607d8b; margin-bottom: 30px; font-size: 0.95em; }}
#   table {{ border-collapse: collapse; width: 100%; margin: 16px 0;
#            background: white; border-radius: 8px; overflow: hidden;
#            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
#   th {{ background: #3f51b5; color: white; padding: 12px 16px; text-align: center; }}
#   td {{ padding: 9px 14px; border-bottom: 1px solid #eceff1; text-align: center; }}
#   tr:last-child td {{ border-bottom: none; }}
#   tr:hover td {{ background: #e8eaf6; }}
#   .best td {{ background: #e8f5e9 !important; font-weight: bold; }}
#   img {{ max-width: 100%; border-radius: 6px; border: 1px solid #ddd;
#          box-shadow: 0 2px 6px rgba(0,0,0,0.07); }}
#   .img-row {{ margin: 12px 0; }}
#   .img-row-half {{ display: flex; gap: 16px; margin: 12px 0; }}
#   .img-row-half img {{ width: 50%; }}
#   .model-section {{ background: white; border-radius: 10px; padding: 24px;
#                     margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
#   .metric-pills {{ display: flex; gap: 10px; margin: 8px 0 16px; flex-wrap: wrap; }}
#   .pill {{ background: #e8eaf6; color: #283593; padding: 5px 14px;
#            border-radius: 20px; font-size: 0.88em; font-weight: bold; }}
#   .section-card {{ background: white; border-radius: 10px; padding: 24px;
#                    margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
#   .toc a {{ display: inline-block; margin: 4px 8px; color: #3f51b5;
#             text-decoration: none; font-weight: bold; }}
#   .toc a:hover {{ text-decoration: underline; }}
# </style>
# </head>
# <body>
# <h1>Evidence Timeline Reconstruction System</h1>
# <p class="subtitle">UCF-Crime Dataset | {len(classes)} Classes | 7-Model Comparative Analysis</p>

# <div class="toc">
#   <b>Jump to:</b>
#   <a href="#overview">Overview</a>
#   <a href="#leaderboard">Leaderboard</a>
#   <a href="#radar">Radar Chart</a>
#   <a href="#heatmap">Class Heatmap</a>
#   <a href="#roc">ROC Curves</a>
#   <a href="#convergence">Convergence</a>
#   <a href="#models">Per-Model Detail</a>
# </div>

# <!-- ── Overview ── -->
# <h2 id="overview">Comparative Overview</h2>
# <div class="section-card">
#   <img src="plots/comparative_bar_chart.png" alt="Comparative Bar Chart" />
# </div>

# <!-- ── Leaderboard ── -->
# <h2 id="leaderboard">Model Leaderboard</h2>
# <div class="section-card">
#   <img src="plots/comparative_leaderboard.png" alt="Leaderboard" />
#   <table>
#     <tr>
#       <th>Rank</th><th>Model</th><th>Accuracy</th><th>Precision</th>
#       <th>Recall</th><th>F1 Macro</th><th>F1 Weighted</th><th>ROC-AUC</th>
#     </tr>
#     {table_rows}
#   </table>
# </div>

# <!-- ── Radar Chart ── -->
# <h2 id="radar">Radar Chart</h2>
# <div class="section-card">
#   <img src="plots/comparative_radar_chart.png" alt="Radar Chart" />
# </div>

# <!-- ── Per-Class Heatmap ── -->
# <h2 id="heatmap">Per-Class F1 Heatmap</h2>
# <div class="section-card">
#   <img src="plots/comparative_per_class_heatmap.png" alt="Heatmap" />
# </div>

# <!-- ── Best Model Per Class ── -->
# <h2 id="bestmodel">Best Model Per Class (Precision / Recall / F1)</h2>
# <div class="section-card">
#   <img src="plots/comparative_per_class_all_metrics.png" alt="Per-class all metrics" />
#   <img src="plots/comparative_best_model_per_class.png" alt="Best model per class" />
# </div>

# <!-- ── ROC Curves ── -->
# <h2 id="roc">ROC Curves (Macro-Averaged OvR)</h2>
# <div class="section-card">
#   <img src="plots/comparative_roc_curves.png" alt="ROC Curves" />
# </div>

# <!-- ── Convergence ── -->
# <h2 id="convergence">Training Convergence</h2>
# <div class="section-card">
#   <img src="plots/comparative_convergence.png" alt="Convergence" />
# </div>

# <!-- ── Per-Model ── -->
# <h2 id="models">Per-Model Analysis</h2>
# {model_sections}

# </body></html>"""

#     path = out_dir / "evaluation_report.html"
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(html)
#     log.info(f"HTML report → {path}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"

    results_dir = Path(cfg["evaluation"]["results_dir"])
    plots_dir   = results_dir / "plots"
    metrics_dir = results_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logs_dir    = Path(cfg["paths"]["logs_dir"])
    weights_dir = Path(cfg["paths"]["weights_dir"])

    # Only process models that are enabled (no enabled:false in config)
    models_cfg = [m for m in cfg["models"] if m.get("enabled", True)]
    if args.model:
        models_cfg = [m for m in models_cfg if m["name"] == args.model]
        if not models_cfg:
            available = [m["name"] for m in cfg["models"] if m.get("enabled", True)]
            log.error(f"Model '{args.model}' not found. Available: {available}")
            return

    # Determine the "comparative" class set for cross-model plots.
    # If any enabled model is grouped, use group names; otherwise use original 14 classes.
    _any_grouped = any(
        mc.get("grouped") and cfg["dataset"].get("grouped_classes")
        for mc in models_cfg)
    if _any_grouped:
        comp_classes = [g["name"] for g in cfg["dataset"]["grouped_classes"]]
    else:
        comp_classes = cfg["dataset"]["classes"]

    all_metrics   = {}
    all_histories = {}

    for mc in models_cfg:
        model_name = mc["name"]
        best_path  = weights_dir / f"{model_name}_best.pt"

        if not best_path.exists():
            log.warning(f"No checkpoint for {model_name} — skipping")
            continue

        # Per-model class info: grouped models have 5 classes, others have 14.
        if mc.get("grouped") and cfg["dataset"].get("grouped_classes"):
            gc = cfg["dataset"]["grouped_classes"]
            m_num_classes = len(gc)
            m_classes = [g["name"] for g in gc]
        else:
            m_num_classes = cfg["dataset"]["num_classes"]
            m_classes = cfg["dataset"]["classes"]

        log.info(f"\n{'='*55}")
        log.info(f"Evaluating: {model_name}  ({m_num_classes} classes: {m_classes})")
        log.info(f"{'='*55}")

        _, _, test_dl = get_dataloaders(cfg, mc)
        model = load_model(mc, m_num_classes, str(best_path), device)

        temporal   = is_temporal_model(mc)
        tta_flips  = cfg["training"].get("tta_flips", False)
        y_true, y_pred, y_prob = run_inference(
            model, test_dl, device, m_num_classes,
            temporal=temporal, tta_flips=tta_flips)
        if tta_flips:
            log.info("  TTA: horizontal flip averaging enabled")

        metrics = compute_metrics(y_true, y_pred, y_prob, m_classes)
        all_metrics[model_name] = metrics

        log.info(f"  Accuracy    : {metrics['accuracy']:.4f}")
        log.info(f"  F1 Macro    : {metrics['f1_macro']:.4f}")
        log.info(f"  F1 Weighted : {metrics['f1_weighted']:.4f}")
        log.info(f"  ROC-AUC     : {metrics.get('roc_auc')}")

        # Per-model plots
        plot_confusion_matrix(metrics["confusion_matrix"], m_classes, model_name, plots_dir)
        plot_class_wise_metrics(metrics["classification_report"], model_name, m_classes, plots_dir)
        plot_confidence_histogram(y_true, y_pred, y_prob, model_name, plots_dir)

        # Learning curves
        hist_path = logs_dir / f"{model_name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            all_histories[model_name] = hist
            plot_learning_curves(hist, model_name, plots_dir)

        # Save per-model metrics JSON (exclude raw arrays + cm)
        saveable = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "_y_true", "_y_prob")}
        with open(metrics_dir / f"{model_name}_metrics.json", "w") as f:
            json.dump(saveable, f, indent=2)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_metrics) < 2:
        log.warning("Fewer than 2 models evaluated — skipping comparative plots.")
    else:
        log.info("\nGenerating comparative analysis plots...")
        plot_comparative_bar(all_metrics, plots_dir)
        plot_radar_chart(all_metrics, plots_dir)
        plot_per_class_heatmap(all_metrics, comp_classes, plots_dir)
        plot_per_class_bestmodel(all_metrics, comp_classes, plots_dir)
        plot_roc_comparison(all_metrics, comp_classes, plots_dir)
        plot_leaderboard_table(all_metrics, plots_dir)
        if all_histories:
            plot_training_convergence(all_histories, plots_dir)

    # Master metrics JSON (no raw arrays)
    master = {
        n: {k: v for k, v in m.items()
            if k not in ("classification_report", "confusion_matrix", "_y_true", "_y_prob")}
        for n, m in all_metrics.items()
    }
    with open(metrics_dir / "all_models_metrics.json", "w") as f:
        json.dump(master, f, indent=2)

    generate_html_report(all_metrics, comp_classes, results_dir, plots_dir)

    # Console leaderboard
    log.info("\n" + "="*60)
    log.info("LEADERBOARD (ranked by F1 Macro)")
    log.info("="*60)
    ranked = sorted(all_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    for rank, (name, m) in enumerate(ranked, 1):
        roc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A "
        log.info(f"  #{rank} {name:20s} Acc={m['accuracy']:.4f}  "
                 f"F1={m['f1_macro']:.4f}  AUC={roc}")
    log.info("="*60)
    log.info("Run next: python src/evaluation/xai.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Timeline — Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model",  default=None,
                        help="Evaluate a single model by name")
    args = parser.parse_args()
    main(args)
