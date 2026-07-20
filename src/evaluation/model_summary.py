"""
Model Summary — Comprehensive per-model statistics
====================================================
Generates a detailed summary for every trained model including:
  - Architecture: backbone, head, parameter counts by group
  - Trainable vs frozen parameters
  - Memory footprint (FP32 / FP16)
  - Config: img_size, normalization, dropout, LR, batch_size, etc.
  - Training: epochs, convergence, best val F1/acc, total time
  - Evaluation: test accuracy, F1, precision, recall, ROC-AUC
  - Per-class breakdown: best / worst class for each model
  - Comparative table: side-by-side all models

Output:
  results/model_summary/model_summary.json     ← machine-readable
  results/model_summary/model_summary.html     ← browser report
  results/model_summary/{name}_summary.json    ← per-model

Usage:
  python src/evaluation/model_summary.py
  python src/evaluation/model_summary.py --model EfficientNetB3
"""

import sys
import json
import yaml
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.model_builder import build_model, is_temporal_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Parameter counting ───────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable

    # Group by component
    groups = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        groups.setdefault(top, {"total": 0, "trainable": 0})
        groups[top]["total"] += param.numel()
        if param.requires_grad:
            groups[top]["trainable"] += param.numel()

    return {
        "total":        total,
        "trainable":    trainable,
        "frozen":       frozen,
        "trainable_pct": round(100.0 * trainable / max(1, total), 1),
        "memory_fp32_mb": round(total * 4 / 1024**2, 1),
        "memory_fp16_mb": round(total * 2 / 1024**2, 1),
        "by_group": {
            k: {"total": v["total"],
                "trainable": v["trainable"],
                "frozen": v["total"] - v["trainable"]}
            for k, v in groups.items()
        },
    }


def get_arch_str(model: nn.Module, max_lines: int = 40) -> str:
    lines = str(model).split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines)-max_lines} more lines)"]
    return "\n".join(lines)


def get_layer_table(model: nn.Module) -> list:
    """Return list of {name, type, shape, params, trainable} for named modules."""
    rows = []
    for name, module in model.named_modules():
        if not list(module.children()):   # leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params == 0:
                continue
            trainable = sum(p.numel() for p in module.parameters()
                            if p.requires_grad)
            rows.append({
                "name":      name,
                "type":      type(module).__name__,
                "params":    params,
                "trainable": trainable,
                "frozen":    params - trainable,
            })
    return rows


# ── Load training history ────────────────────────────────────
def load_history(model_name: str, logs_dir: Path) -> dict:
    path = logs_dir / f"{model_name}_history.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def summarise_history(history: dict) -> dict:
    if not history:
        return {}

    is_temporal = "val_frame_f1" in history

    if is_temporal:
        # Use class-level F1 (val_class_f1) as the primary metric for display
        # so it aligns with the val_f1 key used by non-temporal models.
        val_metric = history.get("val_class_f1", history.get("val_frame_f1", []))
        metric_name = "val_f1"   # normalise key so summary tables show it
    else:
        val_metric = history.get("val_f1", history.get("val_acc", []))
        metric_name = "val_f1" if "val_f1" in history else "val_acc"

    if not val_metric:
        return {}

    best_idx  = int(max(range(len(val_metric)), key=lambda i: val_metric[i]))
    best_val  = float(val_metric[best_idx])

    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])

    # Convergence: first epoch where val_metric reached 90% of best
    conv_epoch = next(
        (i + 1 for i, v in enumerate(val_metric) if v >= 0.9 * best_val),
        len(val_metric)
    )

    # ── Training time ─────────────────────────────────────────
    epoch_times = history.get("epoch_time_sec", [])
    total_sec   = history.get("total_time_sec") or (
        sum(epoch_times) if epoch_times else None)
    avg_epoch_sec = (sum(epoch_times) / len(epoch_times)
                     if epoch_times else None)

    def _fmt_sec(sec):
        if not sec:
            return None
        h, rem = divmod(int(sec), 3600)
        m, s   = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    return {
        "epochs_trained":   len(train_loss),
        "best_epoch":       best_idx + 1,
        f"best_{metric_name}": round(best_val, 4),
        "best_train_loss":  round(float(train_loss[best_idx]), 4) if train_loss else None,
        "best_val_loss":    round(float(val_loss[best_idx]), 4) if val_loss else None,
        "final_train_loss": round(float(train_loss[-1]), 4) if train_loss else None,
        "final_val_loss":   round(float(val_loss[-1]), 4) if val_loss else None,
        "convergence_epoch": conv_epoch,
        "overfit_gap": round(
            float(train_loss[-1]) - float(val_loss[-1]), 4
        ) if train_loss and val_loss else None,
        "total_training_time": _fmt_sec(total_sec),
        "total_training_sec":  round(total_sec, 1) if total_sec else None,
        "avg_epoch_time":      _fmt_sec(avg_epoch_sec),
        "avg_epoch_sec":       round(avg_epoch_sec, 1) if avg_epoch_sec else None,
    }


# ── Load evaluation metrics ──────────────────────────────────
def load_eval_metrics(model_name: str, metrics_dir: Path) -> dict:
    path = metrics_dir / f"{model_name}_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def summarise_eval(metrics: dict, classes: list) -> dict:
    if not metrics:
        return {}

    report = metrics.get("classification_report", {})

    # Per-class F1
    class_f1 = {cls: round(report.get(cls, {}).get("f1-score", 0.0), 3)
                for cls in classes}
    class_prec = {cls: round(report.get(cls, {}).get("precision", 0.0), 3)
                  for cls in classes}
    class_rec  = {cls: round(report.get(cls, {}).get("recall", 0.0), 3)
                  for cls in classes}

    best_f1_class  = max(class_f1,  key=class_f1.get)  if class_f1  else None
    worst_f1_class = min(class_f1,  key=class_f1.get)  if class_f1  else None

    return {
        "accuracy":      round(metrics.get("accuracy",    0), 4),
        "f1_macro":      round(metrics.get("f1_macro",    0), 4),
        "f1_weighted":   round(metrics.get("f1_weighted", 0), 4),
        "precision":     round(metrics.get("precision",   0), 4),
        "recall":        round(metrics.get("recall",      0), 4),
        "roc_auc":       round(metrics.get("roc_auc") or 0, 4),
        "per_class_f1":      class_f1,
        "per_class_precision": class_prec,
        "per_class_recall":    class_rec,
        "best_class_f1":  {"class": best_f1_class,  "score": class_f1.get(best_f1_class, 0)},
        "worst_class_f1": {"class": worst_f1_class, "score": class_f1.get(worst_f1_class, 0)},
    }


# ── Build per-model summary ──────────────────────────────────
def build_model_summary(model_cfg: dict, cfg: dict,
                         logs_dir: Path, metrics_dir: Path,
                         weights_dir: Path, classes: list) -> dict:
    name      = model_cfg["name"]
    temporal  = is_temporal_model(model_cfg)

    # Per-model class count: grouped models have 5, others have 14
    if model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        gc = cfg["dataset"]["grouped_classes"]
        num_cls = len(gc)
        classes = [g["name"] for g in gc]
    else:
        num_cls = cfg["dataset"]["num_classes"]

    log.info(f"  Summarising {name} ({num_cls} classes)...")

    # Build model (CPU, pretrained=False for speed)
    cfg_copy = dict(model_cfg)
    cfg_copy["pretrained"] = False
    model = build_model(cfg_copy, num_cls)
    model.eval()

    params   = count_params(model)
    arch_str = get_arch_str(model)
    layers   = get_layer_table(model)

    # Training config
    tr_cfg = cfg["training"]
    train_config = {
        "learning_rate":    tr_cfg.get("learning_rate"),
        "weight_decay":     tr_cfg.get("weight_decay"),
        "batch_size":       model_cfg.get("batch_size", tr_cfg.get("batch_size")),
        "epochs_max":       tr_cfg.get("epochs"),
        "early_stopping":   tr_cfg.get("early_stopping_patience"),
        "scheduler":        tr_cfg.get("scheduler"),
        "mixed_precision":  tr_cfg.get("mixed_precision"),
        "use_class_weights":tr_cfg.get("use_class_weights"),
        "backbone_lr_scale":model_cfg.get("backbone_lr_scale", 0.1),
        "mixup_alpha":      model_cfg.get("mixup_alpha",
                            tr_cfg.get("mixup_alpha", 0.0)),
        "freeze_epochs":    model_cfg.get("freeze_epochs", 0),
        "frame_level":      model_cfg.get("frame_level", False),
        "fl_frames":        model_cfg.get("fl_frames", None),
        "tta_flips":        tr_cfg.get("tta_flips", False),
    }

    # Model config
    model_config = {
        "type":          model_cfg.get("type"),
        "backbone":      model_cfg.get("backbone", "custom"),
        "pretrained":    model_cfg.get("pretrained", True),
        "img_size":      model_cfg.get("img_size", cfg["frames"]["img_size"]),
        "normalization": model_cfg.get("normalization", "imagenet"),
        "dropout":       model_cfg.get("dropout", 0.5),
        "freeze_backbone": model_cfg.get("freeze_backbone", False),
        "sampling":      model_cfg.get("sampling", "sparse"),
        "input_shape":   (f"(B, {tr_cfg.get('frames_per_video', 16)}, 3, "
                          f"{model_cfg.get('img_size', 224)}, "
                          f"{model_cfg.get('img_size', 224)})"),
        "temporal_model": temporal,
    }

    # Checkpoint info
    best_path = weights_dir / f"{name}_best.pt"
    ckpt_info = {"exists": best_path.exists()}
    if best_path.exists():
        ckpt_info["size_mb"] = round(best_path.stat().st_size / 1024**2, 1)
        try:
            ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
            ckpt_info["saved_epoch"] = ckpt.get("epoch", "?")
            ckpt_info["saved_val_f1"] = round(float(ckpt.get("val_f1", 0)), 4)
        except Exception:
            pass

    history  = load_history(name, logs_dir)
    eval_m   = load_eval_metrics(name, metrics_dir)

    summary = {
        "name":          name,
        "parameters":    params,
        "model_config":  model_config,
        "train_config":  train_config,
        "checkpoint":    ckpt_info,
        "training":      summarise_history(history),
        "evaluation":    summarise_eval(eval_m, classes),
        "architecture":  arch_str,
        "layer_table":   layers,
    }

    del model
    return summary


# ── HTML generator ───────────────────────────────────────────
def _fmt(v, pct=False, bold_threshold=None):
    if v is None:
        return "<span style='color:#999'>N/A</span>"
    if isinstance(v, float):
        s = f"{v:.4f}" if not pct else f"{v:.1f}%"
        if bold_threshold and v >= bold_threshold:
            return f"<b style='color:#2e7d32'>{s}</b>"
        return s
    return str(v)


def _param_fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def generate_html(all_summaries: dict, classes: list, out_path: Path):
    names = list(all_summaries.keys())

    # ── Comparative table ─────────────────────────────────────
    comp_rows = ""
    for name in names:
        s  = all_summaries[name]
        p  = s["parameters"]
        tr = s["training"]
        ev = s["evaluation"]
        mc = s["model_config"]

        acc  = ev.get("accuracy", None)
        f1   = ev.get("f1_macro", None)
        ckpt = s["checkpoint"]

        comp_rows += f"""
        <tr>
          <td><b>{name}</b></td>
          <td>{mc.get('backbone','—')}</td>
          <td>{mc.get('type','—')}</td>
          <td>{_param_fmt(p['total'])}</td>
          <td>{_param_fmt(p['trainable'])}</td>
          <td>{p['trainable_pct']}%</td>
          <td>{p['memory_fp32_mb']} MB</td>
          <td>{_fmt(tr.get('epochs_trained'))}</td>
          <td>{tr.get('total_training_time', '—')}</td>
          <td>{tr.get('avg_epoch_time', '—')}</td>
          <td>{_fmt(tr.get('best_val_f1', tr.get('best_val_acc')), bold_threshold=0.5)}</td>
          <td>{_fmt(acc, bold_threshold=0.5)}</td>
          <td>{_fmt(f1, bold_threshold=0.5)}</td>
          <td>{'✓' if ckpt.get('exists') else '✗'}</td>
        </tr>"""

    # ── Per-class best model table ────────────────────────────
    class_rows = ""
    for cls in classes:
        best_name, best_f1 = "—", 0.0
        best_prec_name, best_prec = "—", 0.0
        best_rec_name, best_rec = "—", 0.0
        for name in names:
            ev = all_summaries[name].get("evaluation", {})
            f1v = ev.get("per_class_f1", {}).get(cls, 0.0)
            pv  = ev.get("per_class_precision", {}).get(cls, 0.0)
            rv  = ev.get("per_class_recall", {}).get(cls, 0.0)
            if f1v  > best_f1:   best_f1,   best_name       = f1v,  name
            if pv   > best_prec: best_prec,  best_prec_name  = pv,   name
            if rv   > best_rec:  best_rec,   best_rec_name   = rv,   name

        def cell(model, val):
            color = "#2e7d32" if val >= 0.5 else "#f57f17" if val >= 0.25 else "#c62828"
            return f"<td><span style='color:{color}'><b>{model}</b> ({val:.2f})</span></td>"

        class_rows += f"<tr><td><b>{cls}</b></td>{cell(best_name,best_f1)}{cell(best_prec_name,best_prec)}{cell(best_rec_name,best_rec)}</tr>"

    # ── Per-model detail sections ─────────────────────────────
    model_sections = ""
    for name in names:
        s  = all_summaries[name]
        p  = s["parameters"]
        tr = s["training"]
        ev = s["evaluation"]
        mc = s["model_config"]
        tc = s["train_config"]

        # Parameter group breakdown
        group_rows = ""
        for grp, vals in p.get("by_group", {}).items():
            group_rows += f"""
            <tr>
              <td>{grp}</td>
              <td>{_param_fmt(vals['total'])}</td>
              <td>{_param_fmt(vals['trainable'])}</td>
              <td>{_param_fmt(vals['frozen'])}</td>
            </tr>"""

        # Per-class metrics
        cls_rows = ""
        for cls in classes:
            f1v = ev.get("per_class_f1",  {}).get(cls, None)
            pv  = ev.get("per_class_precision", {}).get(cls, None)
            rv  = ev.get("per_class_recall",    {}).get(cls, None)
            color = ""
            if f1v is not None:
                color = ("style='background:#e8f5e9'" if f1v >= 0.5 else
                         "style='background:#fff9c4'" if f1v >= 0.25 else
                         "style='background:#ffebee'")
            cls_rows += f"""<tr {color}>
              <td>{cls}</td>
              <td>{_fmt(f1v)}</td>
              <td>{_fmt(pv)}</td>
              <td>{_fmt(rv)}</td>
            </tr>"""

        best_cls  = ev.get("best_class_f1",  {})
        worst_cls = ev.get("worst_class_f1", {})

        model_sections += f"""
        <div class="model-card" id="{name}">
          <div class="model-header">
            <span class="model-name">{name}</span>
            <span class="badge">{mc.get('type','—')}</span>
            <span class="badge">{mc.get('backbone','—')}</span>
            {'<span class="badge badge-green">Checkpoint ✓</span>' if s['checkpoint'].get('exists') else '<span class="badge badge-red">No checkpoint</span>'}
          </div>

          <div class="stats-grid">
            <div class="stat"><div class="val">{_param_fmt(p['total'])}</div><div class="lbl">Total params</div></div>
            <div class="stat"><div class="val">{_param_fmt(p['trainable'])}</div><div class="lbl">Trainable</div></div>
            <div class="stat"><div class="val">{p['memory_fp32_mb']} MB</div><div class="lbl">FP32 memory</div></div>
            <div class="stat"><div class="val">{tr.get('epochs_trained','—')}</div><div class="lbl">Epochs trained</div></div>
            <div class="stat"><div class="val">{tr.get('total_training_time','—')}</div><div class="lbl">Training time</div></div>
            <div class="stat"><div class="val">{tr.get('avg_epoch_time','—')}</div><div class="lbl">Avg epoch time</div></div>
            <div class="stat highlight"><div class="val">{_fmt(ev.get('accuracy'))}</div><div class="lbl">Test Accuracy</div></div>
            <div class="stat highlight"><div class="val">{_fmt(ev.get('f1_macro'))}</div><div class="lbl">F1 Macro</div></div>
            <div class="stat"><div class="val">{_fmt(ev.get('roc_auc'))}</div><div class="lbl">ROC-AUC</div></div>
            <div class="stat"><div class="val">{tr.get('best_epoch','—')}</div><div class="lbl">Best epoch</div></div>
          </div>

          <div class="two-col">
            <div>
              <h4>Model Configuration</h4>
              <table class="cfg-table">
                <tr><td>Backbone</td><td>{mc.get('backbone','—')}</td></tr>
                <tr><td>Image size</td><td>{mc.get('img_size','—')}×{mc.get('img_size','—')}</td></tr>
                <tr><td>Normalization</td><td>{mc.get('normalization','—')}</td></tr>
                <tr><td>Dropout</td><td>{mc.get('dropout','—')}</td></tr>
                <tr><td>Input shape</td><td>{mc.get('input_shape','—')}</td></tr>
                <tr><td>Pretrained</td><td>{'Yes (ImageNet/Kinetics)' if mc.get('pretrained') else 'No'}</td></tr>
                <tr><td>Frame level</td><td>{'Yes — ' + str(tc.get('fl_frames','all')) + ' frames/video' if tc.get('frame_level') else 'No (video-level)'}</td></tr>
                <tr><td>Freeze epochs</td><td>{tc.get('freeze_epochs', 0)}</td></tr>
              </table>
            </div>
            <div>
              <h4>Training Configuration</h4>
              <table class="cfg-table">
                <tr><td>Learning rate</td><td>{tc.get('learning_rate')}</td></tr>
                <tr><td>Backbone LR scale</td><td>{tc.get('backbone_lr_scale')}</td></tr>
                <tr><td>Weight decay</td><td>{tc.get('weight_decay')}</td></tr>
                <tr><td>Batch size</td><td>{tc.get('batch_size')}</td></tr>
                <tr><td>Max epochs</td><td>{tc.get('epochs_max')}</td></tr>
                <tr><td>Early stopping</td><td>{tc.get('early_stopping')} epochs</td></tr>
                <tr><td>Scheduler</td><td>{tc.get('scheduler')}</td></tr>
                <tr><td>Mixup alpha</td><td>{tc.get('mixup_alpha')}</td></tr>
                <tr><td>Mixed precision</td><td>{'Yes (AMP)' if tc.get('mixed_precision') else 'No'}</td></tr>
                <tr><td>TTA (test flip)</td><td>{'Yes' if tc.get('tta_flips') else 'No'}</td></tr>
              </table>
            </div>
          </div>

          {'<div><p><b>Best class (F1):</b> ' + str(best_cls.get('class','—')) + ' (' + str(best_cls.get('score',0)) + ')  &nbsp; <b>Worst class (F1):</b> ' + str(worst_cls.get('class','—')) + ' (' + str(worst_cls.get('score',0)) + ')</p></div>' if ev else ''}

          {'<div><h4>Per-Class Metrics</h4><table class="cfg-table"><tr><th>Class</th><th>F1</th><th>Precision</th><th>Recall</th></tr>' + cls_rows + '</table></div>' if cls_rows else ''}

          <div>
            <h4>Parameter Groups</h4>
            <table class="cfg-table">
              <tr><th>Group</th><th>Total</th><th>Trainable</th><th>Frozen</th></tr>
              {group_rows}
            </table>
          </div>

          <details>
            <summary style="cursor:pointer;color:#3f51b5;font-weight:bold">Architecture (click to expand)</summary>
            <pre style="font-size:0.75em;background:#f5f5f5;padding:16px;border-radius:6px;overflow-x:auto">{s.get('architecture','')}</pre>
          </details>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Model Summary Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 40px;
         background: #f0f2f5; color: #2c3e50; }}
  h1 {{ color: #1a237e; font-size: 1.9em; margin-bottom: 4px; }}
  h2 {{ color: #283593; border-left: 5px solid #3f51b5; padding-left: 12px; margin-top: 36px; }}
  h3 {{ color: #37474f; margin: 16px 0 8px; }}
  h4 {{ color: #455a64; margin: 12px 0 6px; font-size: 0.95em; }}
  .toc a {{ display: inline-block; margin: 4px 8px; color: #3f51b5; text-decoration: none; font-weight: bold; }}
  .card {{ background: white; border-radius: 10px; padding: 24px;
           box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 20px 0; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #3f51b5; color: white; padding: 9px 12px; text-align: center; font-size: 0.88em; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #eceff1; font-size: 0.88em; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f3f4f6; }}
  .model-card {{ background: white; border-radius: 10px; padding: 24px;
                 box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 20px 0; }}
  .model-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
  .model-name {{ font-size: 1.4em; font-weight: bold; color: #1a237e; }}
  .badge {{ background: #e8eaf6; color: #283593; padding: 3px 12px;
            border-radius: 20px; font-size: 0.82em; font-weight: bold; }}
  .badge-green {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-red {{ background: #ffebee; color: #c62828; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
  .stat {{ background: #f8f9fa; border-radius: 8px; padding: 12px; text-align: center; }}
  .stat .val {{ font-size: 1.4em; font-weight: bold; color: #3f51b5; }}
  .stat.highlight .val {{ color: #2e7d32; font-size: 1.6em; }}
  .stat .lbl {{ font-size: 0.78em; color: #607d8b; margin-top: 2px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 12px 0; }}
  .cfg-table td:first-child {{ color: #607d8b; font-weight: bold; width: 45%; }}
  .cfg-table th {{ background: #eceff1; color: #37474f; }}
  pre {{ margin: 0; }}
  details {{ margin-top: 12px; }}
</style>
</head><body>

<h1>Model Summary Report</h1>
<p style="color:#607d8b">UCF-Crime Dataset | {len(classes)} Classes | Evidence Timeline Reconstruction System</p>

<div class="toc">
  <b>Jump to:</b>
  <a href="#overview">Overview</a>
  <a href="#bestclass">Best Model Per Class</a>
  {''.join(f'<a href="#{n}">{n}</a>' for n in names)}
</div>

<!-- ── Overview ── -->
<h2 id="overview">Comparative Overview</h2>
<div class="card">
<table>
  <tr>
    <th>Model</th><th>Backbone</th><th>Type</th>
    <th>Total Params</th><th>Trainable</th><th>Trainable %</th>
    <th>Memory (FP32)</th><th>Epochs</th><th>Train Time</th><th>Avg/Epoch</th>
    <th>Best Val F1</th><th>Test Accuracy</th><th>Test F1</th><th>Checkpoint</th>
  </tr>
  {comp_rows}
</table>
</div>

<!-- ── Best model per class ── -->
<h2 id="bestclass">Best Model Per Crime Class</h2>
<div class="card">
<table>
  <tr>
    <th>Crime Class</th>
    <th>Best F1 (model)</th>
    <th>Best Precision (model)</th>
    <th>Best Recall (model)</th>
  </tr>
  {class_rows}
</table>
</div>

<!-- ── Per-model detail ── -->
<h2>Per-Model Detail</h2>
{model_sections}

</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"HTML summary -> {out_path}")


# ── Main ─────────────────────────────────────────────────────
def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="configs/config.yaml")
        parser.add_argument("--model",  default=None)
        args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Use grouped class names for display if any model is grouped
    _any_grouped = any(
        mc.get("grouped") and cfg["dataset"].get("grouped_classes")
        for mc in (cfg["models"] or []) if mc.get("enabled", True))
    if _any_grouped:
        classes = [g["name"] for g in cfg["dataset"]["grouped_classes"]]
    else:
        classes = cfg["dataset"]["classes"]
    weights_dir = Path(cfg["paths"]["weights_dir"])
    logs_dir    = Path(cfg["paths"]["logs_dir"])
    metrics_dir = Path(cfg["evaluation"]["results_dir"]) / "metrics"
    out_dir     = Path(cfg["evaluation"]["results_dir"]) / "model_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = [m for m in cfg["models"] if m.get("enabled", True)]
    if args.model:
        models_cfg = [m for m in models_cfg if m["name"] == args.model]

    all_summaries = {}

    for mc in models_cfg:
        name = mc["name"]
        try:
            s = build_model_summary(mc, cfg, logs_dir, metrics_dir,
                                    weights_dir, classes)
            all_summaries[name] = s

            # Per-model JSON (without layer_table — too verbose for master)
            per_model = {k: v for k, v in s.items() if k != "layer_table"}
            with open(out_dir / f"{name}_summary.json", "w") as f:
                json.dump(per_model, f, indent=2)
        except Exception as e:
            log.error(f"  Failed {name}: {e}")

    # Master JSON
    master = {}
    for name, s in all_summaries.items():
        master[name] = {k: v for k, v in s.items()
                        if k not in ("architecture", "layer_table")}
    with open(out_dir / "model_summary.json", "w") as f:
        json.dump(master, f, indent=2)
    log.info(f"Master JSON -> {out_dir / 'model_summary.json'}")

    # HTML
    generate_html(all_summaries, classes, out_dir / "model_summary.html")

    # Console summary
    log.info("\n" + "="*70)
    log.info(f"{'Model':<18} {'Params':>10} {'Trainable':>10} {'Epochs':>7} "
             f"{'ValF1':>7} {'TestAcc':>8} {'TestF1':>8}")
    log.info("="*70)
    for name, s in all_summaries.items():
        p  = s["parameters"]
        tr = s["training"]
        ev = s["evaluation"]
        vf1 = tr.get("best_val_f1", tr.get("best_val_acc", "—"))
        log.info(
            f"{name:<18} {_param_fmt(p['total']):>10} "
            f"{_param_fmt(p['trainable']):>10} "
            f"{str(tr.get('epochs_trained','—')):>7} "
            f"{str(vf1 or '—'):>7} "
            f"{str(ev.get('accuracy','—') or '—'):>8} "
            f"{str(ev.get('f1_macro','—') or '—'):>8}"
        )
    log.info("="*70)
    log.info(f"\nOutputs in: {out_dir}/")


if __name__ == "__main__":
    main()
