"""
STEP 2 — Training
==================
Three training modes (auto-detected from config + model type):

  VIDEO-LEVEL  (default, CNN classifiers + 3D video models):
    Input: (B, N, 3, H, W) — B videos, N frames each
    One prediction per video.
    Supports Mixup augmentation (config: mixup_alpha > 0).

  SEQUENCE     (CNNLSTM, CNNTransformer):
    Input: (B, T, 3, H, W) — sliding window sequences
    Per-frame binary + video class joint training.

Usage:
  python src/training/train.py                   # train ALL models
  python src/training/train.py --model ResNet50
  python src/training/train.py --model R2Plus1D
  python src/training/train.py --model CNNLSTM
  python src/training/train.py --resume
"""

import os
import sys
import json
import yaml
import argparse
import logging
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import get_dataloaders, get_sequence_dataloaders
from src.models.model_builder import build_model, count_parameters, is_temporal_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# Mixup augmentation
# ════════════════════════════════════════════════════════════
def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 0.4, device: str = "cuda"):
    """
    Mixup: interpolate between two random training samples.

    Returns mixed inputs, paired labels, and mixing coefficient.
    Prevents the model from memorizing individual training videos by
    training it on convex combinations of examples.

    Reference: Zhang et al., 'mixup: Beyond Empirical Risk Minimization', ICLR 2018
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """Compute Mixup loss as interpolation of two CE losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fix_bn_running_stats(model: nn.Module) -> int:
    """
    Reset BatchNorm running_mean/running_var if they contain NaN.

    Root cause: with FP16 autocast, a large batch (e.g. 256 images) through a
    deep CNN can overflow (value > 65504), producing inf.  inf appears in the
    batch statistics used to update running_mean/running_var via moving average:
        running_mean = 0.9 * running_mean + 0.1 * NaN_batch_mean  →  NaN
    Once NaN the stats stay NaN forever.  model.train() uses live batch stats
    (so training loss looks fine) while model.eval() uses the corrupted running
    stats → NaN logits → NaN val loss.

    Fix: reset to (mean=0, var=1) — BN will re-accumulate correct stats over
    the next few training batches via momentum.
    """
    fixed = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if (torch.isnan(m.running_mean).any() or
                    torch.isnan(m.running_var).any()):
                m.reset_running_stats()
                fixed += 1
    if fixed:
        log.warning(f"  Fixed NaN running stats in {fixed} BN layers "
                    f"(FP16 overflow in early training)")
    return fixed


# ════════════════════════════════════════════════════════════
# Scheduler
# ════════════════════════════════════════════════════════════
def get_scheduler(optimizer, cfg: dict, total_steps: int):
    stype  = cfg["training"]["scheduler"]
    warmup = int(total_steps * 0.05)
    if stype == "cosine":
        base = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup), eta_min=1e-7)
    elif stype == "step":
        base = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=4, factor=0.5)
    wu = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return optim.lr_scheduler.SequentialLR(optimizer, [wu, base], [warmup])


# ════════════════════════════════════════════════════════════
# VIDEO-LEVEL TRAINING (primary mode)
# ════════════════════════════════════════════════════════════
def train_epoch_video(model, loader, optimizer, criterion,
                      scaler, device, scheduler,
                      mixup_alpha: float = 0.0):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Frame-level mode returns (B,3,H,W); model expects (B,N,3,H,W)
        if videos.dim() == 4:
            videos = videos.unsqueeze(1)

        # Apply Mixup if enabled
        use_mixup = mixup_alpha > 0 and model.training
        if use_mixup:
            videos, y_a, y_b, lam = mixup_data(videos, labels, mixup_alpha, device)

        with autocast("cuda", enabled=scaler is not None):
            out = model(videos)
            # CNNLSTM/CNNTransformer return (frame_logits, class_logits) tuple
            logits = out[1] if isinstance(out, (tuple, list)) else out
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

        if scheduler and not isinstance(scheduler,
                                        optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        # For Mixup, use primary label for acc tracking (approximate)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def validate_video(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for videos, labels in tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if videos.dim() == 4:
            videos = videos.unsqueeze(1)
        with autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(videos)
            logits = out[1] if isinstance(out, (tuple, list)) else out
        # Compute loss in FP32: CrossEntropyLoss(weight=..., label_smoothing=...)
        # can produce NaN in FP16 autocast due to small weights * large log-probs.
        loss = criterion(logits.float(), labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


# ════════════════════════════════════════════════════════════
# SEQUENCE TRAINING (CNN-LSTM / CNN-Transformer)
# ════════════════════════════════════════════════════════════
def train_epoch_sequence(model, loader, optimizer,
                          frame_crit, class_crit,
                          scaler, device, scheduler,
                          frame_w: float = 0.6):
    model.train()
    total_loss = 0.0
    all_fp, all_fl, all_cp, all_cl = [], [], [], []

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for seqs, frame_labels, class_labels in pbar:
        seqs         = seqs.to(device, non_blocking=True)
        frame_labels = frame_labels.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=scaler is not None):
            frame_logits, class_logits = model(seqs)
            B, T = frame_labels.shape
            fl   = frame_logits.view(B * T, 2)
            flbl = frame_labels.view(B * T)
            loss = (frame_w * frame_crit(fl, flbl) +
                    (1 - frame_w) * class_crit(class_logits, class_labels))

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

        if scheduler and not isinstance(scheduler,
                                        optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()
        all_fp.extend(fl.argmax(1).cpu().numpy())
        all_fl.extend(flbl.cpu().numpy())
        all_cp.extend(class_logits.argmax(1).cpu().numpy())
        all_cl.extend(class_labels.cpu().numpy())

    return (total_loss / len(loader),
            accuracy_score(all_fl, all_fp),
            accuracy_score(all_cl, all_cp))


@torch.no_grad()
def validate_sequence(model, loader, frame_crit, class_crit,
                       device, frame_w: float = 0.6):
    model.eval()
    total_loss = 0.0
    all_fp, all_fl, all_cp, all_cl = [], [], [], []

    for seqs, frame_labels, class_labels in tqdm(loader, desc="val  ", leave=False, dynamic_ncols=True):
        seqs         = seqs.to(device, non_blocking=True)
        frame_labels = frame_labels.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)
        with autocast("cuda", enabled=torch.cuda.is_available()):
            frame_logits, class_logits = model(seqs)
        B, T = frame_labels.shape
        fl   = frame_logits.float().view(B * T, 2)
        flbl = frame_labels.view(B * T)
        loss = (frame_w * frame_crit(fl, flbl) +
                (1 - frame_w) * class_crit(class_logits.float(), class_labels))
        total_loss += loss.item()
        all_fp.extend(fl.argmax(1).cpu().numpy())
        all_fl.extend(flbl.cpu().numpy())
        all_cp.extend(class_logits.argmax(1).cpu().numpy())
        all_cl.extend(class_labels.cpu().numpy())

    ff1 = f1_score(all_fl, all_fp, average="macro", zero_division=0)
    cf1 = f1_score(all_cl, all_cp, average="macro", zero_division=0)
    return total_loss / len(loader), accuracy_score(all_fl, all_fp), ff1, cf1


# ════════════════════════════════════════════════════════════
# MAIN TRAIN FUNCTION
# ════════════════════════════════════════════════════════════
def train_model(model_cfg: dict, cfg: dict, device: str, resume: bool = False):
    model_name  = model_cfg["name"]
    # Binary models override num_classes to 2
    if model_cfg.get("grouped") and cfg["dataset"].get("grouped_classes"):
        grouped_cls = cfg["dataset"]["grouped_classes"]
        num_classes = len(grouped_cls)
        classes     = [g["name"] for g in grouped_cls]
        log.info(f"Grouped mode: {num_classes} groups — {classes}")
    else:
        num_classes = cfg["dataset"]["num_classes"]
        classes     = cfg["dataset"]["classes"]
    temporal    = is_temporal_model(model_cfg)
    # video_level:true → use VideoLevelDataset even for CNNLSTM/CNNTransformer.
    # Cuts batches/epoch from 6000+ → ~100; trains 60× faster.
    if model_cfg.get("video_level"):
        temporal = False

    weights_dir = Path(cfg["paths"]["weights_dir"])
    ckpt_dir    = Path(cfg["paths"]["checkpoints_dir"])
    logs_dir    = Path(cfg["paths"]["logs_dir"])
    for d in [weights_dir, ckpt_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    best_path = weights_dir / f"{model_name}_best.pt"
    ckpt_path = ckpt_dir    / f"{model_name}_checkpoint.pt"
    log_path  = logs_dir    / f"{model_name}_history.json"

    mode = "TEMPORAL" if temporal else "VIDEO-LEVEL"
    log.info(f"\n{'='*60}")
    log.info(f"Training: {model_name}  [{mode}]")
    log.info(f"{'='*60}")

    if temporal:
        train_dl, val_dl, _ = get_sequence_dataloaders(cfg, model_cfg)
    else:
        train_dl, val_dl, _ = get_dataloaders(cfg, model_cfg)

    log.info(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    model = build_model(model_cfg, num_classes).to(device)

    # ── Two-phase training ────────────────────────────────────
    # Phase 1 (epochs 0..freeze_epochs-1): backbone FROZEN — only head trains.
    #   Forces the model to learn good class boundaries from ImageNet features
    #   before any backbone weights change. Prevents early overfitting on
    #   tiny datasets (~55 videos/class).
    # Phase 2 (epoch freeze_epochs onward): backbone unfrozen with low LR.
    #   Fine-tunes from a well-initialised head → better generalisation.
    freeze_epochs = int(model_cfg.get("freeze_epochs", 0))
    if freeze_epochs > 0:
        for n, p in model.named_parameters():
            if not any(k in n for k in ("classifier", "temporal_pool", "head")):
                p.requires_grad = False
        log.info(f"Phase 1: backbone FROZEN for first {freeze_epochs} epochs")

    log.info(f"Parameters (trainable): {count_parameters(model):,}")

    # Loss — class weighted + label smoothing
    cw_path = Path(cfg["dataset"]["splits_dir"]) / "class_weights.json"
    if cfg["training"]["use_class_weights"] and model_cfg.get("grouped"):
        # Grouped mode: compute inverse-frequency weights from the training set.
        # class_weights.json has original 14-class names; group names won't match.
        ds = train_dl.dataset
        if hasattr(ds, "get_class_counts"):
            # VideoLevelDataset / UCFCrimeDataset — already group-remapped
            raw_counts = ds.get_class_counts(num_classes)
        elif hasattr(ds, "get_class_sample_counts"):
            # SequenceDataset — raw 14-class counts; remap to groups manually
            orig_counts = ds.get_class_sample_counts(cfg["dataset"]["num_classes"])
            from src.data.dataset import build_group_mapping
            _gm = build_group_mapping(cfg["dataset"]["grouped_classes"],
                                      cfg["dataset"]["classes"])
            raw_counts = np.zeros(num_classes, dtype=np.float32)
            for orig_idx, grp_idx in _gm.items():
                if orig_idx < len(orig_counts):
                    raw_counts[grp_idx] += orig_counts[orig_idx]
        else:
            raw_counts = np.ones(num_classes, dtype=np.float32)
        counts = np.array(raw_counts[:num_classes], dtype=np.float32)
        total  = counts.sum()
        cw_arr = (total / (num_classes * np.maximum(counts, 1))).astype(np.float32)
        cw_arr = np.clip(cw_arr / cw_arr.mean(), 0.2, 5.0)
        cw_tensor = torch.tensor(cw_arr, dtype=torch.float32).to(device)
        class_crit = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.1)
        log.info(f"Grouped class weights: "
                 + " ".join(f"{c}={w:.2f}" for c, w in zip(classes, cw_arr)))
    elif cfg["training"]["use_class_weights"] and cw_path.exists():
        with open(cw_path) as f:
            cw = json.load(f)
        cw_tensor = torch.tensor(
            [cw.get(c, 1.0) for c in classes], dtype=torch.float32).to(device)
        class_crit = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.1)
        log.info("Using class-weighted loss + label smoothing 0.1")
    else:
        class_crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    frame_crit = nn.CrossEntropyLoss(
        weight=torch.tensor([0.3, 0.7], device=device))

    lr          = float(cfg["training"]["learning_rate"])
    wd          = float(cfg["training"]["weight_decay"])
    # Per-model mixup_alpha override (3D models set 0.0 — mixing temporal sequences breaks motion)
    mixup_alpha = float(model_cfg.get("mixup_alpha",
                        cfg["training"].get("mixup_alpha", 0.0)))

    if temporal:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd)
    else:
        # Differential LR: backbone gets lower LR to preserve pretrained features.
        # backbone_lr_scale from config:
        #   0.1 (default) — 2D models (EfficientNet, ConvNeXt, Swin)
        #   0.3           — 3D video models (R2Plus1D, S3D): need more adaptation
        #                   because Kinetics→surveillance domain shift is larger
        backbone_lr_scale = float(model_cfg.get("backbone_lr_scale", 0.1))
        backbone_params = []
        head_params     = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in ("classifier", "temporal_pool", "head")):
                head_params.append(p)
            else:
                backbone_params.append(p)

        if backbone_params and backbone_lr_scale > 0:
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": lr * backbone_lr_scale},
                {"params": head_params,     "lr": lr},
            ], weight_decay=wd)
            log.info(f"Differential LR: backbone={lr * backbone_lr_scale:.2e}  "
                     f"head={lr:.2e}  (scale={backbone_lr_scale})")
        else:
            # Fully frozen backbone — only head params
            optimizer = optim.AdamW(head_params, lr=lr, weight_decay=wd)
            log.info(f"Head-only LR: {lr:.2e}  "
                     f"(backbone fully frozen, {sum(p.numel() for p in head_params):,} trainable params)")

    use_amp   = cfg["training"]["mixed_precision"] and device == "cuda"
    scaler    = GradScaler() if use_amp else None
    epochs    = cfg["training"]["epochs"]
    scheduler = get_scheduler(optimizer, cfg, epochs * len(train_dl))
    patience  = cfg["training"]["early_stopping_patience"]

    if temporal:
        history = {"train_loss":[], "train_frame_acc":[], "train_class_acc":[],
                   "val_loss":[], "val_frame_acc":[], "val_frame_f1":[], "val_class_f1":[],
                   "epoch_time_sec":[], "total_time_sec": 0.0}
    else:
        history = {"train_loss":[], "train_acc":[],
                   "val_loss":[], "val_acc":[], "val_f1":[],
                   "epoch_time_sec":[], "total_time_sec": 0.0}

    best_val_f1      = 0.0
    patience_counter = 0
    start_epoch      = 0

    if resume and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_f1 = ckpt.get("best_val_f1", 0.0)
            history     = ckpt.get("history", history)
            # Restore scheduler so LR continues correctly (not restart warmup from epoch 0)
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            # Restore AMP scaler (keeps loss scaling history intact)
            if scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            log.info(f"Resumed from epoch {start_epoch}, best_f1={best_val_f1:.4f}")
        except Exception as e:
            log.warning(f"Cannot resume checkpoint (architecture mismatch?): {e}")
            log.warning("Starting fresh training.")

    t0 = time.time()

    epoch_bar = tqdm(range(start_epoch, epochs), desc=model_name, unit="epoch",
                     dynamic_ncols=True)
    for epoch in epoch_bar:

        # ── Phase 2 transition: unfreeze backbone ─────────────
        if freeze_epochs > 0 and epoch == freeze_epochs:
            backbone_lr_scale = float(model_cfg.get("backbone_lr_scale", 0.1))
            backbone_p = []
            for n, p in model.named_parameters():
                p.requires_grad = True
                if not any(k in n for k in ("classifier", "temporal_pool", "head")):
                    backbone_p.append(p)
            # Add backbone as a NEW param group — preserves head's Adam state
            # (momentum/variance from phase 1) rather than resetting everything.
            optimizer.add_param_group({"params": backbone_p,
                                       "lr": lr * backbone_lr_scale})
            # Update head group LR in case scheduler decayed it during phase 1
            optimizer.param_groups[0]["lr"] = lr
            log.info(f"Phase 2: backbone UNFROZEN at epoch {epoch+1} "
                     f"(backbone_lr={lr*backbone_lr_scale:.2e}, head_lr={lr:.2e})")

        epoch_start = time.time()

        if temporal:
            tr_loss, tr_facc, tr_cacc = train_epoch_sequence(
                model, train_dl, optimizer, frame_crit, class_crit,
                scaler, device, scheduler)
            vl_loss, vl_facc, vl_ff1, vl_cf1 = validate_sequence(
                model, val_dl, frame_crit, class_crit, device)
            val_f1 = 0.6 * vl_ff1 + 0.4 * vl_cf1

            history["train_loss"].append(tr_loss)
            history["train_frame_acc"].append(tr_facc)
            history["train_class_acc"].append(tr_cacc)
            history["val_loss"].append(vl_loss)
            history["val_frame_acc"].append(vl_facc)
            history["val_frame_f1"].append(vl_ff1)
            history["val_class_f1"].append(vl_cf1)

            epoch_bar.set_postfix(TrLoss=f"{tr_loss:.4f}", VlClsF1=f"{vl_cf1:.3f}")
            tqdm.write(
                f"\n{'─'*80}\n"
                f"[{model_name}] Epoch {epoch+1:03d}/{epochs}  ({(time.time()-t0)/60:.1f}min)\n"
                f"  TRAIN  loss={tr_loss:.4f}  frame_acc={tr_facc:.4f}  cls_acc={tr_cacc:.4f}\n"
                f"  VAL    loss={vl_loss:.4f}  frame_f1={vl_ff1:.4f}   cls_f1={vl_cf1:.4f}\n"
                f"  BEST   F1={max(history['val_class_f1']):.4f}\n"
                f"{'─'*80}"
            )
        else:
            tr_loss, tr_acc = train_epoch_video(
                model, train_dl, optimizer, class_crit, scaler, device, scheduler,
                mixup_alpha=mixup_alpha)
            fix_bn_running_stats(model)   # guard against FP16 overflow corrupting BN stats
            vl_loss, vl_acc, vl_f1 = validate_video(
                model, val_dl, class_crit, device)
            val_f1 = vl_f1

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
            history["val_f1"].append(vl_f1)

            epoch_bar.set_postfix(TrLoss=f"{tr_loss:.4f}", VlF1=f"{vl_f1:.4f}")
            tqdm.write(
                f"\n{'─'*80}\n"
                f"[{model_name}] Epoch {epoch+1:03d}/{epochs}  ({(time.time()-t0)/60:.1f}min)\n"
                f"  TRAIN  loss={tr_loss:.4f}  acc={tr_acc:.4f}\n"
                f"  VAL    loss={vl_loss:.4f}  acc={vl_acc:.4f}  f1={vl_f1:.4f}\n"
                f"  BEST   F1={max(history['val_f1']):.4f}\n"
                f"{'─'*80}"
            )

        epoch_sec = time.time() - epoch_start
        history["epoch_time_sec"].append(round(epoch_sec, 1))
        history["total_time_sec"] = round(time.time() - t0, 1)

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            tmp_best = best_path.with_suffix(".tmp")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_f1":           val_f1,
                "model_cfg":        model_cfg,
                "temporal":         temporal,
            }, tmp_best)
            os.replace(tmp_best, best_path)   # atomic — safe against power cut
            log.info(f"  ✓ Saved best (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

        # ── Atomic checkpoint save ────────────────────────────
        # Write to a .tmp file first, then os.replace() — which is atomic
        # on the same filesystem. Prevents a corrupted checkpoint if power
        # cuts mid-write (a plain torch.save() leaves a partial file).
        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
                                    if hasattr(scheduler, "state_dict") else None,
            "scaler_state_dict":    scaler.state_dict() if scaler else None,
            "best_val_f1":          best_val_f1,
            "history":              history,
            "model_cfg":            model_cfg,
        }
        tmp_ckpt = ckpt_path.with_suffix(".tmp")
        torch.save(ckpt_data, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_path)   # atomic rename

        # ── Atomic log save ───────────────────────────────────
        tmp_log = log_path.with_suffix(".tmp")
        with open(tmp_log, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_log, log_path)     # atomic rename

    log.info(f"Done. Best F1={best_val_f1:.4f} → {best_path}")
    return history, best_val_f1


# ── Main ─────────────────────────────────────────────────────
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    # GPU optimizations — free speedup, no accuracy cost
    torch.backends.cudnn.benchmark    = True   # auto-tune kernels for fixed input shapes
    torch.backends.cuda.matmul.allow_tf32 = True  # tensor cores on Ampere/Ada (RTX 30/40)
    torch.backends.cudnn.allow_tf32   = True

    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    models_cfg = cfg["models"]
    # Skip models with enabled: false
    models_cfg = [m for m in models_cfg if m.get("enabled", True)]

    if args.model:
        models_cfg = [m for m in models_cfg if m["name"] == args.model]
        if not models_cfg:
            log.error(f"Model '{args.model}' not found or disabled in config.")
            log.error(f"Available: {[m['name'] for m in cfg['models'] if m.get('enabled', True)]}")
            return

    summary = {}
    for mc in models_cfg:
        try:
            hist, best_f1 = train_model(mc, cfg, device, resume=args.resume)
            summary[mc["name"]] = {"best_val_f1": best_f1}
        except Exception as e:
            log.error(f"FAILED: {mc['name']} — {e}")
            import traceback; traceback.print_exc()
            summary[mc["name"]] = {"best_val_f1": 0.0, "error": str(e)}

    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n" + "="*60)
    log.info("TRAINING SUMMARY:")
    for name, s in summary.items():
        log.info(f"  {name:25s}: best_f1={s['best_val_f1']:.4f}")
    log.info("Run next: python src/evaluation/evaluate.py")
    log.info("Then:      python src/evaluation/timeline_reconstruct.py")
    log.info("="*60)

    # Auto-generate model summary after training
    try:
        from src.evaluation.model_summary import main as make_summary
        import types
        dummy = types.SimpleNamespace(config=args.config, model=None)
        make_summary(dummy)
    except Exception as e:
        log.warning(f"Model summary skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model",  default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
