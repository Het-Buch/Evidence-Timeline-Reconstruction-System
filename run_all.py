"""
Full Pipeline — Evidence Timeline Reconstruction
=================================================
Runs the complete post-training pipeline:

  Step 1  Train remaining models (CNNLSTM if checkpoint absent)
  Step 2  Evaluate all trained models → confusion matrices, ROC, F1, etc.
  Step 3  XAI on best model → GradCAM + LIME overlays
  Step 4  Timeline reconstruction on all test videos (best model)
  Step 5  Demo on a randomly picked crime video (slow-motion mode)

Usage:
  python run_all.py                         # full pipeline
  python run_all.py --skip-train            # skip training, run eval+xai+demo
  python run_all.py --video data/raw/Robbery/Robbery001_x264.mp4
"""

import sys
import os
import json
import yaml
import random
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def run(cmd: str, check: bool = True):
    log.info(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(HERE))
    if check and result.returncode != 0:
        log.error(f"Command failed (code {result.returncode}): {cmd}")
        return False
    return True


def pick_random_video(raw_dir: Path, prefer_classes=("Robbery", "Shooting", "Assault")) -> Path:
    """Pick a random video from raw_dir, preferring crime classes."""
    for cls in prefer_classes:
        cls_dir = raw_dir / cls
        if cls_dir.exists():
            vids = sorted(cls_dir.glob("*.mp4")) + sorted(cls_dir.glob("*.avi"))
            if vids:
                return random.choice(vids)
    # Fallback: any video in any class
    for cls_dir in sorted(raw_dir.iterdir()):
        if cls_dir.is_dir():
            vids = sorted(cls_dir.glob("*.mp4")) + sorted(cls_dir.glob("*.avi"))
            if vids:
                return vids[0]
    return None


def _build_master_dashboard(cfg: dict, results: Path):
    """Generate a master index.html that links all reports."""
    from datetime import datetime

    metrics_path = results / "metrics" / "all_models_metrics.json"
    best_model = "R2Plus1D"
    leaderboard_rows = ""
    if metrics_path.exists():
        with open(metrics_path) as f:
            mm = json.load(f)
        if mm:
            best_model = max(mm, key=lambda k: mm[k].get("f1_macro", 0))
            ranked = sorted(mm.items(), key=lambda x: x[1].get("f1_macro", 0), reverse=True)
            for rank, (name, m) in enumerate(ranked, 1):
                badge = "style='background:#c8e6c9;font-weight:bold'" if rank == 1 else ""
                leaderboard_rows += f"""
                <tr {badge}>
                  <td>#{rank}</td><td><b>{name}</b></td>
                  <td>{m.get('accuracy',0):.4f}</td>
                  <td>{m.get('f1_macro',0):.4f}</td>
                  <td>{m.get('roc_auc') and f"{m['roc_auc']:.4f}" or 'N/A'}</td>
                </tr>"""

    # Collect XAI images
    xai_dir = Path(cfg["xai"]["output_dir"]) / best_model
    gradcam_imgs = sorted(xai_dir.glob("gradcam/*.png"))[:6] if xai_dir.exists() else []
    lime_imgs    = sorted(xai_dir.glob("lime/*.png"))[:4]    if xai_dir.exists() else []

    def img_tag(p: Path, results_root: Path) -> str:
        try:
            rel = p.relative_to(results_root)
            return f'<img src="{rel.as_posix()}" style="max-width:100%;border-radius:8px;margin:4px;" />'
        except ValueError:
            return ""

    gradcam_html = "\n".join(
        f'<div style="display:inline-block;width:32%;vertical-align:top;padding:4px">{img_tag(p, results)}</div>'
        for p in gradcam_imgs)
    lime_html = "\n".join(
        f'<div style="display:inline-block;width:24%;vertical-align:top;padding:4px">{img_tag(p, results)}</div>'
        for p in lime_imgs)

    # Demo results
    demo_dir = results / "demo"
    demo_htmls = sorted(demo_dir.glob("**/*_report.html")) if demo_dir.exists() else []
    demo_links = "".join(
        f'<li><a href="{p.relative_to(results).as_posix()}" target="_blank">{p.stem}</a></li>'
        for p in demo_htmls)

    # Timeline results
    tl_html = results / "timeline" / best_model / f"{best_model}_timeline_report.html"
    tl_link = (f'<a href="{tl_html.relative_to(results).as_posix()}" target="_blank">'
               f'Timeline Report — {best_model}</a>'
               if tl_html.exists() else "Not generated yet")

    # Comparative plots
    comp_plots = [
        "plots/comparative_bar_chart.png",
        "plots/comparative_radar_chart.png",
        "plots/comparative_per_class_heatmap.png",
        "plots/comparative_leaderboard.png",
        "plots/comparative_convergence.png",
    ]
    comp_html = "\n".join(
        f'<div style="margin:10px 0"><img src="{p}" style="max-width:100%;border-radius:8px" /></div>'
        for p in comp_plots if (results / p).exists())

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8" />
<title>Evidence Timeline Reconstruction — Master Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0;
         background: #e8eaf6; color: #1a237e; }}
  header {{ background: linear-gradient(135deg,#1a237e,#283593);
            color: white; padding: 32px 48px; }}
  header h1 {{ margin: 0; font-size: 2em; }}
  header p {{ margin: 6px 0 0; opacity: 0.8; }}
  nav {{ background: #3f51b5; padding: 12px 48px; display: flex; gap: 18px; flex-wrap: wrap; }}
  nav a {{ color: white; text-decoration: none; font-weight: bold; font-size: 0.92em;
           padding: 6px 14px; border-radius: 20px; background: rgba(255,255,255,0.15); }}
  nav a:hover {{ background: rgba(255,255,255,0.3); }}
  main {{ padding: 32px 48px; max-width: 1400px; margin: auto; }}
  section {{ background: white; border-radius: 12px; padding: 28px;
             margin: 24px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.07); }}
  h2 {{ color: #283593; border-left: 5px solid #3f51b5; padding-left: 14px; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #3f51b5; color: white; padding: 10px 16px; }}
  td {{ padding: 9px 16px; border-bottom: 1px solid #e8eaf6; text-align: center; }}
  tr:hover td {{ background: #e8eaf6; }}
  .links a {{ display: inline-block; margin: 6px 8px; padding: 10px 20px;
              border-radius: 25px; background: #3f51b5; color: white;
              text-decoration: none; font-weight: bold; font-size: 0.9em; }}
  .links a:hover {{ background: #1a237e; }}
  .badge {{ background: #e8f5e9; color: #2e7d32; padding: 4px 12px;
            border-radius: 12px; font-size: 0.85em; font-weight: bold; }}
</style>
</head><body>
<header>
  <h1>Evidence Timeline Reconstruction System</h1>
  <p>UCF-Crime Dataset | 5-Group Classification | {datetime.now().strftime("%Y-%m-%d %H:%M")} |
     Best Model: <b>{best_model}</b></p>
</header>
<nav>
  <a href="#leaderboard">Leaderboard</a>
  <a href="#comparative">Comparative Analysis</a>
  <a href="#xai">XAI Explanations</a>
  <a href="#timeline">Timeline</a>
  <a href="#demo">Demo</a>
  <a href="evaluation_report.html" target="_blank">Full Eval Report</a>
  <a href="model_summary/model_summary.html" target="_blank">Model Summary</a>
  <a href="EDA_Visualization.html" target="_blank">EDA</a>
</nav>
<main>

<section id="leaderboard">
  <h2>Model Leaderboard</h2>
  <table>
    <tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>F1 Macro</th><th>ROC-AUC</th></tr>
    {leaderboard_rows or "<tr><td colspan='5'>Run evaluate.py first</td></tr>"}
  </table>
  <div class="links" style="margin-top:16px">
    <a href="evaluation_report.html" target="_blank">Full Evaluation Report</a>
    <a href="model_summary/model_summary.html" target="_blank">Model Summary</a>
  </div>
</section>

<section id="comparative">
  <h2>Comparative Analysis</h2>
  {comp_html or "<p>Run evaluate.py to generate comparative plots.</p>"}
</section>

<section id="xai">
  <h2>XAI — {best_model} (GradCAM + LIME)</h2>
  <h3>GradCAM Overlays</h3>
  <div>{gradcam_html or "<p>Run xai.py to generate explanations.</p>"}</div>
  <h3>LIME Importance Maps</h3>
  <div>{lime_html or "<p>Run xai.py to generate LIME explanations.</p>"}</div>
</section>

<section id="timeline">
  <h2>Timeline Reconstruction</h2>
  <p>{tl_link}</p>
</section>

<section id="demo">
  <h2>Demo Results</h2>
  <ul class="links">
    {demo_links or "<li>Run demo.py to generate demo results.</li>"}
  </ul>
</section>

</main>
</body></html>"""

    out = results / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Master dashboard → {out}")


def main():
    parser = argparse.ArgumentParser(description="Full Evidence Timeline Pipeline")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip model training (use existing checkpoints)")
    parser.add_argument("--video",       default=None,
                        help="Specific video for demo (default: random crime video)")
    parser.add_argument("--threshold",   type=float, default=0.45)
    parser.add_argument("--xai-samples", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights_dir = Path(cfg["paths"]["weights_dir"])

    # ════════════════════════════════════════════════════════════
    # STEP 1 — Train remaining models
    # ════════════════════════════════════════════════════════════
    if not args.skip_train:
        enabled = [m for m in cfg["models"] if m.get("enabled", True)]
        for mc in enabled:
            name = mc["name"]
            ckpt = weights_dir / f"{name}_best.pt"
            if ckpt.exists():
                log.info(f"[SKIP TRAIN] {name} — checkpoint exists ({ckpt})")
                continue
            log.info(f"\n{'='*60}")
            log.info(f"Training: {name}")
            log.info(f"{'='*60}")
            ok = run(f"python src/training/train.py --model {name} --config {args.config}")
            if not ok:
                log.warning(f"Training failed for {name}; continuing with next model.")
    else:
        log.info("Skipping training (--skip-train)")

    # ════════════════════════════════════════════════════════════
    # STEP 2 — Evaluate all trained models
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("STEP 2: Evaluation")
    log.info(f"{'='*60}")
    run(f"python src/evaluation/evaluate.py --config {args.config}", check=False)

    # ════════════════════════════════════════════════════════════
    # STEP 3 — XAI on best model
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("STEP 3: XAI (GradCAM + LIME)")
    log.info(f"{'='*60}")
    run(f"python src/evaluation/xai.py --config {args.config} --num-samples {args.xai_samples} --all-models",
        check=False)

    # ════════════════════════════════════════════════════════════
    # STEP 4 — Timeline reconstruction on test set (best model)
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("STEP 4: Timeline Reconstruction")
    log.info(f"{'='*60}")
    run(f"python src/evaluation/timeline_reconstruct.py --config {args.config}"
        f" --threshold {args.threshold}", check=False)

    # ════════════════════════════════════════════════════════════
    # STEP 5 — Demo on a random crime video (slow-motion mode)
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("STEP 5: Demo (random crime video)")
    log.info(f"{'='*60}")

    video_path = args.video
    if not video_path:
        raw_dir = Path(cfg["dataset"]["raw_dir"])
        vid = pick_random_video(raw_dir)
        if vid:
            video_path = str(vid)
            log.info(f"  Randomly selected: {video_path}")
        else:
            log.warning("No raw video found for demo. Skipping.")
            video_path = None

    if video_path and Path(video_path).exists():
        run(f'python demo.py --video "{video_path}" --config {args.config}'
            f" --threshold {args.threshold} --fps 3 --smooth 5",
            check=False)
    else:
        log.warning(f"Video not found: {video_path}")

    # ════════════════════════════════════════════════════════════
    # STEP 6 — EDA Notebook
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("STEP 6: EDA Notebook")
    log.info(f"{'='*60}")
    nb_path = HERE / "notebooks" / "EDA_Visualization.ipynb"
    nb_out  = HERE / "results" / "EDA_Visualization_executed.ipynb"
    nb_out.parent.mkdir(parents=True, exist_ok=True)
    if nb_path.exists():
        run(f'jupyter nbconvert --to notebook --execute "{nb_path}"'
            f' --output "{nb_out}" --ExecutePreprocessor.timeout=600',
            check=False)
        # Also export to HTML for easy viewing
        run(f'jupyter nbconvert --to html "{nb_out}"'
            f' --output "{HERE / "results" / "EDA_Visualization.html"}"',
            check=False)
        log.info(f"EDA notebook executed → results/EDA_Visualization.html")
    else:
        log.warning(f"EDA notebook not found at {nb_path}")

    # ════════════════════════════════════════════════════════════
    # STEP 7 — Model summary
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("STEP 7: Model Summary")
    log.info(f"{'='*60}")
    run(f"python src/evaluation/model_summary.py --config {args.config}", check=False)

    # ════════════════════════════════════════════════════════════
    # STEP 8 — Build master HTML dashboard (links all reports)
    # ════════════════════════════════════════════════════════════
    results = Path(cfg["evaluation"]["results_dir"])
    _build_master_dashboard(cfg, results)

    # ════════════════════════════════════════════════════════════
    # Print report locations
    # ════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}")
    log.info("ALL DONE — Open these in your browser:")
    log.info(f"{'='*60}")
    master = results / "index.html"
    if master.exists():
        log.info(f"  MASTER DASHBOARD : {master}")
    for p in [
        results / "evaluation_report.html",
        results / "model_summary" / "model_summary.html",
        results / "EDA_Visualization.html",
    ]:
        if p.exists():
            log.info(f"  {p}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
