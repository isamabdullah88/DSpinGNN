"""
visualizations/jvalidation.py
=====================================
Loads a trained DSpinGNN checkpoint, runs the test split through it,
collects predicted and ground-truth J (exchange coupling) values,
and saves a publication-quality correlation plot.

Usage (from repo root)
----------------------
python visualizations/jvalidatoin.py \
    --checkpoint checkpoints/latest-model.pt \
    --datasetpath DataSets/GNN/RattleGNN.pth \
    --modelname ExchangeMLP --gt_attr exchange \
    --split test --units meV --out visualizations/J_correlation.pdf

Arguments
---------
--checkpoint   Path to .pt checkpoint file (required)
--datasetpath  Path to the .pth dataset (same flag as in main.py)
--modelname    ExchangeMLP | StructureModel
--gt_attr      Batch attribute that holds ground-truth J (default: exchange)
--split        Which data split to evaluate: test | val | train (default: test)
--batch_size   Loader batch size (default: 32)
--units        Units label shown on axes (default: meV)
--out          Output file path (default: visualizations/J_correlation.pdf)
--mps          Use Apple MPS backend if available
--no_show      Skip plt.show() (useful on headless servers)
"""

import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

# ── Repo-root import fix ──────────────────────────────────────────────────────
# This file lives in  <root>/visualizations/  so we step one level up.
_VIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_VIS_DIR, ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from model import StructureGNN, ExchangeMLP          # noqa: E402
from data  import DatasetManager                     # noqa: E402
from train import load_checkpoint                    # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(modelname: str) -> torch.nn.Module:
    """Instantiate the correct model class from a modelname string."""
    if modelname == "StructureModel":
        return StructureGNN()
    if modelname in ("ExchangeMLP", "ExchangeModel"):
        return ExchangeMLP()
    raise ValueError(f"Unknown modelname '{modelname}'. Choose 'StructureModel' or 'ExchangeMLP'.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Forward pass helper
# ─────────────────────────────────────────────────────────────────────────────

def _forward(model, batch, modelname: str):
    """
    Run one forward pass and return pred_exchange regardless of how many
    values the model outputs.  Mirrors the logic in Trainer.validate_epoch
    but is robust to single-tensor vs tuple returns.
    """
    if modelname == "StructureModel":
        batch.pos.requires_grad_(True)
        energy   = model(batch)
        exchange = torch.zeros(batch.num_edges, device=batch.pos.device)
        return exchange

    # ExchangeMLP / ExchangeModel path
    out = model(batch)

    if isinstance(out, (tuple, list)):
        exchange = out[1]   # some variants return (energy, exchange)
    else:
        exchange = out       # pure exchange tensor

    return exchange


def _get_ground_truth(batch, gt_attr: str) -> torch.Tensor:
    """Return ground-truth J from a batch, trying a chain of attribute names."""
    candidates = [gt_attr, "exchange", "J", "j", "y", "exchange_coupling"]
    for attr in candidates:
        val = getattr(batch, attr, None)
        if val is not None and isinstance(val, torch.Tensor):
            return val
    raise AttributeError(f"Cannot find ground-truth exchange in batch. Tried: {candidates}. Pass the correct name via --gt_attr.")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Inference loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(model, loader, device, modelname, gt_attr):
    """
    Iterate over `loader`, run the forward pass, and accumulate
    (pred_J, true_J) as flat numpy arrays.
    """
    model.eval()
    all_pred, all_true = [], []

    # enable_grad because some models compute forces via autograd internally
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)

            pred_exchange = _forward(model, batch, modelname)
            true_exchange = batch.y_exchange.view(-1)

            pred_np = pred_exchange.detach().cpu().numpy().flatten()
            true_np = true_exchange.detach().cpu().numpy().flatten()

            # Guard against edge-count mismatches (e.g. padded batches)
            n = min(len(pred_np), len(true_np))
            all_pred.append(pred_np[:n])
            all_true.append(true_np[:n])

    pred_J = np.concatenate(all_pred)
    true_J = np.concatenate(all_true)
    print(f"[collect]  {len(pred_J):,} J pairs collected from {len(loader)} batches.")
    print(f"Range of true J: {true_J.min():.3f} to {true_J.max():.3f}  |  Range of pred J: {pred_J.min():.3f} to {pred_J.max():.3f}")
    return pred_J, true_J

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_J, true_J):
    r2               = r2_score(true_J, pred_J)
    mae              = mean_absolute_error(true_J, pred_J)
    rmse             = float(np.sqrt(np.mean((pred_J - true_J) ** 2)))
    pearson_r, pval  = stats.pearsonr(true_J, pred_J)
    slope, intercept, *_ = stats.linregress(true_J, pred_J)
    return dict(r2=r2, mae=mae, rmse=rmse, pearson_r=pearson_r, p_value=pval,
                slope=slope, intercept=intercept, n=len(pred_J))


def print_metrics(m, units):
    bar = "─" * 44
    print(f"\n{bar}")
    print("  DSpinGNN  ·  J Correlation Metrics")
    print(f"{bar}")
    print(f"  Pairs      : {m['n']:>12,}")
    print(f"  R²         : {m['r2']:>12.4f}")
    print(f"  Pearson r  : {m['pearson_r']:>12.4f}   p = {m['p_value']:.2e}")
    print(f"  MAE        : {m['mae']:>12.4f}  {units}")
    print(f"  RMSE       : {m['rmse']:>12.4f}  {units}")
    print(f"  Fit slope  : {m['slope']:>12.4f}")
    print(f"  Fit intcpt : {m['intercept']:>12.4f}  {units}")
    print(f"{bar}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plot
# ─────────────────────────────────────────────────────────────────────────────

# APS / PRB Styling Dictionary
_PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Matches Times serif styling for math
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",      # PRB requires inward pointing ticks
    "ytick.direction": "in",
    "xtick.top": True,            # Full bounding box ticks
    "ytick.right": True,
    "axes.spines.top": True,      # Full bounding box spines
    "axes.spines.right": True,
    "grid.linestyle": "--",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
    "legend.frameon": False,      # PRB legends typically float without a heavy box
    "legend.fontsize": 8,
}


def make_plot(pred_J, true_J, m, units, out_path, split_label, show):
    plt.rcParams.update(_PUB_RC)

    lo = min(true_J.min(), pred_J.min())
    hi = max(true_J.max(), pred_J.max())
    pad = (hi - lo) * 0.04
    lo -= pad; hi += pad

    # PRB Single Column Width is 3.375 inches
    fig, ax = plt.subplots(figsize=(3.375, 3.375), constrained_layout=True)

    # ── data ─────────────────────────────────────────────────────────────
    DENSE_THRESHOLD = 500

    if m["n"] >= DENSE_THRESHOLD:
        hb = ax.hexbin(true_J, pred_J, gridsize=50, cmap="Blues", mincnt=1, linewidths=0.1, zorder=3)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Count", fontsize=9)
        cb.ax.tick_params(labelsize=8, length=3, width=0.8, direction="in")
        cb.outline.set_linewidth(0.8)
    else:
        ax.scatter(true_J, pred_J, s=10, alpha=0.7, color="#1f77b4", edgecolors="none", zorder=3)

    # ── parity line ───────────────────────────────────────────────────────
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, zorder=5, label="$y = x$")

    # ── axes ─────────────────────────────────────────────────────────────
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"DFT $J$ ({units})")
    ax.set_ylabel(f"Predicted $J$ ({units})")
    ax.grid(True, zorder=0)
    ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.5)

    # ── metrics box ───────────────────────────────────────────────────────
    ann = (f"$R^2 = {m['r2']:.4f}$\n"
           f"$\\mathrm{{MAE}} = {m['mae']:.4f}$ {units}")
    ax.text(0.95, 0.05, ann, transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#AAAAAA", linewidth=0.5))

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # 600 DPI is the APS standard requirement for vector/line-art graphs
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"[plot]  Saved → {out_path}")

    if show:
        plt.show()
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── device ────────────────────────────────────────────────────────────
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[setup]  Device : {device}")

    # ── data ──────────────────────────────────────────────────────────────
    print(f"[data]   Loading dataset from {args.datasetpath} …")
    dataset_manager = DatasetManager()
    train_loader, test_loader = dataset_manager.dataloaders(args.datasetpath)
    split_map = {"train": train_loader, "test": test_loader}
    if args.split not in split_map:
        raise ValueError(f"--split must be one of {list(split_map)}.")
    loader = split_map[args.split]
    print(f"[data]   Split '{args.split}' : {len(loader.dataset):,} samples")

    # ── model ─────────────────────────────────────────────────────────────
    print(f"[model]  Building {args.modelname} …")
    model = build_model(args.modelname)

    print(f"[ckpt]   Loading weights from {args.checkpoint} …")
    model, _, start_epoch, _ = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model]  Resumed from epoch {start_epoch}  |  {n_params:,} parameters")

    # ── inference ─────────────────────────────────────────────────────────
    print("[infer]  Running forward pass …")
    pred_J, true_J = collect_predictions(model, loader, device, modelname=args.modelname, gt_attr=args.gt_attr)

    # ── metrics ───────────────────────────────────────────────────────────
    m = compute_metrics(pred_J, true_J)
    print_metrics(m, args.units)

    # ── plot ──────────────────────────────────────────────────────────────
    make_plot(pred_J, true_J, m=m, units=args.units, out_path=args.out,
              split_label=args.split, show=not args.no_show)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSpinGNN — J correlation plot",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the .pt checkpoint file")
    parser.add_argument("--datasetpath", default="./DataSets/GNN/Exchange-Test.pth", type=str, help="Path to the dataset .pth file")
    parser.add_argument("--modelname", default="ExchangeMLP", choices=["StructureModel", "ExchangeMLP"], help="Model architecture to instantiate")
    parser.add_argument("--gt_attr", default="exchange", type=str, help="Batch attribute name that holds ground-truth J values")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Which data split to evaluate")
    parser.add_argument("--batch_size", default=32, type=int, help="DataLoader batch size")
    parser.add_argument("--units", default="meV", type=str, help="Physical units label shown on plot axes")
    parser.add_argument("--out", default="visualizations/J_correlation.pdf", type=str, help="Output file path (.pdf recommended)")
    parser.add_argument("--mps", action="store_true", help="Use Apple MPS backend if available (same flag as main.py)")
    parser.add_argument("--no_show", action="store_true", help="Suppress plt.show() — useful on headless servers")

    args = parser.parse_args()
    main(args)