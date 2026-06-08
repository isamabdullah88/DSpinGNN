"""
visualizations/strain_analysis.py
=========================================
Computes per-atom local strain tensors from the deformation gradient relative
to frame 0 and reports:

  1. Peak compressive principal strain across all / Cr-only atoms
  2. Local strain at the AFM interference centre
  3. Above-threshold duration — consecutive frames where peak Cr strain
     exceeds `threshold` × 100 %, multiplied by the 5 fs write interval
  4. Spatial extent — convex hull area and effective radius of the
     above-threshold Cr-atom patch at the peak strain frame

Theory
------
For atom i with neighbours j within `cutoff` Å:

    F_i  = (Σ_j r_ij ⊗ R_ij) · (Σ_j R_ij ⊗ R_ij)^{-1}   deformation gradient
    ε_i  = (F_i + F_i^T) / 2  −  I                         linearised strain tensor

Eigenvalues of ε_i are the principal strains (ascending).
Column 0 = most compressive (most negative).

Usage
-----
python visualizations/strain_analysis.py \
    --traj    sim_output/trajectory.xyz \
    --frames  36 89 \
    --cutoff  6.0 \
    --threshold 0.06 \
    --window  50
"""

import argparse
import numpy as np
from pathlib import Path
from ase.io import read
from ase.neighborlist import NeighborList
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── constants ─────────────────────────────────────────────────────────────────
Z_CR             = 24
FRAME_INTERVAL_FS = 5.0   # trajectory write interval (fs)

_PUB_RC = {
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "xtick.major.size": 5,    "ytick.major.size": 5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.linestyle": "--",   "grid.color": "#D0D0D0",
    "grid.linewidth": 0.8,
    "legend.frameon": True,   "legend.edgecolor": "#AAAAAA",
    "legend.fontsize": 10,    "legend.framealpha": 1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Core strain calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_local_strain(ref_atoms, cur_atoms, cutoff=6.0):
    """
    Per-atom deformation gradient → linearised strain → principal strains.

    Returns ndarray (N, 3): principal strains per atom, ascending.
    Column 0 = most compressive eigenvalue.
    """
    N       = len(ref_atoms)
    pos_ref = ref_atoms.get_positions()
    pos_cur = cur_atoms.get_positions()
    cell    = ref_atoms.get_cell().array

    nl = NeighborList([cutoff / 2.0] * N, skin=0.0,
                      self_interaction=False, bothways=True)
    nl.update(ref_atoms)

    eigvals = np.zeros((N, 3))
    for i in range(N):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) < 3:
            continue
        R_ij = pos_ref[indices] + offsets @ cell - pos_ref[i]
        r_ij = pos_cur[indices] + offsets @ cell - pos_cur[i]
        A = R_ij.T @ R_ij
        B = r_ij.T @ R_ij
        try:
            F   = B @ np.linalg.inv(A)
            eps = 0.5 * (F + F.T) - np.eye(3)
            eigvals[i] = np.linalg.eigvalsh(eps)
        except np.linalg.LinAlgError:
            pass
    return eigvals

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_nearest_atom(atoms, xy_target):
    pos_2d = atoms.get_positions()[:, :2]
    return int(np.argmin(np.linalg.norm(pos_2d - xy_target, axis=1)))


def interference_centre(atoms):
    cr_mask = atoms.get_atomic_numbers() == Z_CR
    for key in ("Local_J", "local_j"):
        try:
            local_j = atoms.get_array(key)
            break
        except RuntimeError:
            continue
    else:
        raise KeyError("No 'Local_J' / 'local_j' array in this frame.")
    j_cr   = local_j[cr_mask]
    pos_cr = atoms.get_positions()[cr_mask, :2]
    afm    = j_cr < 0
    if afm.sum() < 3:
        return pos_cr[np.argmin(j_cr)]
    return np.average(pos_cr[afm], axis=0, weights=-j_cr[afm])

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Above-threshold duration
# ─────────────────────────────────────────────────────────────────────────────

def scan_peak_cr_strain(frames, ref_atoms, frame_indices, cutoff):
    """
    Compute peak Cr-only compressive principal strain for each index in
    `frame_indices`.  Prints a progress line.

    Returns dict {frame_idx: peak_strain (signed, negative = compression)}.
    """
    results = {}
    total   = len(frame_indices)
    for k, fidx in enumerate(sorted(frame_indices)):
        if fidx < 0 or fidx >= len(frames):
            continue
        print(f"\r[scan]   frame {fidx:4d}  ({k + 1}/{total})", end="", flush=True)
        eigvals = compute_local_strain(ref_atoms, frames[fidx], cutoff)
        cr_mask = frames[fidx].get_atomic_numbers() == Z_CR
        results[fidx] = float(eigvals[cr_mask, 0].min())
    print()
    return results


def above_threshold_duration(peak_by_frame, threshold):
    """
    Find maximal runs of consecutive frame indices where the Cr peak strain
    is more compressive than -threshold (i.e., peak < -threshold).

    Assumes frame indices are 1-apart (fixed write interval).

    Returns list of dicts:
        start_frame, end_frame, n_frames, duration_fs,
        start_fs (= start_frame * FRAME_INTERVAL_FS),
        end_fs   (= end_frame   * FRAME_INTERVAL_FS)
    """
    sorted_frames = sorted(peak_by_frame)
    runs, in_run, run_start = [], False, None

    for i, fidx in enumerate(sorted_frames):
        exceeds = peak_by_frame[fidx] < -threshold
        # Detect gap (non-consecutive index breaks the run)
        gap = (i > 0) and (fidx - sorted_frames[i - 1] > 1)

        if exceeds and (not in_run or gap):
            if in_run and gap:          # close interrupted run
                prev = sorted_frames[i - 1]
                n    = prev - run_start + 1
                runs.append(_make_run(run_start, prev, n))
            in_run    = True
            run_start = fidx
        elif not exceeds and in_run:
            prev = sorted_frames[i - 1]
            n    = prev - run_start + 1
            runs.append(_make_run(run_start, prev, n))
            in_run = False

    if in_run:
        last = sorted_frames[-1]
        runs.append(_make_run(run_start, last, last - run_start + 1))

    return runs


def _make_run(start, end, n):
    return {
        'start_frame': start,
        'end_frame':   end,
        'n_frames':    n,
        'duration_fs': n * FRAME_INTERVAL_FS,
        'start_fs':    start * FRAME_INTERVAL_FS,
        'end_fs':      end   * FRAME_INTERVAL_FS,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Spatial extent at peak-strain frame
# ─────────────────────────────────────────────────────────────────────────────

def spatial_extent_above_threshold(atoms, eigvals, threshold):
    """
    Find Cr atoms whose most compressive principal strain < -threshold and
    compute the convex-hull area and effective radius of that patch.

    Returns dict:
        n_atoms    — number of above-threshold Cr atoms
        area_nm2   — convex hull area (nm²)
        radius_nm  — effective radius sqrt(area / π)  (nm)
        pos_nm     — (M, 2) positions of the above-threshold atoms (nm)
        hull       — scipy ConvexHull object (or None if < 3 atoms)
    """
    cr_mask     = atoms.get_atomic_numbers() == Z_CR
    eps_cr      = eigvals[cr_mask, 0]
    pos_cr_nm   = atoms.get_positions()[cr_mask, :2] / 10   # Å → nm
    exceed_mask = eps_cr < -threshold
    n_exceed    = int(exceed_mask.sum())

    if n_exceed < 3:
        return {'n_atoms': n_exceed, 'area_nm2': None, 'radius_nm': None,
                'pos_nm': pos_cr_nm[exceed_mask], 'hull': None}

    pts = pos_cr_nm[exceed_mask]
    try:
        hull      = ConvexHull(pts)
        area_nm2  = hull.volume       # ConvexHull.volume = area in 2-D
        radius_nm = np.sqrt(area_nm2 / np.pi)
    except QhullError:
        # Near-collinear fallback: use bounding-box area
        area_nm2  = float(pts[:, 0].ptp() * pts[:, 1].ptp())
        radius_nm = np.sqrt(area_nm2 / np.pi)
        hull      = None

    return {'n_atoms':   n_exceed,
            'area_nm2':  float(area_nm2),
            'radius_nm': float(radius_nm),
            'pos_nm':    pts,
            'hull':      hull}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_strain_map(atoms, eps_min_col, frame_idx, centre_xy, out_path):
    plt.rcParams.update(_PUB_RC)
    pos     = atoms.get_positions()[:, :2] / 10
    eps_pct = eps_min_col * 100
    clim    = np.percentile(np.abs(eps_pct), 98)

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=eps_pct, cmap="RdBu",
                    s=8, vmin=-clim, vmax=clim, linewidths=0)
    ax.scatter(*centre_xy / 10, marker="x", s=120, color="k",
               linewidths=1.5, zorder=5, label="Interference centre")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, shrink=0.7)
    cb.set_label("Min. principal strain (%)", fontsize=10)
    cb.ax.tick_params(labelsize=9, width=0.8)
    cb.outline.set_linewidth(0.8)
    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$y$ (nm)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", handletextpad=0.3)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"[plot]   Saved → {out_path}")
    plt.close(fig)


def plot_strain_histogram(eps_min_col, frame_idx, eps_peak, eps_centre, out_path):
    plt.rcParams.update(_PUB_RC)
    eps_pct = eps_min_col * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(eps_pct, bins=60, color="#1f77b4", edgecolor="white", linewidth=0.3)
    ax.axvline(eps_peak   * 100, color="#d62728", linewidth=1.5, linestyle="--", label="Peak compression")
    ax.axvline(eps_centre * 100, color="k",       linewidth=1.5, linestyle=":",  label="Interference centre")
    ax.axvline(0,               color="gray",     linewidth=0.8, linestyle="-")
    ax.set_xlabel("Min. principal strain (%)")
    ax.set_ylabel("Atom count")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"[plot]   Saved → {out_path}")
    plt.close(fig)


def plot_duration_scan(peak_by_frame, threshold, runs, target_frame, out_path):
    """Line plot of peak Cr strain vs frame index with threshold and event bands."""
    plt.rcParams.update(_PUB_RC)
    sorted_f = sorted(peak_by_frame)
    strains  = [peak_by_frame[f] * 100 for f in sorted_f]

    fig, ax = plt.subplots(figsize=(5, 4))
    for run in runs:
        ax.axvspan(run['start_frame'], run['end_frame'],
                   color="#1f77b4", alpha=0.15, zorder=1)
    ax.axhline(-threshold * 100, color="#d62728", linewidth=1.2,
               linestyle="--", label=f"Threshold ({threshold * 100:.0f}%)", zorder=3)
    ax.plot(sorted_f, strains, color="#1f77b4", linewidth=1.8,
            label="Peak Cr strain", zorder=4)
    ax.axvline(target_frame, color="k", linewidth=1.0, linestyle=":",
               label=f"Frame {target_frame}", zorder=5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Peak Cr strain (%)")
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"[plot]   Saved → {out_path}")
    plt.close(fig)


def plot_spatial_extent(atoms, eigvals, threshold, frame_idx, extent, out_path):
    """Scatter map highlighting above-threshold Cr atoms and their convex hull."""
    plt.rcParams.update(_PUB_RC)
    cr_mask    = atoms.get_atomic_numbers() == Z_CR
    pos_cr_nm  = atoms.get_positions()[cr_mask, :2] / 10
    eps_cr     = eigvals[cr_mask, 0]
    below_mask = eps_cr >= -threshold    # FM / background Cr atoms

    fig, ax = plt.subplots(figsize=(5, 5))
    # Background Cr atoms
    ax.scatter(pos_cr_nm[below_mask, 0], pos_cr_nm[below_mask, 1],
               s=12, color="#BBBBBB", linewidths=0, zorder=2, label="Cr (below threshold)")
    # Above-threshold atoms
    ax.scatter(extent['pos_nm'][:, 0], extent['pos_nm'][:, 1],
               s=25, color="#d62728", edgecolors="white", linewidths=0.5,
               zorder=4, label=f"Cr strain > {threshold*100:.0f}%  (N={extent['n_atoms']})")
    # Convex hull
    if extent['hull'] is not None:
        hull = extent['hull']
        for simplex in hull.simplices:
            ax.plot(extent['pos_nm'][simplex, 0], extent['pos_nm'][simplex, 1],
                    color="#d62728", linewidth=1.2, zorder=5)
        ax.text(0.97, 0.03,
                f"Area = {extent['area_nm2']:.2f} nm²\n"
                f"$r_{{eff}}$ = {extent['radius_nm']:.2f} nm",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox=dict(boxstyle="square,pad=0.4", facecolor="white",
                          edgecolor="#AAAAAA", linewidth=0.6))
    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$y$ (nm)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", handletextpad=0.3)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"[plot]   Saved → {out_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load]   Reading {args.traj} …")
    frames    = read(args.traj, index=":")
    n_frames  = len(frames)
    print(f"[load]   {n_frames} frames loaded.  Frame 0 used as reference.")

    ref = frames[0]
    bar = "─" * 56

    for fidx in args.frames:
        if fidx >= n_frames:
            print(f"[warn]   Frame {fidx} out of range ({n_frames} total) — skipping.")
            continue

        cur = frames[fidx]
        print(f"\n[strain] Computing local strain for frame {fidx} (cutoff = {args.cutoff} Å) …")
        eigvals = compute_local_strain(ref, cur, cutoff=args.cutoff)
        eps_min = eigvals[:, 0]

        eps_peak      = float(eps_min.min())
        eps_peak_atom = int(np.argmin(eps_min))
        centre_xy     = interference_centre(cur)
        centre_atom   = find_nearest_atom(cur, centre_xy)
        eps_centre    = float(eps_min[centre_atom])
        cr_mask       = cur.get_atomic_numbers() == Z_CR
        eps_cr_peak   = float(eps_min[cr_mask].min())

        # ── above-threshold duration ──────────────────────────────────────
        win_start = max(0, fidx - args.window)
        win_end   = min(n_frames - 1, fidx + args.window)
        win_range = list(range(win_start, win_end + 1))
        print(f"[scan]   Scanning {len(win_range)} frames [{win_start}–{win_end}] "
              f"for above-threshold duration …")
        peak_by_frame = scan_peak_cr_strain(frames, ref, win_range, args.cutoff)
        runs          = above_threshold_duration(peak_by_frame, args.threshold)

        # Find the run that contains fidx (or is closest)
        containing_run = None
        for run in runs:
            if run['start_frame'] <= fidx <= run['end_frame']:
                containing_run = run
                break

        # ── spatial extent ────────────────────────────────────────────────
        print(f"[extent] Computing spatial extent at frame {fidx} …")
        extent = spatial_extent_above_threshold(cur, eigvals, args.threshold)

        # ── report ────────────────────────────────────────────────────────
        print(f"\n{bar}")
        print(f"  Frame {fidx}  —  Local Strain Analysis")
        print(f"{bar}")
        print(f"  Cutoff / threshold       : {args.cutoff} Å  /  {args.threshold * 100:.0f}%")
        print(f"  Peak strain (all atoms)  : {eps_peak    * 100:+.1f}%  (atom {eps_peak_atom})")
        print(f"  Peak Cr-only strain      : {eps_cr_peak * 100:+.1f}%")
        print(f"  Strain at centre         : {eps_centre  * 100:+.1f}%  (atom {centre_atom})")
        print(f"  Interference centre      : ({centre_xy[0]/10:.2f}, {centre_xy[1]/10:.2f}) nm")
        print(f"{bar}")

        if containing_run:
            r = containing_run
            print(f"  Above-threshold episode  : frames {r['start_frame']}–{r['end_frame']}")
            print(f"  Duration                 : {r['duration_fs']:.0f} fs "
                  f"({r['start_fs']:.0f}–{r['end_fs']:.0f} fs)")
        else:
            print(f"  Above-threshold episode  : none containing frame {fidx} found in window")
        if len(runs) > 1:
            durs = [r['duration_fs'] for r in runs]
            print(f"  All episodes in window   : {len(runs)} × "
                  f"{min(durs):.0f}–{max(durs):.0f} fs")

        print(f"{bar}")
        if extent['n_atoms'] >= 3:
            print(f"  Cr atoms above threshold : {extent['n_atoms']}")
            print(f"  Convex hull area         : {extent['area_nm2']:.2f} nm²")
            print(f"  Effective radius         : {extent['radius_nm']:.2f} nm")
        else:
            print(f"  Cr atoms above threshold : {extent['n_atoms']}  (< 3, no hull)")
        print(f"{bar}")
        print(f"\n  ── Manuscript fill-in ──────────────────────────────────────")
        print(f"  Peak local strain: {eps_cr_peak * 100:.1f}%")
        if containing_run:
            r = containing_run
            print(f"  Above-threshold duration: {r['start_fs']:.0f}–{r['end_fs']:.0f} fs")
        if extent['n_atoms'] >= 3:
            print(f"  Spatial extent: {extent['radius_nm']:.2f} nm  "
                  f"(area {extent['area_nm2']:.2f} nm²)")
        print(f"  ────────────────────────────────────────────────────────────\n")

        # ── plots ─────────────────────────────────────────────────────────
        plot_strain_map(cur, eps_min, fidx, centre_xy,
                        out_dir / f"strain_map_frame{fidx:04d}.pdf")
        plot_strain_histogram(eps_min, fidx, eps_peak, eps_centre,
                              out_dir / f"strain_hist_frame{fidx:04d}.pdf")
        plot_duration_scan(peak_by_frame, args.threshold, runs, fidx,
                           out_dir / f"duration_scan_frame{fidx:04d}.pdf")
        if extent['n_atoms'] >= 3:
            plot_spatial_extent(cur, eigvals, args.threshold, fidx, extent,
                                out_dir / f"spatial_extent_frame{fidx:04d}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DSpinGNN — Per-atom local strain, duration, and spatial extent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--traj",      required=True,  type=str,   help="Path to trajectory.xyz")
    parser.add_argument("--frames",    nargs="+",       type=int,   default=[36, 89],
                        help="Peak frame indices to analyse")
    parser.add_argument("--cutoff",    default=6.0,    type=float,
                        help="Neighbour cutoff in Å")
    parser.add_argument("--threshold", default=0.06,   type=float,
                        help="Compressive strain threshold (fraction, e.g. 0.06 = 6%%)")
    parser.add_argument("--window",    default=50,     type=int,
                        help="Frames to scan on each side of each peak frame for duration")
    parser.add_argument("--out_dir",   default="visualizations/strain", type=str,
                        help="Output directory")
    args = parser.parse_args()
    main(args)