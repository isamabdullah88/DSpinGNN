"""
visualizations/domain_analysis.py
========================================
Extracts the radial domain-wall width ξ from a DSpinGNN trajectory by
fitting the radial Local_J profile at constructive-interference (peak-AFM)
frames to the hyperbolic-tangent domain-wall model (Eq. 6 in manuscript):

    J(r) = (J_FM + J_AFM)/2 + (J_FM - J_AFM)/2 · tanh((r - r₀) / ξ)

v2 fix: AFM sites are first clustered per frame with DBSCAN so that multiple
coexisting circular domains are each fitted independently. The original single-
centroid approach failed for frames with more than one domain: the phantom
centroid placed between domains produced a radial profile where negative and
positive J values averaged out at every shell, leaving no tanh transition to fit.

Usage (from repo root)
----------------------
python visualizations/domain_analysis.py \
    --traj      sim_output/trajectory.xyz \
    --out_dir   visualizations/domain_wall \
    --n_bins    40 \
    --eps       8.0 \
    --min_cluster_size 5 \
    --min_r2    0.5
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from ase.io import read
import matplotlib.pyplot as plt

# ── constants ─────────────────────────────────────────────────────────────────
Z_CR = 24           # atomic number of Chromium
ANG_PER_NM = 10.0   # 1 nm = 10 Å

# ── publication rcParams (PRB Compliant) ──────────────────────────────────────
_PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "grid.linestyle": "--",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 12,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Domain-wall model  (Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def tanh_model(r, j_fm, j_afm, r0, xi):
    """
    Bloch domain-wall profile.
    All of r, r0, xi must share the same unit; j_fm / j_afm in meV.

    At r → 0   : J → j_afm   (AFM core)
    At r → ∞   : J → j_fm    (FM bulk)
    At r = r0  : J = (j_fm + j_afm) / 2   (wall centre)
    """
    return (j_fm + j_afm) / 2.0 + (j_fm - j_afm) / 2.0 * np.tanh((r - r0) / xi)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Trajectory I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_frames(xyz_path):
    print(f"[load]   Reading {xyz_path} …")
    frames = read(xyz_path, index=":")
    print(f"[load]   {len(frames)} frames loaded.")
    return frames

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_cr(atoms):
    """
    Return (pos_2d, j_vals) for all Cr atoms only.

    pos_2d : (N_Cr, 2)  x-y positions in Å  (CrI3 is a 2-D material)
    j_vals : (N_Cr,)    Local_J per site — sum of ~3 nearest-neighbour J bonds
                        accumulated by scatter_add in DSpinGNNCalculator
    """
    cr_mask = atoms.get_atomic_numbers() == Z_CR
    for key in ("Local_J", "local_j"):
        try:
            local_j = atoms.get_array(key)
            break
        except RuntimeError:
            continue
    else:
        raise KeyError("No 'Local_J' / 'local_j' array found in this frame.")

    return atoms.get_positions()[cr_mask, :2], local_j[cr_mask]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  AFM-extent metric and peak detection
# ─────────────────────────────────────────────────────────────────────────────

def afm_fraction(j_vals):
    return float(np.mean(j_vals < 0))


def detect_peak_frames(frames, prominence, min_afm):
    """
    Build the per-frame AFM-fraction time-series and return local maxima
    (constructive-interference events).
    """
    extents = []
    for atoms in frames:
        try:
            _, j = extract_cr(atoms)
            extents.append(afm_fraction(j))
        except (KeyError, RuntimeError):
            extents.append(0.0)

    extents = np.array(extents)
    peak_idxs, _ = find_peaks(extents, prominence=prominence, height=min_afm)
    print(f"[detect] {len(peak_idxs)} constructive-interference event(s) detected "
          f"at frames: {peak_idxs.tolist()}")
    return peak_idxs, extents

# ─────────────────────────────────────────────────────────────────────────────
# 5.  General 2-D minimum-image convention
# ─────────────────────────────────────────────────────────────────────────────

def _mic_2d(delta_cart, cell_2x2):
    """
    2-D minimum-image correction.  Works for rectangular and hexagonal cells.

    delta_cart : (N, 2) Cartesian displacement vectors
    cell_2x2   : (2, 2) matrix whose rows are the 2-D lattice vectors
    """
    frac = np.linalg.solve(cell_2x2.T, delta_cart.T).T
    frac -= np.round(frac)
    return frac @ cell_2x2

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DBSCAN clustering of AFM sites  ← KEY FIX
# ─────────────────────────────────────────────────────────────────────────────

def find_afm_clusters(pos_2d, j_vals, cell_2x2, eps_ang, min_cluster_size):
    """
    Partition AFM (J < 0) Cr sites into spatially distinct domains using DBSCAN.

    Why this is necessary
    ---------------------
    Frames at the second (larger) AFM-fraction peak often contain multiple
    small AFM domains rather than one large coherent domain.  Computing a single
    centroid over all AFM atoms places it in the FM space between domains; the
    resulting radial profile mixes positive and negative J at every shell so
    the tanh transition disappears entirely — exactly what we observed in
    frame 89.  DBSCAN finds each connected AFM region independently.

    Parameters
    ----------
    eps_ang        : DBSCAN neighbourhood radius in Å.
                     Rule of thumb: ~2× the Cr-Cr nearest-neighbour distance
                     (~4 Å in CrI3) → default 8 Å.
    min_cluster_size : minimum AFM Cr atoms to form a cluster.

    Returns
    -------
    List of dicts: {'center': (x, y) in Å, 'n_afm': int}

    Note: DBSCAN operates in Cartesian space without PBC.  This is valid as
    long as each domain is smaller than half the cell in every direction.
    """
    afm_mask = j_vals < 0
    if int(afm_mask.sum()) < min_cluster_size:
        return []

    pos_afm = pos_2d[afm_mask]
    j_afm   = j_vals[afm_mask]

    labels = DBSCAN(eps=eps_ang, min_samples=min_cluster_size).fit_predict(pos_afm)

    clusters = []
    for lbl in sorted(set(labels) - {-1}):
        in_cluster = labels == lbl
        cp = pos_afm[in_cluster]
        cj = j_afm[in_cluster]
        weights = -cj                            # positive; more AFM → more weight

        # PBC-aware weighted centroid within this cluster
        anchor = cp[np.argmax(weights)]
        delta  = _mic_2d(cp - anchor, cell_2x2)
        center = anchor + np.average(delta, axis=0, weights=weights)

        clusters.append({'center': center, 'n_afm': int(in_cluster.sum())})

    return clusters

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Radial binning  (all Cr atoms, distance from a given centre)
# ─────────────────────────────────────────────────────────────────────────────

def radial_profile(pos_2d, j_vals, centre, cell_2x2, n_bins):
    """
    Compute mean and std of J in concentric 2-D shells around `centre`.
    Uses ALL Cr atoms so the profile spans AFM core → domain wall → FM bulk.
    Empty bins are dropped. All distances returned in Å.
    """
    delta = _mic_2d(pos_2d - centre, cell_2x2)
    r     = np.hypot(delta[:, 0], delta[:, 1])

    edges  = np.linspace(0.0, r.max(), n_bins + 1)
    r_mid  = 0.5 * (edges[:-1] + edges[1:])
    j_mean = np.full(n_bins, np.nan)
    j_std  = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if mask.sum() > 0:
            j_mean[i] = j_vals[mask].mean()
            j_std[i]  = j_vals[mask].std()
            counts[i] = int(mask.sum())

    valid = ~np.isnan(j_mean)
    return r_mid[valid], j_mean[valid], j_std[valid], counts[valid]

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Tanh fitting and R² quality check
# ─────────────────────────────────────────────────────────────────────────────

def fit_tanh(r, j_mean):
    """
    Fit tanh_model to the radial (r [Å], J [meV]) profile.
    Returns (popt, perr) on success, or None on failure.
    popt = [j_fm, j_afm, r0, xi]  (same Å / meV units as input).
    """
    j_fm_guess  = float(np.percentile(j_mean, 90))
    j_afm_guess = float(np.percentile(j_mean, 10))
    midpoint    = 0.5 * (j_fm_guess + j_afm_guess)
    r0_guess    = float(r[np.argmin(np.abs(j_mean - midpoint))])
    xi_guess    = max(1.0, (r.max() - r.min()) * 0.08)

    p0     = [j_fm_guess, j_afm_guess, r0_guess, xi_guess]
    bounds = ([-np.inf, -np.inf, float(r.min()), 0.5],
              [ np.inf,  np.inf, float(r.max()), float(r.max())])

    try:
        popt, pcov = curve_fit(tanh_model, r, j_mean, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        _, _, r0, xi = popt
        if xi <= 0 or xi > r.max() or r0 < 0:
            return None
        if popt[0] < popt[1]:                    # enforce j_fm > j_afm
            popt[[0, 1]] = popt[[1, 0]]
            perr[[0, 1]] = perr[[1, 0]]
        return popt, perr
    except (RuntimeError, ValueError):
        return None


def compute_r2(r, j_mean, popt):
    j_pred = tanh_model(r, *popt)
    ss_res = np.sum((j_mean - j_pred) ** 2)
    ss_tot = np.sum((j_mean - j_mean.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_afm_timeseries(extents, peak_idxs, out_path):
    plt.rcParams.update(_PUB_RC)
    fig, ax = plt.subplots(figsize=(3.375, 2.6), constrained_layout=True)
    
    ax.plot(extents, color="#1f77b4", linewidth=1.5, label="AFM Cr fraction")
    if len(peak_idxs) > 0:
        ax.scatter(peak_idxs, extents[peak_idxs], color="#d62728", s=40, zorder=5,
                   edgecolors="white", linewidths=0.6,
                   label="Peak interference events")
                   
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("AFM Cr Fraction")
    ax.grid(True, zorder=0)
    ax.legend(loc="upper right")
    
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"[plot]   Saved → {out_path}")
    plt.close(fig)


def plot_radial_fit(r_nm, j_mean, j_std, popt_nm, frame_idx, cluster_idx, xi_nm, xi_err_nm, r2, out_path):
    """All distances in nm, J in meV (accumulated units)."""
    plt.rcParams.update(_PUB_RC)
    r_fit = np.linspace(r_nm.min(), r_nm.max(), 400)
    j_fit = tanh_model(r_fit, *popt_nm)

    fig, ax = plt.subplots(figsize=(3.375, 2.6), constrained_layout=True)
    
    # FM / AFM boundary
    ax.axhline(0, color="#d62728", linewidth=1.0, linestyle="--", label="FM / AFM Boundary", zorder=2)
    
    # Radial profile
    ax.errorbar(r_nm, j_mean, yerr=j_std, fmt="o", ms=5, color="#1f77b4",
                mec="white", mew=0.5,
                ecolor="#CCCCCC", elinewidth=0.6, capsize=2, capthick=0.6,
                label="Radial profile", zorder=4)
                
    # Tanh fit
    # fit_label = r"Tanh fit ($\xi = {:.2f} \pm {:.2f}$ nm, $R^2 = {:.2f}$)".format(xi_nm, xi_err_nm, r2)
    fit_label = "Tanh fit"
    ax.plot(r_fit, j_fit, color="k", linewidth=1.5, label=fit_label, zorder=5)
    
    ax.set_xlabel(r"$r$ (nm)")
    ax.set_ylabel(r"$J$ (meV)")
    ax.grid(True, zorder=0)
    ax.legend(loc="upper left")
    
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"[plot]   Saved → {out_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.traj)
    peak_idxs, extents = detect_peak_frames(frames, prominence=args.prominence, min_afm=args.min_afm)
    plot_afm_timeseries(extents, peak_idxs, out_dir / "afm_timeseries.pdf")

    if len(peak_idxs) == 0:
        print("[warn]   No peak frames found. Try lowering --prominence or --min_afm.")
        return

    # Accumulate ξ across all (frame, cluster) events
    xi_nm_list, xi_err_list, event_labels = [], [], []

    for fidx in peak_idxs:
        atoms  = frames[fidx]
        pos_2d, j_vals = extract_cr(atoms)
        cell_2x2 = atoms.cell[:2, :2].copy()

        n_afm = int((j_vals < 0).sum())
        clusters = find_afm_clusters(pos_2d, j_vals, cell_2x2,
                                     eps_ang=args.eps, min_cluster_size=args.min_cluster_size)

        print(f"\n[frame {fidx:>4d}]  AFM Cr sites: {n_afm}  |  "
              f"DBSCAN clusters: {len(clusters)}")

        if len(clusters) == 0:
            print(f"           No clusters meet min_cluster_size={args.min_cluster_size} — skipping frame.")
            continue

        for cidx, cluster in enumerate(clusters):
            centre = cluster['center']
            print(f"  cluster {cidx}:  {cluster['n_afm']} AFM sites  |  "
                  f"centre = ({centre[0]/ANG_PER_NM:.2f}, {centre[1]/ANG_PER_NM:.2f}) nm")

            r, j_mean, j_std, counts = radial_profile(pos_2d, j_vals, centre, cell_2x2, args.n_bins)

            if len(r) < 8:
                print(f"             Too few radial bins ({len(r)}) — skipping cluster.")
                continue

            result = fit_tanh(r, j_mean)
            if result is None:
                print(f"             Tanh fit did not converge — skipping cluster.")
                continue

            popt, perr = result
            r2         = compute_r2(r, j_mean, popt)
            xi_ang     = abs(popt[3])
            xi_nm      = xi_ang / ANG_PER_NM
            xi_err_nm  = perr[3] / ANG_PER_NM
            r0_nm      = popt[2] / ANG_PER_NM

            print(f"             ξ = {xi_nm:.3f} ± {xi_err_nm:.3f} nm  |  "
                  f"r₀ = {r0_nm:.3f} nm  |  R² = {r2:.3f}  |  "
                  f"J_FM = {popt[0]:.2f}  J_AFM = {popt[1]:.2f} meV")

            if r2 < args.min_r2:
                print(f"             R² = {r2:.3f} < threshold {args.min_r2} — rejected.")
                continue

            xi_nm_list.append(xi_nm)
            xi_err_list.append(xi_err_nm)
            event_labels.append((fidx, cidx))

            r_nm    = r / ANG_PER_NM
            popt_nm = [popt[0], popt[1], popt[2] / ANG_PER_NM, popt[3] / ANG_PER_NM]
            plot_radial_fit(r_nm, j_mean, j_std, popt_nm, fidx, cidx, xi_nm, xi_err_nm, r2,
                            out_dir / f"radial_fit_frame{fidx:04d}_cluster{cidx}.pdf")

    if not xi_nm_list:
        print("\n[warn]   No fits passed the R² threshold. "
              "Try lowering --min_r2 or adjusting --eps / --min_cluster_size.")
        return

    xi_arr  = np.array(xi_nm_list)
    xi_mean = xi_arr.mean()
    xi_std  = xi_arr.std(ddof=1) if len(xi_arr) > 1 else xi_err_list[0]

    bar = "─" * 52
    print(f"\n{bar}")
    print("  Domain Wall Width  ξ  —  Summary")
    print(f"{bar}")
    print(f"  Events accepted  : {len(xi_arr)}")
    for (fidx, cidx), xi in zip(event_labels, xi_nm_list):
        print(f"    frame {fidx:>4d}  cluster {cidx}  →  ξ = {xi:.3f} nm")
    print(f"  ξ (mean ± std)   : {xi_mean:.2f} ± {xi_std:.2f} nm")
    print(f"{bar}")
    print(f"\n  ── Manuscript fill-in ──────────────────────────────────")
    print(f"  ξ = {xi_mean:.1f} ± {xi_std:.1f} nm   (N = {len(xi_arr)} events)")
    print(f"  ────────────────────────────────────────────────────────\n")

    summary_path = out_dir / "xi_summary.txt"
    with open(summary_path, "w") as f:
        f.write("frame\tcluster\txi_nm\txi_err_nm\n")
        for (fidx, cidx), xi, err in zip(event_labels, xi_nm_list, xi_err_list):
            f.write(f"{fidx}\t{cidx}\t{xi:.4f}\t{err:.4f}\n")
        f.write(f"\nMean xi (nm) : {xi_mean:.4f}\n")
        f.write(f"Std  xi (nm) : {xi_std:.4f}\n")
        f.write(f"N events     : {len(xi_arr)}\n")
    print(f"[out]    Summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSpinGNN — Domain wall width analysis (v2: per-cluster fitting)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--traj", required=True, type=str, help="Path to trajectory.xyz (extxyz format)")
    parser.add_argument("--out_dir", default="visualizations/domain_wall", type=str, help="Output directory")
    parser.add_argument("--n_bins", default=40, type=int, help="Number of radial bins per cluster profile")
    parser.add_argument("--eps", default=8.0, type=float, help="DBSCAN neighbourhood radius in Å (~2× Cr-Cr distance)")
    parser.add_argument("--min_cluster_size", default=5, type=int, help="Minimum AFM Cr atoms to form a cluster")
    parser.add_argument("--min_r2", default=0.5, type=float, help="Minimum R² to accept a tanh fit")
    parser.add_argument("--prominence", default=0.05, type=float, help="Peak-detection prominence in AFM fraction")
    parser.add_argument("--min_afm", default=0.05, type=float, help="Minimum AFM fraction to qualify as a peak event")
    args = parser.parse_args()
    main(args)