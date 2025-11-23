#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
plot_gkp_cnot_qcomb_from_data.py

Lightweight plotting script for GKP CNOT scan results.

Loads:
  data_gkp_cnot_qcomb/gkp_cnot_scan_data.npz

and regenerates:
  figs_gkp_cnot_qcomb_from_data/
    overlap_N_vs_Delta.png
    Favg_SUM_N_vs_Delta.png
    Fstate_truth_table_SUM.png

No QuTiP needed, only numpy + matplotlib.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ────────── GLOBAL STYLE (edit this freely) ──────────
plt.rcParams.update({
    "font.size": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": False,
})
mpl.rcParams["axes.unicode_minus"] = False


def plot_heatmap_N_Delta(N_list, Delta_list, data,
                         title, cbar_label, fname,
                         vmin=None, vmax=None,
                         outdir="figs_gkp_cnot_qcomb_from_data"):
    os.makedirs(outdir, exist_ok=True)

    N_arr = np.array(N_list, dtype=float)
    D_arr = np.array(Delta_list, dtype=float)

    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[D_arr[0], D_arr[-1], N_arr[0], N_arr[-1]],
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$N$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()

    outpath = os.path.join(outdir, fname)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_truth_table_fidelities(N_list, Delta_list, F_state_grid,
                                outdir="figs_gkp_cnot_qcomb_from_data"):
    """
    F_state_grid: shape (4, nN, nD)
      index 0 → |00> → |00>
      index 1 → |01> → |01>
      index 2 → |10> → |11>
      index 3 → |11> → |10>
    """
    os.makedirs(outdir, exist_ok=True)

    labels = [
        r"$|0_L,0_L\rangle \to |0_L,0_L\rangle$",
        r"$|0_L,1_L\rangle \to |0_L,1_L\rangle$",
        r"$|1_L,0_L\rangle \to |1_L,1_L\rangle$",
        r"$|1_L,1_L\rangle \to |1_L,0_L\rangle$",
    ]

    N_arr = np.array(N_list, dtype=float)
    D_arr = np.array(Delta_list, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    vmin, vmax = 0.0, 1.0

    for idx in range(4):
        ax = axes[idx // 2, idx % 2]
        data = F_state_grid[idx]
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[D_arr[0], D_arr[-1], N_arr[0], N_arr[-1]],
            cmap="bwr",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(r"$\Delta$")
        ax.set_ylabel(r"$N$")
        ax.set_title(labels[idx])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$F_{\mathrm{state}}$")

    fig.suptitle("Per-basis GKP CNOT state fidelities (SUM gate, q-comb)", y=0.95)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    outpath = os.path.join(outdir, "Fstate_truth_table_SUM.png")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    data_dir = "data_gkp_cnot_qcomb"
    fname = "gkp_cnot_scan_data.npz"
    path = os.path.join(data_dir, fname)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Data file '{path}' not found. "
            "Run scan_gkp_cnot_qcomb.py first."
        )

    data = np.load(path, allow_pickle=True)

    N_list = data["N_list"]
    Delta_list = data["Delta_list"]
    overlap_Z_grid = data["overlap_Z_grid"]
    Favg_SUM_grid = data["Favg_SUM_grid"]
    theta_opt_grid = data["theta_opt_grid"]
    F_state_grid = data["F_state_grid"]

    # Optionally you can also read M, r, etc. from the file:
    M = int(data["M"])
    r = float(data["r"])
    theta0 = float(data["theta0"])
    theta_span = float(data["theta_span"])
    n_theta = int(data["n_theta"])

    print("Loaded scan data from", path)
    print(f"M={M}, r={r}")
    print(f"N_list     = {list(N_list)}")
    print(f"Delta_list = {[float(d) for d in Delta_list]}")
    print(f"θ search: center≈√π={theta0:.3f}, span={theta_span}, n={n_theta}\n")

    outdir = "figs_gkp_cnot_qcomb_from_data"

    # Re-generate figures (feel free to tweak styles above)
    plot_heatmap_N_Delta(
        N_list, Delta_list,
        overlap_Z_grid,
        title=r"Logical overlap $|\langle 0_L|1_L\rangle|^2$",
        cbar_label=r"$|\langle 0_L|1_L\rangle|^2$",
        fname="overlap_N_vs_Delta.png",
        vmin=0.0,
        vmax=float(np.max(overlap_Z_grid)),
        outdir=outdir,
    )

    plot_heatmap_N_Delta(
        N_list, Delta_list,
        Favg_SUM_grid,
        title=r"Logical CNOT $F_{\mathrm{avg}}$ (SUM gate, q-comb)",
        cbar_label=r"$F_{\mathrm{avg}}$",
        fname="Favg_SUM_N_vs_Delta.png",
        vmin=0.0,
        vmax=1.0,
        outdir=outdir,
    )

    plot_truth_table_fidelities(
        N_list, Delta_list,
        F_state_grid,
        outdir=outdir,
    )

    print("\nDone re-plotting from data.")
    print(f"Figures in ./{outdir}/")


if __name__ == "__main__":
    main()
