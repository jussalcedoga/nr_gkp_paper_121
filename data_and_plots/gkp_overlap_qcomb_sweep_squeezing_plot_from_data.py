#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gkp_overlap_qcomb_sweep_squeezing_plot_from_data.py

Re-plot stored overlap heatmaps using a LOG-SCALE color normalization.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import matplotlib
# ----------------- Matplotlib aesthetics -----------------
matplotlib.use("Agg")  # Use the 'Agg' backend for PNG output

plt.rcParams.update({
    "font.size": 25,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": True,
})
# mpl.rcParams["axes.unicode_minus"] = False


def main():
    data_dir = "data_gkp_overlap_qcomb"
    figs_dir = "figs_gkp_overlap_qcomb_sweep_squeezing_from_data_log"
    os.makedirs(figs_dir, exist_ok=True)

    data_path = os.path.join(
        data_dir, "gkp_overlap_qcomb_sweep_squeezing.npz"
    )
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Could not find data file at '{data_path}'. "
            "Run gkp_overlap_qcomb_sweep_squeezing.py first."
        )

    data = np.load(data_path, allow_pickle=True)

    N_list          = data["N_list"]
    squeeze_dB_list = data["squeeze_dB_list"]
    Delta_list      = data["Delta_list"]
    overlap         = data["overlap"]     # (nD, nN, nS)

    print("=== Loaded q-comb overlap scan data ===")
    print(f"N_list          = {N_list}")
    print(f"squeeze_dB_list = {squeeze_dB_list} dB")
    print(f"Delta_list      = {Delta_list}")
    print(f"overlap shape   = {overlap.shape}")
    print()

    # Slight floor to avoid log(0)
    eps = 1e-12
    overlap_safe = np.clip(overlap, eps, None)

    cmap = "bwr_r"  # Better for log-scale perception

    s_arr = squeeze_dB_list.astype(float)
    N_arr = N_list.astype(float)

    for iD, Delta in enumerate(Delta_list):
        data_D = overlap_safe[iD]  # shape: (nN, nS)

        fig, ax = plt.subplots(figsize=(7, 5))

        im = ax.imshow(
            data_D,
            origin="lower",
            aspect="auto",
            extent=[s_arr[0], s_arr[-1], N_arr[0], N_arr[-1]],
            cmap=cmap,
            norm=LogNorm(vmin=eps, vmax=np.max(data_D)),
        )

        ax.set_xlabel(r"$s \ \rm{[dB]}$", fontsize=30)
        ax.set_ylabel(r"$N$", fontsize=30)
        # ax.set_title(
        #     rf"$|\langle 0_L|1_L\rangle|^2$   (\ = {Delta:.3f})"
        # )
        
        ax.text(
            0.03,
            0.05,
            rf"$\Delta = {Delta:.1f}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=25,
            color="white",
            # bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.4)
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|\langle 0_L|1_L\rangle|^2$")
        cbar.set_ticks([1e-1, 1e-3, 1e-6, 1e-10])

        fig.tight_layout()
        fpath = os.path.join(
            figs_dir,
            f"overlap_vs_N_squeezing_log_Delta{iD}_{Delta:.3f}.png"
        )
        fig.savefig(fpath, dpi=300)
        plt.close(fig)
        print(f"Saved log-scale heatmap for Δ={Delta:.3f} → {fpath}")

    print("\nDone (LOG scale).")
    print(f"Figures in ./{figs_dir}/")


if __name__ == "__main__":
    main()
