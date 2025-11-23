#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gkp_cnot_gyrator_sweep_plot_from_data.py

Re-plot stored CNOT fidelity heatmaps from

  data_gkp_cnot_gyrator_sweep/gkp_cnot_gyrator_sweep.npz

and generate figures of:

    • F_avg(N, θ)              : average logical gate fidelity
    • F_state_k(N, θ)          : per-basis logical state fidelity

for k = 0,1,2,3 :
    k = 0: |0_L,0_L> → |0_L,0_L>
    k = 1: |0_L,1_L> → |0_L,1_L>
    k = 2: |1_L,0_L> → |1_L,1_L>
    k = 3: |1_L,1_L> → |1_L,0_L>

Each heatmap:
    - uses the bwr colormap,
    - has math-mode colorbar tick labels,
    - has a black text label in the upper-right corner with Δ and s (dB),
    - NO title on the axes (clean panel style).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ────────── GLOBAL STYLE (match overlap figures) ──────────
plt.rcParams.update({
    "font.size": 25,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": True,
})
mpl.rcParams["axes.unicode_minus"] = False


def main():
    data_dir = "data_gkp_cnot_gyrator_sweep"
    figs_dir = "figs_gkp_cnot_gyrator_sweep_from_data"
    os.makedirs(figs_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "gkp_cnot_gyrator_sweep.npz")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Could not find data file at '{data_path}'. "
            "Run gkp_cnot_gyrator_sweep.py first."
        )

    data = np.load(data_path, allow_pickle=True)

    N_list     = data["N_list"]          # (nN,)
    theta_list = data["theta_list"]      # (nT,)
    Delta_list = data["Delta_list"]      # (nD,)
    F_avg      = data["F_avg"]           # (nD, nN, nT)
    F_state    = data["F_state"]         # (nD, nN, nT, 4)
    M          = int(data["M"])
    s_dB       = float(data["squeeze_dB"])
    r_sq       = float(data["r_squeeze"])

    print("=== Re-plotting GKP CNOT gyrator scan from stored data ===")
    print(f"M          = {M}")
    print(f"squeeze_dB = {s_dB:.2f} dB (r = {r_sq:.3f})")
    print(f"N_list     = {N_list}")
    print(f"theta_list = {theta_list}")
    print(f"Delta_list = {Delta_list}")
    print()

    N_arr  = np.array(N_list, dtype=float)
    th_arr = np.array(theta_list, dtype=float)

    # ------------ helper: cbar formatter in math mode ------------
    def math_tick_formatter(y, pos):
        return r"$%.2f$" % y

    # ------------ helper: linear heatmap with in-plot label ------------
    def plot_heatmap(ax, grid, cbar_label, Delta_val,
                     s_db_val, cmap="bwr"):
        # vmin fixed at 0, vmax from data (ignoring NaNs)
        finite = grid[np.isfinite(grid)]
        if finite.size > 0:
            vmax = float(finite.max())
            vmax = max(vmax, 1e-6)  # avoid zero range
        else:
            vmax = 1.0

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[th_arr[0], th_arr[-1], N_arr[0], N_arr[-1]],
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlabel(r"$\theta \ \mathrm{[rad.]}$")
        ax.set_ylabel(r"$N$")

        # Colorbar with math-mode tick labels
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(math_tick_formatter))

        annotation_text = (
            r"$\begin{array}{rl}"
            rf"\Delta & = {Delta_val:.1f} \\"
            rf"s & = {s_db_val:.1f}\,\mathrm{{dB}}"
            r"\end{array}$"
        )

        ax.text(
            0.97,
            0.95,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=22,
            color="black",
        )

        return im

    # Labels for the four basis transitions (for filenames only)
    basis_labels = [
        "00_to_00",
        "01_to_01",
        "10_to_11",
        "11_to_10",
    ]

    # ---------- loop over Δ ----------
    for iD, Delta in enumerate(Delta_list):
        F_avg_D   = F_avg[iD]     # shape: (nN, nT)
        F_state_D = F_state[iD]   # shape: (nN, nT, 4)

        # ---- compute optimal θ using the largest N row of F_avg ----
        last_row = F_avg_D[-1, :]              # F_avg at N = N_list[-1]
        if np.all(np.isnan(last_row)):
            print(f"[Δ = {Delta:.3f}] No finite F_avg for largest N = {N_list[-1]}")
        else:
            j_opt = int(np.nanargmax(last_row))
            theta_opt = theta_list[j_opt]
            F_opt = last_row[j_opt]
            print(
                f"[Δ = {Delta:.3f}] optimal θ at largest N = {N_list[-1]}: "
                f"θ_opt = {theta_opt:.3f} rad, F_avg = {F_opt:.4f}"
            )

        # ------------------ Average gate fidelity ------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_heatmap(
            ax,
            F_avg_D,
            cbar_label=r"$F_{\mathrm{avg}}$",
            Delta_val=Delta,
            s_db_val=s_dB,
            cmap="bwr",
        )
        fig.tight_layout()
        fpath = os.path.join(
            figs_dir,
            f"Favg_vs_N_theta_Delta{iD}_Delta{Delta:.3f}.png"
        )
        fig.savefig(fpath, dpi=250)
        plt.close(fig)
        print(f"Saved {fpath}")

        # ------------------ Per-basis state fidelities -------------
        for k in range(4):
            grid_k = F_state_D[:, :, k]

            fig, ax = plt.subplots(figsize=(7, 5))
            plot_heatmap(
                ax,
                grid_k,
                cbar_label=r"$F_{\mathrm{state}}$",
                Delta_val=Delta,
                s_db_val=s_dB,
                cmap="bwr",
            )
            fig.tight_layout()
            fpath = os.path.join(
                figs_dir,
                f"Fstate{k}_{basis_labels[k]}_vs_N_theta_Delta{iD}_Delta{Delta:.3f}.png"
            )
            fig.savefig(fpath, dpi=300)
            plt.close(fig)
            print(f"Saved {fpath}")

    print("\nDone re-plotting. All figures in ./figs_gkp_cnot_gyrator_sweep_from_data/\n")


if __name__ == "__main__":
    main()
