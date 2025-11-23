#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gkp_overlap_qcomb_plot_from_data.py

Convenience plotting script that ONLY reads:
    data_gkp_overlap_qcomb/gkp_overlap_qcomb_scan.npz

and regenerates heatmaps of:
    |<0_L|1_L>|^2

as a function of (N, Δ) for each squeezing setting, reported in dB.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import matplotlib
# ----------------- Matplotlib aesthetics -----------------
matplotlib.use("Agg")  # Use the 'Agg' backend for PNG output

plt.rcParams.update({
    "font.size": 30,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": True,
})
mpl.rcParams["axes.unicode_minus"] = False


def main():
    data_path = "data_gkp_overlap_qcomb/gkp_overlap_qcomb_scan.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find '{data_path}'. "
            "Run gkp_overlap_qcomb_scan.py first."
        )

    data = np.load(data_path, allow_pickle=True)

    N_list = data["N_list"]        # (nN,)
    r_list = data["r_list"]        # (nR,)
    r_dB_list = data["r_dB_list"]  # (nR,)
    Delta_list = data["Delta_list"]  # (nD,)
    overlap = data["overlap"]      # (nR, nN, nD)

    fig_dir = "figs_gkp_overlap_qcomb_from_data"
    os.makedirs(fig_dir, exist_ok=True)

    print("=== Re-plotting GKP q-comb overlap from stored data ===")
    print(f"N_list     = {N_list}")
    print("r_dB_list  = [" + ", ".join(f"{rdB:.2f} dB" for rdB in r_dB_list) + "]")
    print("Delta_list = [" + ", ".join(f"{d:.3f}" for d in Delta_list) + "]")
    print()

    finite = overlap[np.isfinite(overlap)]
    if finite.size > 0:
        vmin = 0.0
        vmax = float(np.nanmax(finite))
        vmax = max(vmax, 1e-6)
        vmax = min(vmax, 1.0)
    else:
        vmin, vmax = 0.0, 1.0

    N_arr = np.array(N_list, dtype=float)
    D_arr = np.array(Delta_list, dtype=float)

    for i_r, (r, r_dB) in enumerate(zip(r_list, r_dB_list)):
        data_slice = overlap[i_r]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            data_slice,
            origin="lower",
            aspect="auto",
            extent=[D_arr[0], D_arr[-1], N_arr[0], N_arr[-1]],
            cmap="bwr",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel(r"$\Delta$")
        ax.set_ylabel(r"$N$")
        ax.set_title(
            r"$|\langle 0_L|1_L\rangle|^2$"
            + fr"  (squeezing ≈ {r_dB:.1f} dB)"
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|\langle 0_L|1_L\rangle|^2$")
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        fig.tight_layout()
        outfig = os.path.join(
            fig_dir,
            f"overlap_from_data_N_Delta_squeezing_{r_dB:+04.1f}dB.png",
        )
        fig.savefig(outfig, dpi=230)
        plt.close(fig)
        print(f"Saved {outfig}")

    print("\nDone re-plotting. All figures in ./figs_gkp_overlap_qcomb_from_data/\n")


if __name__ == "__main__":
    main()
