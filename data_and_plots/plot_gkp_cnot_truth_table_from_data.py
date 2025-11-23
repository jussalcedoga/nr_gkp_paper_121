#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
plot_gkp_cnot_truth_table_from_data.py

Lightweight plotting script for the GKP CNOT truth-table metrics
(using q-comb states).

Loads:
  data_gkp_cnot_truth_table/gkp_cnot_truth_table_qcomb.npz

and regenerates:
  - figs_truth_table_qcomb_from_data/state_fidelities_logical.png
  - figs_truth_table_qcomb_from_data/overlap_vs_Delta.png
  - figs_truth_table_qcomb_from_data/truth_table_*.png

No QuTiP needed; uses only numpy + matplotlib.
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
    "font.size": 25,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": True,
})
mpl.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------
#  Per-basis fidelities
# ---------------------------------------------------------
def plot_state_fidelities(
    ops_labels,
    F_vals,
    F_avg_opt,
    outdir="figs_truth_table_qcomb_from_data",
):
    """
    Bar plot of per-basis state fidelities F_state.
    """
    os.makedirs(outdir, exist_ok=True)

    x = np.arange(len(F_vals))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, F_vals, width=0.6, color="royalblue")

    ax.set_xticks(x)
    ax.set_xticklabels(ops_labels, rotation=20, ha="right", fontsize=25)
    # ax.set_ylabel(r"Per-basis logical state fidelity $F_{\mathrm{state}}$")
    ax.set_ylabel(r"$F_{\mathrm{state}}$", fontsize=30)

    ax.set_ylim(0.0, 1.0)
    ax.grid(False)

    # annotate bars with F_state values
    for xi, bi, Fi in zip(x, bars, F_vals):
        ax.text(
            xi,
            bi.get_height() + 0.02,
            rf"${Fi:.3f}$",
            ha="center",
            va="bottom",
            fontsize=25,
        )

    outpath = os.path.join(outdir, "state_fidelities_logical.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved logical fidelity summary figure to {outpath}")


# ---------------------------------------------------------
#  Overlap vs Delta
# ---------------------------------------------------------
def plot_overlap_vs_delta(
    Delta_scan,
    overlap_Z,
    outdir="figs_truth_table_qcomb_from_data",
):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Delta_scan, overlap_Z, "o-", lw=2)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$|\langle 0_L|1_L\rangle|^2$")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()

    outpath = os.path.join(outdir, "overlap_vs_Delta.png")
    fig.savefig(outpath, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlap-vs-Delta figure to {outpath}")


# ---------------------------------------------------------
#  Wigner panels from stored data (Option B labelling)
# ---------------------------------------------------------
def plot_wigner_panels_from_data(
    N,
    Delta_main,
    theta_opt,
    x_grid,
    Wigner_all,
    basis_order,
    panel_labels,
    outdir="figs_truth_table_qcomb_from_data",
):
    r"""
    Reconstruct the 2×2 Wigner truth-table panels for each basis input
    from stored Wigner arrays.

    Wigner_all shape: (4, 4, nX, nX)
      axis0: basis index  (0:00, 1:01, 2:10, 3:11)
      axis1: panel index  (0:control_in, 1:target_in,
                           2:control_out, 3:target_out)

    Panel layout (Option B):
        [0,0]: control mode, before SUM
        [0,1]: target mode,  before SUM
        [1,0]: control mode, after SUM
        [1,1]: target mode,  after SUM

    Lower-left text: logical ket (input or CNOT output).
    Upper-left text: "control" or "target".
    No titles, no Δ labels inside panels.
    """
    os.makedirs(outdir, exist_ok=True)

    X, P = np.meshgrid(x_grid, x_grid)
    n_basis = Wigner_all.shape[0]

    # We assume x_grid covers ~[-5, 5]
    tick_vals = [-5, 0, 5]
    x_min, x_max = x_grid[0], x_grid[-1]

    for idx_b in range(n_basis):
        basis_str = basis_order[idx_b]  # e.g. "00"
        c_label = basis_str[0]          # '0' or '1'
        t_label = basis_str[1]          # '0' or '1'

        c_int = int(c_label)
        t_int = int(t_label)
        t_out_int = c_int ^ t_int       # CNOT target = c ⊕ t
        t_out_label = str(t_out_int)

        # Logical labels:
        logical_control_in  = rf"$|{c_label}_L\rangle$"
        logical_target_in   = rf"$|{t_label}_L\rangle$"
        logical_control_out = rf"$|{c_label}_L\rangle$"
        logical_target_out  = rf"$|{t_out_label}_L\rangle$"

        W_basis = Wigner_all[idx_b]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(9, 8),
            constrained_layout=False,
        )
        # enlarge spacing to avoid overlaps of labels / colorbars
        fig.subplots_adjust(
            left=0.08,
            right=0.95,
            bottom=0.10,
            top=0.98,
            hspace=0.70,
            wspace=0.65,
        )

        all_W = W_basis
        vmax = float(np.max(np.abs(all_W)))
        vmin = -vmax

        def panel(ax, W, logical_label, role_text, n_ticks):
            cf = ax.contourf(
                X,
                P,
                W,
                220,
                cmap="bwr",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)

            # Ticks at -5, 0, 5 (as in your example)
            ax.set_xticks(tick_vals)
            ax.set_yticks(tick_vals)

            ax.set_xlabel(r"$q$", fontsize=26)
            ax.set_ylabel(r"$p$", fontsize=26)
            ax.tick_params(labelsize=22)
            ax.grid(ls=":", alpha=0.4)

            # Colorbar: use n_ticks = 5 (initial) or 9 (final)
            ticks = np.linspace(vmin, vmax, n_ticks)
            cb = fig.colorbar(
                cf,
                ax=ax,
                fraction=0.046,
                pad=0.03,
                ticks=ticks,
            )
            cb.ax.set_ylabel(r"$W(q,p)$", fontsize=22)
            cb.ax.tick_params(labelsize=20)

            if abs(vmax) > 0.01:
                fmt = r"$%.2f$"
            else:
                fmt = r"$%.4f$"
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

            # LOWER-LEFT annotation: logical state (no Δ)
            ax.text(
                0.03,
                0.05,
                logical_label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=22,
            )

            # UPPER-LEFT annotation: which subsystem (control / target)
            ax.text(
                0.03,
                0.95,
                role_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=22,
            )

        # Panels (no titles, only inside annotations)
        # Top row: initial states → 5 cbar ticks
        panel(
            axes[0, 0],
            all_W[0],
            logical_control_in,
            r"control",
            n_ticks=5,
        )
        panel(
            axes[0, 1],
            all_W[1],
            logical_target_in,
            r"target",
            n_ticks=5,
        )
        # Bottom row: post-CNOT states → 9 cbar ticks
        panel(
            axes[1, 0],
            all_W[2],
            logical_control_out,
            r"control",
            n_ticks=9,
        )
        panel(
            axes[1, 1],
            all_W[3],
            logical_target_out,
            r"target",
            n_ticks=9,
        )

        outname = (
            f"truth_table_ctrl{c_label}_tgt{t_label}_"
            f"N{N}_Delta{Delta_main:.3f}.png"
        )
        outpath = os.path.join(outdir, outname)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Wigner truth-table panel to {outpath}")

def plot_state_infidelities_logscale(
    ops_labels,
    F_vals,
    outdir="figs_truth_table_qcomb_from_data",
):
    """
    Bar plot of logical state *infidelities* in log scale:
        log10(1 - F_state)
    Lower bars = better performance.
    """
    os.makedirs(outdir, exist_ok=True)

    # Avoid log(0) by clipping very small numbers
    infid = np.clip(1.0 - np.array(F_vals), 1e-12, 1.0)
    log_inf = np.log10(infid)

    x = np.arange(len(F_vals))
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.bar(x, log_inf, width=0.6, color="firebrick")

    ax.set_xticks(x)
    ax.set_xticklabels(ops_labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\log_{10}(1 - F_{\mathrm{state}})$", fontsize=30)

    # annotate bars with actual infidelity values
    for xi, bi, li, fi in zip(x, bars, log_inf, F_vals):
        inf_display = (1.0 - fi)
        ax.text(
            xi,
            li + 0.05,
            rf"$\scriptstyle {inf_display:.1e}$",
            ha="center",
            va="bottom",
            fontsize=24,
        )

    fig.tight_layout()
    outpath = os.path.join(outdir, "state_infidelities_log10.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved log-scale infidelity summary figure to {outpath}")

# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------
def main():
    data_dir = "data_gkp_cnot_truth_table"
    fname = "gkp_cnot_truth_table_qcomb.npz"
    path = os.path.join(data_dir, fname)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Data file '{path}' not found. "
            "Run gkp_cnot_truth_table_qcomb.py first."
        )

    data = np.load(path, allow_pickle=True)

    N = int(data["N"])
    M = int(data["M"])
    r = float(data["r"])
    Delta_main = float(data["Delta_main"])
    theta0 = float(data["theta0"])
    theta_span = float(data["theta_span"])
    n_theta = int(data["n_theta"])
    theta_opt = float(data["theta_opt"])
    F_avg_opt = float(data["F_avg_opt"])
    op_labels = data["op_labels"]
    F_state = data["F_state"]
    x_grid = data["x_grid"]
    Wigner_all = data["Wigner_all"]
    Delta_scan = data["Delta_scan"]
    overlap_Z = data["overlap_Z"]
    panel_labels = data["panel_labels"]
    basis_order = data["basis_order"]

    print("Loaded truth-table data from", path)
    print(f"N={N}, M={M}, r={r}, Delta_main={Delta_main:.3f}")
    print(f"θ search: center≈√π={theta0:.3f}, span={theta_span}, n={n_theta}")
    print(f"θ_opt={theta_opt:.4f}, F_avg_opt={F_avg_opt:.4f}")
    print("Per-basis logical state fidelities (F_state):")
    for lbl, F in zip(op_labels, F_state):
        print(f"  {lbl} : {F:.4f}")
    print()

    # --- per-basis fidelity bar plot ---
    plot_state_fidelities(
        op_labels,
        F_state,
        F_avg_opt,
        outdir="figs_truth_table_qcomb_from_data",
    )

    # --- overlap vs Δ ---
    plot_overlap_vs_delta(
        Delta_scan,
        overlap_Z,
        outdir="figs_truth_table_qcomb_from_data",
    )

    # --- Wigner panels from stored data ---
    plot_wigner_panels_from_data(
        N,
        Delta_main,
        theta_opt,
        x_grid,
        Wigner_all,
        basis_order,
        panel_labels,
        outdir="figs_truth_table_qcomb_from_data",
    )

    plot_state_infidelities_logscale(
        op_labels,
        F_state,
        outdir="figs_truth_table_qcomb_from_data",
    )

    print("\nDone re-plotting from stored data.")
    print("Figures in ./figs_truth_table_qcomb_from_data/")


if __name__ == "__main__":
    main()
