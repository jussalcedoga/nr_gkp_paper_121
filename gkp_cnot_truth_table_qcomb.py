#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gkp_cnot_truth_table_qcomb.py

Visual GKP CNOT "truth table" using the non-reciprocal SUM gate
and the q-comb GKP construction:

    U_sum = exp(-i θ q1 p2)

Logical single-mode states are imported from
    gkp_qcomb_stateprep_panels.logical_eigenstates_qcomb

In that script, the codewords are built via a 1D q-comb with
Gaussian envelope

    w_s = exp[ - q_s^2 / (2 Δ^2) ],

so Δ is the *linewidth* of the comb envelope along q:
smaller Δ → sharper envelope → higher energy and better
orthogonality.

We:
  • choose a single (N, Δ_main) and optimize θ for the SUM gate using
    the logical average gate fidelity F_avg on the GKP codespace,
  • prepare the four computational-basis inputs
        |0_L,0_L>, |0_L,1_L>, |1_L,0_L>, |1_L,1_L>,
  • apply the optimized SUM gate, and
  • plot Wigner functions in 2×2 panels:

        top row:    initial control, initial target
        bottom row: final control,   final target

Also:
  • compute per-basis state fidelities for the ideal CNOT action,
  • compute the overlap |<0_L|1_L>|^2 as a function of Δ for a
    user-defined Delta_scan_list,
  • export all metrics and Wigner data to

        data_gkp_cnot_truth_table/gkp_cnot_truth_table_qcomb.npz

so that plots can be regenerated later without re-running the
full QuTiP simulation.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from concurrent.futures import ProcessPoolExecutor

from qutip import (
    destroy, qeye, tensor as qt_tensor,
    Qobj, wigner as qutip_wigner, ptrace
)

# q-comb logical states (Δ as linewidth of comb envelope in q)
from gkp_qcomb_stateprep_panels import logical_eigenstates_qcomb

# ────────── GLOBAL STYLE ──────────
plt.rcParams.update({
    "font.size": 30,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": False,   # consistent with other q-comb scripts
})
mpl.rcParams["axes.unicode_minus"] = False


# ============================================================
#   Two-mode gates & logical fidelity machinery
# ============================================================
def canonical_two_mode(N: int):
    """Return (a1, a2, q1, p1, q2, p2) in the 2-mode Fock space."""
    a = destroy(N)
    I = qeye(N)
    a1 = qt_tensor(a, I)
    a2 = qt_tensor(I, a)
    q1 = (a1 + a1.dag()) / np.sqrt(2.0)
    p1 = (a1 - a1.dag()) / (1j * np.sqrt(2.0))
    q2 = (a2 + a2.dag()) / np.sqrt(2.0)
    p2 = (a2 - a2.dag()) / (1j * np.sqrt(2.0))
    return a1, a2, q1, p1, q2, p2


def build_sum_gate(N: int, theta: float) -> Qobj:
    r"""Non-reciprocal SUM gate:  U = exp(-i θ q1 p2)."""
    a1, a2, q1, p1, q2, p2 = canonical_two_mode(N)
    H = q1 * p2
    return (-1j * theta * H).expm()


def build_logical_basis(psi0: Qobj, psi1: Qobj):
    """Return [|00_L>, |01_L>, |10_L>, |11_L>] in the two-mode space."""
    b00 = qt_tensor(psi0, psi0)
    b01 = qt_tensor(psi0, psi1)
    b10 = qt_tensor(psi1, psi0)
    b11 = qt_tensor(psi1, psi1)
    return [b00, b01, b10, b11]


def ideal_logical_cnot_matrix():
    r"""
    4×4 CNOT in computational basis (|00>,|01>,|10>,|11>):

      |00> -> |00>
      |01> -> |01>
      |10> -> |11>
      |11> -> |10>
    """
    U = np.eye(4, dtype=complex)
    U[2, 2] = 0.0
    U[3, 3] = 0.0
    U[2, 3] = 1.0
    U[3, 2] = 1.0
    return U


def project_unitary_to_logical(U_phys: Qobj, basis_kets):
    """
    Project physical 2-mode unitary onto the span of basis_kets.

    Returns:
        U_log : 4x4 numpy array, with elements
            (U_log)_{j,i} = <b_j | U_phys | b_i>
    """
    d = len(basis_kets)
    U_log = np.zeros((d, d), dtype=complex)
    for i, ket_in in enumerate(basis_kets):
        psi_out = U_phys * ket_in
        for j, ket_out in enumerate(basis_kets):
            U_log[j, i] = ket_out.overlap(psi_out)
    return U_log


def logical_average_gate_fidelity(U_log, U_ideal) -> float:
    r"""
    Average gate fidelity on the logical subspace:

      F_avg = ( |Tr(U_ideal^† U_log)|^2 + d ) / [ d (d+1) ],

    with d = 4 for a 2-logical-qubit gate.
    """
    d = U_ideal.shape[0]
    tr_term = np.trace(np.conjugate(U_ideal.T) @ U_log)
    F_avg = (abs(tr_term) ** 2 + d) / (d * (d + 1))
    return float(np.real(F_avg))


def ket_fidelity(psi: Qobj, phi: Qobj) -> float:
    ov = psi.overlap(phi)
    return float(abs(ov) ** 2)


# ============================================================
#   Wigner helper
# ============================================================
def single_mode_wigner(rho_or_psi: Qobj, x: np.ndarray):
    """
    Compute Wigner on a square grid defined by vector x (for q and p).

    Returns:
        W : 2D array, same shape as np.meshgrid(x, x).
    """
    return qutip_wigner(rho_or_psi, x, x)


# ============================================================
#   θ optimization for given Δ (q-comb states)
# ============================================================
def optimize_theta_for_delta(N, M, r, Delta,
                             theta0=np.sqrt(np.pi),
                             theta_span=0.8,
                             n_theta=41):
    """
    For a fixed (N, M, r, Delta), scan θ around theta0 to maximize
    logical F_avg for the SUM gate using q-comb GKP states.
    """
    psi0, psi1, _, _, _, _ = logical_eigenstates_qcomb(N, M, r, Delta)
    basis_kets = build_logical_basis(psi0, psi1)
    U_id = ideal_logical_cnot_matrix()

    theta_grid = np.linspace(theta0 - theta_span,
                             theta0 + theta_span,
                             n_theta)

    best_F = -1.0
    best_theta = theta0

    for th in theta_grid:
        U_sum = build_sum_gate(N, th)
        U_log = project_unitary_to_logical(U_sum, basis_kets)
        F_tmp = logical_average_gate_fidelity(U_log, U_id)
        if F_tmp > best_F:
            best_F = F_tmp
            best_theta = th

    print(
        f"[optimize_theta_for_delta] Δ={Delta:.3f} → "
        f"θ_opt={best_theta:.4f}, F_avg(SUM)={best_F:.4f}"
    )
    return best_theta, best_F


# ============================================================
#   Summary plot: per-basis state fidelities (used here once)
# ============================================================
def plot_state_fidelities(ops_labels, F_vals,
                          outdir="figs_truth_table_qcomb"):
    os.makedirs(outdir, exist_ok=True)

    x = np.arange(len(F_vals))

    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(x, F_vals, width=0.6, color="royalblue")

    ax.set_xticks(x)
    ax.set_xticklabels(ops_labels, rotation=20, ha="right")
    ax.set_ylabel(r"State fidelity $F_{\mathrm{state}}$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(r"GKP CNOT per-basis state fidelities", pad=20)
    ax.grid(False)

    for xi, bi, Fi in zip(x, bars, F_vals):
        ax.text(
            xi, bi.get_height() + 0.02,
            f"{Fi:.3f}",
            ha="center", va="bottom", fontsize=24
        )

    outpath = os.path.join(outdir, "state_fidelities_per_basis.png")
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-basis fidelity summary figure to {outpath}")


# ============================================================
#   Data saving helper
# ============================================================
def save_truth_table_data(
    N, M, r,
    Delta_main,
    theta0, theta_span, n_theta,
    theta_opt, F_avg_opt,
    op_labels, F_state_list,
    x_grid, Wigner_all,
    Delta_scan_list, overlap_Z_scan,
    data_dir="data_gkp_cnot_truth_table",
    fname="gkp_cnot_truth_table_qcomb.npz",
):
    """
    Save all relevant truth-table and Wigner data to a compressed .npz file.

    Args:
        Wigner_all: array with shape (4, 4, nX, nX)
            axis0: basis index  (0:00, 1:01, 2:10, 3:11)
            axis1: panel index  (0: control in, 1: target in,
                                 2: control out, 3: target out)
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, fname)

    op_labels_arr = np.array(op_labels, dtype=object)
    F_state_arr = np.array(F_state_list, dtype=float)
    Delta_scan_arr = np.array(Delta_scan_list, dtype=float)
    overlap_Z_arr = np.array(overlap_Z_scan, dtype=float)

    np.savez(
        path,
        N=int(N),
        M=int(M),
        r=float(r),
        Delta_main=float(Delta_main),
        theta0=float(theta0),
        theta_span=float(theta_span),
        n_theta=int(n_theta),
        theta_opt=float(theta_opt),
        F_avg_opt=float(F_avg_opt),
        op_labels=op_labels_arr,
        F_state=F_state_arr,
        x_grid=np.array(x_grid, dtype=float),
        Wigner_all=np.array(Wigner_all, dtype=float),
        Delta_scan=Delta_scan_arr,
        overlap_Z=overlap_Z_arr,
        panel_labels=np.array(
            ["control_in", "target_in", "control_out", "target_out"],
            dtype=object,
        ),
        basis_order=np.array(
            ["00", "01", "10", "11"],
            dtype=object,
        ),
    )
    print(f"\nSaved truth-table + Wigner + overlap data to {path}\n")

def squeezing_db_to_r(s_db: float) -> float:
    """
    Convert squeezing in dB (noise reduction in the squeezed quadrature)
    to the squeezing parameter r used in S(r).

    We take:
      Var_q = (1/2) e^{-2r}
      s_dB = -10 log10(Var_q / (1/2)) = 20 r / ln(10)
    so:
      r = s_dB * ln(10) / 20.
    """
    return float(s_db * np.log(10.0) / 20.0)

# ============================================================
#   Main
# ============================================================
def main():
    # --- configuration (edit to taste) ---
    N = 70
    M = 5
    r = squeezing_db_to_r(12.0)

    # main Δ used for Wigner truth-table
    Delta_main = 0.3   # Δ = linewidth of comb envelope in q

    # Δ scan for overlap |<0_L|1_L>|^2 vs Δ
    Delta_scan_list = np.linspace(0.18, 0.32, 9)

    theta0 = np.sqrt(np.pi)
    theta_span = 0.8
    n_theta = 41

    outdir = "figs_truth_table_qcomb"

    print("=== GKP CNOT truth table (q-comb, NR SUM) ===")
    print(f"N={N}, M={M}, r={r}")
    print(f"Delta_main  = {Delta_main:.3f}")
    print(f"Delta_scan_list = {[f'{d:.3f}' for d in Delta_scan_list]}")
    print(f"θ search: center≈√π={theta0:.3f}, span={theta_span}, n={n_theta}")
    print()

    # ---- overlap vs Δ scan ----
    overlap_Z_scan = []
    for D in Delta_scan_list:
        psi0, psi1, _, _, _, _ = logical_eigenstates_qcomb(N, M, r, D)
        ov = ket_fidelity(psi0, psi1)
        overlap_Z_scan.append(ov)
        print(f"[overlap scan] Δ={D:.3f} → |<0_L|1_L>|^2 = {ov:.3e}")
    print()

    # ---- optimize θ once for (N, Delta_main) ----
    theta_opt, F_opt = optimize_theta_for_delta(
        N=N, M=M, r=r, Delta=Delta_main,
        theta0=theta0,
        theta_span=theta_span,
        n_theta=n_theta
    )
    print(f"Using θ_SUM,opt={theta_opt:.4f} with F_avg={F_opt:.4f}\n")

    # ---- per-basis state fidelities (ideal CNOT action) ----
    psi0_main, psi1_main, _, _, _, _ = logical_eigenstates_qcomb(
        N, M, r, Delta_main
    )
    basis_kets = build_logical_basis(psi0_main, psi1_main)
    U_sum_opt = build_sum_gate(N, theta_opt)

    # index convention: [|00>, |01>, |10>, |11>]
    basis_index = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    # CNOT mapping in this basis
    ideal_index = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (1, 0): (1, 1),
        (1, 1): (1, 0),
    }

    print("Per-basis state fidelities (CNOT action on logical q-comb GKP states):")
    F_state_list = []
    op_labels = []

    for c_label in (0, 1):
        for t_label in (0, 1):
            idx_in = basis_index[(c_label, t_label)]
            idx_out = basis_index[ideal_index[(c_label, t_label)]]

            psi_in = basis_kets[idx_in]
            psi_ideal = basis_kets[idx_out]
            psi_out = U_sum_opt * psi_in

            F_state = ket_fidelity(psi_out, psi_ideal)
            F_state_list.append(F_state)

            c_out, t_out = ideal_index[(c_label, t_label)]
            op_label = rf"$|{c_label}_L,{t_label}_L\rangle \to |{c_out}_L,{t_out}_L\rangle$"
            op_labels.append(op_label)

            print(
                f"  |{c_label}_L,{t_label}_L> → "
                f"ideal |{c_out}_L,{t_out}_L>, "
                f"F_state = {F_state:.4f}"
            )
    print()

    # --- compute Wigner data for all basis states and panels ---
    xlim = 5.0
    grid_pts = 201
    x_grid = np.linspace(-xlim, xlim, grid_pts)

    # Wigner_all[basis_idx, panel_idx, i, j]
    # basis_idx: 0:00, 1:01, 2:10, 3:11
    # panel_idx: 0:control_in, 1:target_in, 2:control_out, 3:target_out
    Wigner_all = np.zeros((4, 4, grid_pts, grid_pts), dtype=float)

    print("Computing Wigner data for all basis states...")
    for (c_label, t_label), idx_b in basis_index.items():
        psi_in = basis_kets[idx_b]
        psi_out = U_sum_opt * psi_in

        rho_in = psi_in.proj()
        rho_out = psi_out.proj()

        rho_in_c = ptrace(rho_in, 0)
        rho_in_t = ptrace(rho_in, 1)
        rho_out_c = ptrace(rho_out, 0)
        rho_out_t = ptrace(rho_out, 1)

        W_in_c = single_mode_wigner(rho_in_c, x_grid)
        W_in_t = single_mode_wigner(rho_in_t, x_grid)
        W_out_c = single_mode_wigner(rho_out_c, x_grid)
        W_out_t = single_mode_wigner(rho_out_t, x_grid)

        Wigner_all[idx_b, 0, :, :] = W_in_c
        Wigner_all[idx_b, 1, :, :] = W_in_t
        Wigner_all[idx_b, 2, :, :] = W_out_c
        Wigner_all[idx_b, 3, :, :] = W_out_t

    print("Done computing Wigner data.\n")

    # --- save metrics + Wigner data ---
    save_truth_table_data(
        N, M, r,
        Delta_main,
        theta0, theta_span, n_theta,
        theta_opt, F_opt,
        op_labels, F_state_list,
        x_grid, Wigner_all,
        Delta_scan_list, overlap_Z_scan,
        data_dir="data_gkp_cnot_truth_table",
        fname="gkp_cnot_truth_table_qcomb.npz",
    )

    # --- quick summary plots directly from current run (optional) ---
    # 1) bar plot (same as what the from-data script can regenerate)
    plot_state_fidelities(op_labels, F_state_list, outdir=outdir)

    # 2) overlap vs Δ
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Delta_scan_list, overlap_Z_scan, "o-", lw=2)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$|\langle 0_L|1_L\rangle|^2$")
    ax.set_title(r"Logical overlap vs comb linewidth $\Delta$")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    overlap_path = os.path.join(outdir, "overlap_vs_Delta.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(overlap_path, dpi=220)
    plt.close(fig)
    print(f"Saved overlap-vs-Delta figure to {overlap_path}")

    # 3) Wigner truth-table panels, now using Wigner_all we just computed
    #    (this is just for convenience so you don't need the from-data script
    #     to see them the first time).
    def plot_wigner_panel_from_arrays(
        c_label, t_label, basis_idx,
        x_grid, W_all_basis,
        outdir="figs_truth_table_qcomb"
    ):
        os.makedirs(outdir, exist_ok=True)
        X, P = np.meshgrid(x_grid, x_grid)

        panels = ["Init control", "Init target",
                  "Final control (SUM)", "Final target (SUM)"]
        fig, axes = plt.subplots(
            2, 2, figsize=(20, 14), constrained_layout=False
        )
        fig.subplots_adjust(top=0.78, hspace=0.55, wspace=0.35)

        all_W = W_all_basis
        vmax = float(np.max(np.abs(all_W)))
        vmin = -vmax

        def panel(ax, W, title):
            cf = ax.contourf(
                X, P, W, 220, cmap="bwr",
                vmin=vmin, vmax=vmax
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_grid[0], x_grid[-1])
            ax.set_ylim(x_grid[0], x_grid[-1])
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.set_title(title, pad=12)
            ax.tick_params(labelsize=26)

            ticks = np.linspace(vmin, vmax, 7)
            cb = fig.colorbar(
                cf, ax=ax,
                fraction=0.046, pad=0.03,
                ticks=ticks
            )
            cb.ax.set_ylabel(r"$W(q,p)$", fontsize=26)
            cb.ax.tick_params(labelsize=26)
            if abs(vmax) > 0.01:
                fmt = "%.2f"
            else:
                fmt = "%.3f"
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

        panel(axes[0, 0], all_W[0], panels[0])
        panel(axes[0, 1], all_W[1], panels[1])
        panel(axes[1, 0], all_W[2], panels[2])
        panel(axes[1, 1], all_W[3], panels[3])

        label_math = rf"$|{c_label}_L,{t_label}_L\rangle$"
        fig.suptitle(
            rf"GKP CNOT truth-table snapshot ({label_math}), "
            rf"$N={N}$, $\Delta={Delta_main:.3f}$, "
            rf"$\theta_\mathrm{{SUM}}={theta_opt:.3f}$",
            y=0.97
        )

        outname = (
            f"truth_table_ctrl{c_label}_tgt{t_label}_"
            f"N{N}_Delta{Delta_main:.3f}.png"
        )
        outpath = os.path.join(outdir, outname)
        fig.savefig(outpath, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved truth-table panel to {outpath}")

    for (c_label, t_label), idx_b in basis_index.items():
        plot_wigner_panel_from_arrays(
            c_label, t_label, idx_b,
            x_grid, Wigner_all[idx_b],
            outdir=outdir
        )

    print("\nDone.")
    print(f"Truth-table panels and fidelity summary saved in ./{outdir}/")
    print("Truth-table metrics + Wigner + overlap data saved in ./data_gkp_cnot_truth_table/")


if __name__ == "__main__":
    main()
