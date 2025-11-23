#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gkp_qcomb_stateprep_panels.py

Diagnostics for finite-energy GKP logical eigenstates using a 1D q-comb
construction in the Fock basis (no explicit p-lattice).

This version:
- Uses Helvetica for all plot text.
- Enforces identical axis limits and square aspect ratio in all panels.
- Labels colorbars as W(q,p).
- Parallelizes the Delta sweep with ProcessPoolExecutor.
- Uses Delta as a *squeezing-like* envelope parameter in q:

      w_s ∝ exp[ - (Δ^2 / 2) q_s^2 ],

  so larger Δ → narrower envelope in q (more energy, smaller effective
  linewidth σ_q ~ 1/Δ). This is the parametrization that produced the
  "good" combs in your original plots.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

from qutip import (
    basis, destroy, displace, squeeze, expect,
    Qobj, wigner as qutip_wigner
)

import matplotlib

# ----------------- Matplotlib aesthetics -----------------
matplotlib.use('Agg')  # Use the 'Agg' backend for PNG output

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['text.usetex'] = True

sqrt_pi = np.sqrt(np.pi)

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

# ----------------- Global config -----------------
CFG = dict(
    N=80,                       # Fock cutoff
    M=5,                        # number of teeth on each side (s in [-M..M])
    r=squeezing_db_to_r(12.0),                      # squeezing of seed S(r)|0>, r>0 squeezes q
    # Delta is a *squeezing-like* parameter for the comb envelope in q:
    #   w_s ∝ exp[- (Δ^2 / 2) q_s^2]
    # Larger Δ → narrower envelope in q (higher energy, more orthogonality).
    
    # Delta_list=[0.1, 0.2, 0.3, 0.4],
    Delta_list = [0.3],
    xlim=5.0,                   # Wigner window in q and p
    grid_pts=201,               # Wigner resolution
    cmap="bwr",
)

# ============================================================
#   1D q-comb GKP codewords in Fock basis
# ============================================================
def gkp_qcomb_state(N, which="0", r=0.8, M=5, Delta=0.25, amp_cut=1e-12):
    """
    Finite-energy GKP |0_L> or |1_L> built from a *1D q-comb* in Fock space.

      which='0' -> |0_L>: peaks at q = 2s * sqrt(pi)
      which='1' -> |1_L>: peaks at q = (2s+1) * sqrt(pi)

    Construction:
        psi ~ sum_s w_s D(q_s) S(r) |0>,  q_s = (2s+offset)*sqrt(pi)

    with weights (Gaussian envelope in q):
        w_s = exp[ - (Δ^2 / 2) q_s^2 ],

    so Δ controls an *inverse* linewidth in q:
        larger Δ → narrower envelope (more energy, smaller σ_q ~ 1/Δ).
    """
    vac = basis(N, 0)
    S   = squeeze(N, r) * vac  # squeezed seed

    psi = 0 * vac
    offset = 0 if which == "0" else 1

    for s in range(-M, M + 1):
        q_s = (2 * s + offset) * sqrt_pi

        # Gaussian envelope in q-space, Delta as "squeezing-like"
        w_s = np.exp(-0.5 * (Delta ** 2) * (q_s ** 2))
        if w_s < amp_cut:
            continue

        # Purely real displacement → q shift
        alpha = q_s / np.sqrt(2.0)
        psi += w_s * (displace(N, alpha) * S)

    if psi.norm() == 0:
        return vac
    return psi.unit()


def logical_eigenstates_qcomb(N, M, r, Delta):
    """
    Build the six logical eigenstates using the q-comb construction:
      |0_L>, |1_L>, |+_X>, |-_X>, |+_Y>, |-_Y>.
    """
    psi0 = gkp_qcomb_state(N, "0", r=r, M=M, Delta=Delta)
    psi1 = gkp_qcomb_state(N, "1", r=r, M=M, Delta=Delta)

    Xp = (psi0 + psi1).unit()
    Xm = (psi0 - psi1).unit()
    Yp = (psi0 + 1j * psi1).unit()
    Ym = (psi0 - 1j * psi1).unit()
    return psi0, psi1, Xp, Xm, Yp, Ym


# ----------------- Fidelity helper -----------------
def ket_fidelity(psi: Qobj, phi: Qobj) -> float:
    ov = psi.overlap(phi)
    return float(abs(ov) ** 2)


# ============================================================
#   Parallel worker: metrics vs Delta
# ============================================================
def delta_metrics_task(args):
    """
    Worker for ProcessPoolExecutor:
      input : (N, M, r, Delta)
      output: (Delta, |<0_L|1_L>|^2, |<+_X|-_X>|^2, |<+_Y|-_Y>|^2, <n>_0, <n>_1)
    """
    N, M, r, Delta = args
    psi0, psi1, Xp, Xm, Yp, Ym = logical_eigenstates_qcomb(N, M, r, Delta)
    ov_Z = ket_fidelity(psi0, psi1)
    ov_X = ket_fidelity(Xp, Xm)
    ov_Y = ket_fidelity(Yp, Ym)

    n_op = destroy(N).dag() * destroy(N)
    n0 = float(expect(n_op, psi0).real)
    n1 = float(expect(n_op, psi1).real)
    return Delta, ov_Z, ov_X, ov_Y, n0, n1


# ============================================================
#   Wigner helpers & panels
# ============================================================
def single_mode_wigner(psi: Qobj, xlim, grid_pts):
    x = np.linspace(-xlim, xlim, grid_pts)
    W = qutip_wigner(psi, x, x)
    X, P = np.meshgrid(x, x)
    vmax = np.max(np.abs(W))
    vmin = -vmax
    return X, P, W, vmin, vmax


def plot_panel_two_rows(states_by_Delta, which, cfg, outpath):
    """
    Generic 2×NΔ panel for a given axis (Z, X, or Y).

    states_by_Delta: dict[Delta] -> (psi_top, psi_bottom)
    which          : 'Z', 'X', or 'Y'
    """
    Delta_list = sorted(states_by_Delta.keys())
    nD = len(Delta_list)
    xlim = cfg["xlim"]
    grid_pts = cfg["grid_pts"]
    cmap = cfg["cmap"]

    # Precompute Wigners & global color scale
    W_data = {}
    global_vmax = 0.0
    for D in Delta_list:
        psi_top, psi_bot = states_by_Delta[D]
        Xg, Pg, Wt, _, _ = single_mode_wigner(psi_top, xlim, grid_pts)
        _, _, Wb, _, _ = single_mode_wigner(psi_bot, xlim, grid_pts)
        W_data[D] = (Xg, Pg, Wt, Wb)
        global_vmax = max(global_vmax,
                          np.max(np.abs(Wt)),
                          np.max(np.abs(Wb)))
    vmin, vmax = -global_vmax, global_vmax

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig, axes = plt.subplots(2, nD, figsize=(3.6 * nD, 6.2), constrained_layout=True)

    for j, D in enumerate(Delta_list):
        Xg, Pg, Wt, Wb = W_data[D]

        # --- top row ---
        ax_top = axes[0, j] if nD > 1 else axes[0]
        cf = ax_top.contourf(Xg, Pg, Wt, 220, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.set_xlim(-xlim, xlim)
        ax_top.set_ylim(-xlim, xlim)
        ax_top.set_xlabel(r"$q$")
        ax_top.set_ylabel(r"$p$")
        ax_top.grid(False)

        # in-panel label (lower-left corner)
        if which == "Z":
            label_top = r"$|0_L\rangle$, $\Delta={:.1f}$".format(D)
        elif which == "X":
            label_top = r"$|+_X\rangle$, $\Delta={:.1f}$".format(D)
        else:
            label_top = r"$|+_Y\rangle$, $\Delta={:.1f}$".format(D)
        ax_top.text(
            0.03, 0.04, label_top,
            transform=ax_top.transAxes,
            ha="left", va="bottom",
            fontsize=11
        )

        # Colorbar with ~5 ticks and larger labels
        ticks = np.linspace(vmin, vmax, 3)
        cb = fig.colorbar(cf, ax=ax_top, fraction=0.046, pad=0.03, ticks=ticks)
        cb.ax.tick_params(labelsize=15)
        cb.set_label(r"$W(q,p)$", fontsize=15)

        fmt = r"$%.2f$" if abs(vmax) > 0.1 else r"$%.3f$"
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))

        # --- bottom row ---
        ax_bot = axes[1, j] if nD > 1 else axes[1]
        cf = ax_bot.contourf(Xg, Pg, Wb, 220, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_bot.set_aspect("equal", adjustable="box")
        ax_bot.set_xlim(-xlim, xlim)
        ax_bot.set_ylim(-xlim, xlim)
        ax_bot.set_xlabel(r"$q$")
        ax_bot.set_ylabel(r"$p$")
        ax_bot.grid(False)

        if which == "Z":
            label_bot = r"$|1_L\rangle$, $\Delta={:.1f}$".format(D)
        elif which == "X":
            label_bot = r"$|-_X\rangle$, $\Delta={:.1f}$".format(D)
        else:
            label_bot = r"$|-_Y\rangle$, $\Delta={:.1f}$".format(D)
        ax_bot.text(
            0.03, 0.04, label_bot,
            transform=ax_bot.transAxes,
            ha="left", va="bottom",
            fontsize=11
        )

        ticks = np.linspace(vmin, vmax, 3)
        cb = fig.colorbar(cf, ax=ax_bot, fraction=0.046, pad=0.03, ticks=ticks)
        cb.ax.tick_params(labelsize=15)
        cb.set_label(r"$W(q,p)$", fontsize=15)

        fmt = r"$%.2f$" if abs(vmax) > 0.1 else r"$%.3f$"
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))

        # √π guides
        s = np.sqrt(np.pi)
        for ax in (ax_top, ax_bot):
            for m in range(-3, 4):
                ax.axvline(m * s, lw=0.5, ls=":", color="k", alpha=0.3)
                ax.axhline(m * s, lw=0.5, ls=":", color="k", alpha=0.3)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved panel: {outpath}")


# ============================================================
#   Main
# ============================================================
def main():
    N = CFG["N"]
    M = CFG["M"]
    r = CFG["r"]
    D_list = CFG["Delta_list"]

    print("=== GKP logical eigenstate diagnostics (q-comb construction) ===")
    print(f"N={N}, M={M}, r={r:.2f}")
    print("Delta (squeezing-like) list: " + ", ".join(f"{d:.3f}" for d in D_list) + "\n")

    # ---- parallel metrics vs Delta ----
    tasks = [(N, M, r, D) for D in D_list]
    print("Computing metrics vs Delta (parallelized over Delta):")
    with ProcessPoolExecutor() as pool:
        results = list(tqdm(pool.map(delta_metrics_task, tasks),
                            total=len(tasks), desc="Delta sweep"))

    results.sort(key=lambda x: x[0])

    print("\nSummary metrics:")
    print("  Delta   |<0_L|1_L>|^2   |<+_X|-_X>|^2   |<+_Y|-_Y>|^2      <n>_0      <n>_1")
    for D, ovZ, ovX, ovY, n0, n1 in results:
        print(f"  {D:5.3f}   {ovZ:10.3e}   {ovX:10.3e}   {ovY:10.3e}   {n0:8.2f}   {n1:8.2f}")
    print("\nLarger Delta → sharper peaks & lower overlaps, but larger photon number.")
    print("Pick a Delta where the overlaps are small enough AND <n> is compatible")
    print("with your chosen N for the later CNOT/SUM simulations.\n")

    # ---- build states for plotting (sequential; relatively cheap) ----
    Z_states_by_Delta = {}
    X_states_by_Delta = {}
    Y_states_by_Delta = {}

    for D in D_list:
        psi0, psi1, Xp, Xm, Yp, Ym = logical_eigenstates_qcomb(N, M, r, D)
        Z_states_by_Delta[D] = (psi0, psi1)
        X_states_by_Delta[D] = (Xp, Xm)
        Y_states_by_Delta[D] = (Yp, Ym)

    outdir = "figs_gkp_diag_qcomb_squeezing_dB"
    os.makedirs(outdir, exist_ok=True)

    plot_panel_two_rows(
        Z_states_by_Delta, which="Z", cfg=CFG,
        outpath=os.path.join(outdir, "gkp_Z_vs_delta_panel_qcomb.png")
    )
    plot_panel_two_rows(
        X_states_by_Delta, which="X", cfg=CFG,
        outpath=os.path.join(outdir, "gkp_X_vs_delta_panel_qcomb.png")
    )
    plot_panel_two_rows(
        Y_states_by_Delta, which="Y", cfg=CFG,
        outpath=os.path.join(outdir, "gkp_Y_vs_delta_panel_qcomb.png")
    )

    print("\nDone.")
    print(f"Panels saved in '{outdir}/'.")
    print("This uses Delta as a squeezing-like envelope parameter in q; "
          "the effective q-linewidth scales as σ_q ~ 1/Δ.\n")


if __name__ == "__main__":
    main()
