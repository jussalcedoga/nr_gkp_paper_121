#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gkp_overlap_qcomb_sweep_squeezing.py

Scan the logical overlap |<0_L|1_L>|^2 for q-comb GKP states as a function of

    • Fock cutoff N,
    • squeezing in dB (s_dB),
    • comb parameter Δ (as in gkp_qcomb_stateprep_panels.py).

We use logical_eigenstates_qcomb(N, M, r, Delta) from
gkp_qcomb_stateprep_panels.py, where r is the squeezing parameter of S(r).
Here we scan s_dB and convert to r via

    s_dB = 20 r / ln(10)   ⇒   r = s_dB * ln(10) / 20.

Outputs
-------

Data:
  data_gkp_overlap_qcomb/gkp_overlap_qcomb_sweep_squeezing.npz

    Contains:
      N_list        (nN,)
      squeeze_dB_list (nS,)
      Delta_list    (nD,)
      overlap       (nD, nN, nS)
        overlap[iD, iN, iS] = |<0_L|1_L>|^2
      M             (scalar, int)

Figures:
  figs_gkp_overlap_qcomb_sweep_squeezing/overlap_vs_N_squeezing_Delta_#.png

One heatmap per Δ, with axes:
  x-axis: squeezing (dB)
  y-axis: N
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

from qutip import Qobj

# Import q-comb logical states (Delta, r, etc. as defined there)
from gkp_qcomb_stateprep_panels import logical_eigenstates_qcomb

# ────────── GLOBAL STYLE ──────────
plt.rcParams.update({
    "font.size": 16,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.usetex": False,
})
mpl.rcParams["axes.unicode_minus"] = False


def ket_fidelity(psi: Qobj, phi: Qobj) -> float:
    """Return |<psi|phi>|^2."""
    ov = psi.overlap(phi)
    return float(abs(ov) ** 2)


def build_integer_list(n_min: int, n_max: int, n_points: int):
    """
    Helper to build a sorted list of integers between [n_min, n_max]
    with ~n_points (approximately evenly spaced) entries.
    """
    arr = np.linspace(n_min, n_max, n_points)
    ints = sorted(set(int(round(x)) for x in arr))
    return ints


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


def _worker_overlap(task):
    """
    Worker for ProcessPoolExecutor.

    Input:
      (iD, iN, iS, N, s_db, Delta, M)

    Output:
      (iD, iN, iS, overlap_val)
    """
    iD, iN, iS, N, s_db, Delta, M = task
    try:
        r = squeezing_db_to_r(s_db)
        psi0, psi1, _, _, _, _ = logical_eigenstates_qcomb(N, M, r, Delta)
        ov = ket_fidelity(psi0, psi1)
    except Exception as e:
        print(
            f"Warning: failed at Δ={Delta:.3f}, N={N}, "
            f"s_dB={s_db:.2f}: {e}"
        )
        ov = np.nan
    return iD, iN, iS, ov


def main():
    # ============================================================
    #   Configuration
    # ============================================================

    # q-comb "teeth" parameter (as in your state-prep script)
    M = 5

    # Fock cutoff N range (exactly n_N integer points)
    N_list = list(range(10, 71, 2))

    # Squeezing in dB (noise reduction in the squeezed quadrature)
    # e.g. from 0 dB (no squeezing) up to ~12 dB.
    s_db_min = 0.0
    s_db_max = 12.0
    n_s      = len(N_list)
    squeeze_dB_list = np.linspace(s_db_min, s_db_max, n_s)

    # Comb envelope Δ: choose three representative values
    # (e.g. relatively broad, intermediate, and narrow envelope)
    Delta_list = np.array([0.2, 0.3], dtype=float)
    n_D = len(Delta_list)

    data_dir = "data_gkp_overlap_qcomb"
    figs_dir = "figs_gkp_overlap_qcomb_sweep_squeezing"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    print("=== GKP q-comb overlap scan (squeezing sweep) ===")
    print(f"N_list          = {N_list}")
    print(f"squeeze_dB_list = {[f'{s:.2f}' for s in squeeze_dB_list]} dB")
    print(f"Delta_list      = {[f'{d:.3f}' for d in Delta_list]}")
    print()

    # ============================================================
    #   Prepare tasks for parallel execution
    # ============================================================

    tasks = []
    for iD, Delta in enumerate(Delta_list):
        for iN, N in enumerate(N_list):
            for iS, s_db in enumerate(squeeze_dB_list):
                tasks.append((iD, iN, iS, int(N), float(s_db), float(Delta), int(M)))

    total_tasks = len(tasks)
    print(f"Total grid points to evaluate: {total_tasks}")

    # overlap[iD, iN, iS] = |<0_L|1_L>|^2
    overlap = np.zeros((n_D, len(N_list), len(squeeze_dB_list)), dtype=float)

    # ============================================================
    #   Parallel computation
    # ============================================================
    with ProcessPoolExecutor() as pool:
        for (iD, iN, iS, ov) in tqdm(
            pool.map(_worker_overlap, tasks),
            total=total_tasks,
            desc="Overlap grid"
        ):
            overlap[iD, iN, iS] = ov

    print("\nDone computing overlap grid.\n")

    # ============================================================
    #   Save data
    # ============================================================
    out_path = os.path.join(
        data_dir, "gkp_overlap_qcomb_sweep_squeezing.npz"
    )
    np.savez(
        out_path,
        N_list=np.array(N_list, dtype=int),
        squeeze_dB_list=np.array(squeeze_dB_list, dtype=float),
        Delta_list=np.array(Delta_list, dtype=float),
        overlap=overlap,
        M=int(M),
    )
    print(f"Saved overlap data to {out_path}\n")

    # ============================================================
    #   Quick plots: one heatmap per Δ
    # ============================================================
    N_arr = np.array(N_list, dtype=float)
    s_arr = np.array(squeeze_dB_list, dtype=float)

    finite_vals = overlap[np.isfinite(overlap)]
    if finite_vals.size > 0:
        vmin = 0.0
        vmax = float(np.nanmax(finite_vals))
        vmax = max(vmax, 1e-6)
        vmax = min(vmax, 1.0)
    else:
        vmin, vmax = 0.0, 1.0

    for iD, Delta in enumerate(Delta_list):
        data_D = overlap[iD]  # shape: (nN, nS)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            data_D,
            origin="lower",
            aspect="auto",
            extent=[s_arr[0], s_arr[-1], N_arr[0], N_arr[-1]],
            cmap="bwr",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("squeezing (dB)")
        ax.set_ylabel(r"$N$")
        ax.set_title(
            rf"$|\langle 0_L|1_L\rangle|^2$  (Δ = {Delta:.3f})"
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|\langle 0_L|1_L\rangle|^2$")

        fig.tight_layout()
        fig_name = f"overlap_vs_N_squeezing_Delta_{iD}_Delta{Delta:.3f}.png"
        fig_path = os.path.join(figs_dir, fig_name)
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)
        print(f"Saved heatmap for Δ={Delta:.3f} to {fig_path}")

    print("\nAll figures saved in ./figs_gkp_overlap_qcomb_sweep_squeezing/")
    print("You can now see how overlap behaves versus squeezing (dB), N, and Δ.\n")


if __name__ == "__main__":
    main()
