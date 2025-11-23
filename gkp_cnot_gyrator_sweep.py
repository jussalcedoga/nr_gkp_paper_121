#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gkp_cnot_gyrator_sweep.py

Scan the logical CNOT performance for q-comb GKP states as a function of

    • Fock cutoff N,
    • gyrator/SUM parameter θ (dimensionless angle),
    • comb parameter Δ,

for a fixed squeezing value (in dB) used in the single-mode GKP preparation.

We use logical_eigenstates_qcomb(N, M, r, Delta) from
gkp_qcomb_stateprep_panels.py, where r is the squeezing parameter of S(r).

For each grid point (Δ, N, θ) we compute:

    • U_log: 4×4 logical gate matrix in the basis
        {|0_L,0_L>, |0_L,1_L>, |1_L,0_L>, |1_L,1_L>}

    • F_avg(Δ, N, θ): average logical gate fidelity wrt ideal CNOT

    • F_state[Δ, N, θ, k]: per-basis state fidelity for the 4 transitions
        k = 0: |0_L,0_L> → |0_L,0_L>
        k = 1: |0_L,1_L> → |0_L,1_L>
        k = 2: |1_L,0_L> → |1_L,1_L>
        k = 3: |1_L,1_L> → |1_L,0_L>

Outputs
-------

Data:
  data_gkp_cnot_gyrator_sweep/gkp_cnot_gyrator_sweep.npz

    Contains:
      N_list        (nN,)
      theta_list    (nT,)
      Delta_list    (nD,)
      M             (scalar, int)
      squeeze_dB    (scalar, float)
      r_squeeze     (scalar, float)
      F_avg         (nD, nN, nT)
      F_state       (nD, nN, nT, 4)
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

from qutip import Qobj, destroy, qeye, tensor

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


def build_qp_single(N: int):
    """
    Build single-mode q, p operators for a Hilbert space of dimension N.

    We choose:
        q = (a + a^†)/√2
        p = (a - a^†)/(i√2)

    so that [q, p] = i.

    IMPORTANT: we use destroy(N) so that the operator dimension matches
    the states returned by logical_eigenstates_qcomb(N, ...).
    """
    a = destroy(N)  # <-- dimension N, not N+1
    q = (a + a.dag()) / np.sqrt(2.0)
    p = (a - a.dag()) / (1j * np.sqrt(2.0))
    return q, p


def compute_logical_cnot_metrics(N: int,
                                 M: int,
                                 r: float,
                                 Delta: float,
                                 theta: float):
    """
    For given N, M, r, Delta, and θ, construct single-mode q-comb GKP states,
    build the 2-mode SUM gate U = exp(-i θ q1 p2), and compute:

      - F_avg : average logical gate fidelity wrt ideal CNOT
      - F_state[4] : per-basis state fidelities for the 4 CNOT transitions.

    Returns:
      F_avg (float), F_state (np.ndarray, shape (4,))
    """
    # Single-mode logical eigenstates (q-comb GKP)
    psi0, psi1, _, _, _, _ = logical_eigenstates_qcomb(N, M, r, Delta)
    # At this point psi0, psi1 live in Hilbert space of dimension N.

    # Two-mode logical basis states: {|00>, |01>, |10>, |11>}
    basis_logical = [
        tensor(psi0, psi0),  # |0_L,0_L>
        tensor(psi0, psi1),  # |0_L,1_L>
        tensor(psi1, psi0),  # |1_L,0_L>
        tensor(psi1, psi1),  # |1_L,1_L>
    ]

    # Ideal CNOT action on basis indices: 0→0, 1→1, 2→3, 3→2
    cnot_perm = [0, 1, 3, 2]

    # Build two-mode q1, p2 operators
    q_single, p_single = build_qp_single(N)  # dim N
    I_single = qeye(N)
    q1 = tensor(q_single, I_single)
    p2 = tensor(I_single, p_single)

    # SUM Hamiltonian: H = q1 p2  (we absorb coupling*duration into θ)
    H_sum = q1 * p2

    # Unitary: U = exp(-i θ H_sum)
    U = (-1j * theta * H_sum).expm()

    # ------------------ Per-basis state fidelities ------------------
    F_state = np.zeros(4, dtype=float)

    # For constructing U_log (4×4 logical gate matrix)
    U_log = np.zeros((4, 4), dtype=complex)

    for j, ket_in in enumerate(basis_logical):
        ket_out = U * ket_in

        # Ideal output state according to CNOT mapping
        ideal_idx = cnot_perm[j]
        ket_ideal = basis_logical[ideal_idx]

        # Per-basis state fidelity
        F_state[j] = float(abs(ket_ideal.overlap(ket_out)) ** 2)

        # Column j of logical gate matrix: overlaps with logical basis
        for i, ket_basis in enumerate(basis_logical):
            U_log[i, j] = ket_basis.overlap(ket_out)

    # ------------------ Average gate fidelity ------------------
    d = 4  # dimension of logical subspace (two qubits)
    # Ideal CNOT in computational basis {|00>,|01>,|10>,|11>}:
    # matrix with ones on (0,0), (1,1), (2,3), (3,2)
    U_id = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)

    tr_term = np.trace(U_id.conj().T @ U_log)
    F_avg = (abs(tr_term) ** 2 + d) / (d * (d + 1))

    return float(F_avg), F_state


def _worker(task):
    """
    Worker for ProcessPoolExecutor.

    Input:
      (iD, iN, iT, N, theta, Delta, M, r)

    Output:
      (iD, iN, iT, F_avg, F_state)
    """
    iD, iN, iT, N, theta, Delta, M, r = task
    try:
        F_avg, F_state = compute_logical_cnot_metrics(N, M, r, Delta, theta)
    except Exception as e:
        print(
            f"Warning: failed at Δ={Delta:.3f}, N={N}, "
            f"θ={theta:.3f}: {e}"
        )
        F_avg = np.nan
        F_state = np.full(4, np.nan, dtype=float)
    return iD, iN, iT, F_avg, F_state


def main():
    # ============================================================
    #   Configuration
    # ============================================================

    # q-comb "teeth" parameter (as in your state-prep script)
    M = 5

    # Fixed squeezing (in dB) used to prepare the single-mode GKP states.
    squeeze_dB = 12.0
    r_squeeze = squeezing_db_to_r(squeeze_dB)

    # Fock cutoff N range (dimension of single-mode Hilbert space)
    # N_list = list(range(10, 41, 5))  # 10, 12, ..., 70
    N_list = list(range(10, 71, 2))

    # Gyrator / SUM parameter θ (dimensionless, θ = g * t).
    # θ ~ sqrt(pi) is the CNOT point in our model.
    theta_min = 0.5
    theta_max = 2.5
    n_theta = len(N_list)  # same number of points as N_list, for convenience
    theta_list = np.linspace(theta_min, theta_max, n_theta)

    # Comb envelope Δ: allow multiple values if desired
    Delta_list = np.array([0.3], dtype=float)
    n_D = len(Delta_list)

    data_dir = "data_gkp_cnot_gyrator_sweep"
    os.makedirs(data_dir, exist_ok=True)

    print("=== GKP CNOT gyrator scan (θ sweep) ===")
    print(f"M               = {M}")
    print(f"squeeze_dB      = {squeeze_dB:.2f} dB  (r = {r_squeeze:.3f})")
    print(f"N_list          = {N_list}")
    print(f"theta_list      = {[f'{t:.3f}' for t in theta_list]}")
    print(f"Delta_list      = {[f'{d:.3f}' for d in Delta_list]}")
    print()

    # ============================================================
    #   Prepare tasks for parallel execution
    # ============================================================

    tasks = []
    for iD, Delta in enumerate(Delta_list):
        for iN, N in enumerate(N_list):
            for iT, theta in enumerate(theta_list):
                tasks.append(
                    (iD, iN, iT, int(N), float(theta),
                     float(Delta), int(M), float(r_squeeze))
                )

    total_tasks = len(tasks)
    print(f"Total grid points to evaluate: {total_tasks}")

    # F_avg[iD, iN, iT]     = average gate fidelity
    # F_state[iD, iN, iT, k] = per-basis fidelity for 4 transitions
    F_avg = np.zeros((n_D, len(N_list), len(theta_list)), dtype=float)
    F_state = np.zeros((n_D, len(N_list), len(theta_list), 4), dtype=float)

    # ============================================================
    #   Parallel computation
    # ============================================================
    with ProcessPoolExecutor(max_workers=80) as pool:
        for (iD, iN, iT, Fav, Fst) in tqdm(
            pool.map(_worker, tasks),
            total=total_tasks,
            desc="CNOT grid"
        ):
            F_avg[iD, iN, iT] = Fav
            F_state[iD, iN, iT, :] = Fst

    print("\nDone computing CNOT fidelity grid.\n")

    # ============================================================
    #   Save data
    # ============================================================
    out_path = os.path.join(
        data_dir, "gkp_cnot_gyrator_sweep.npz"
    )
    np.savez(
        out_path,
        N_list=np.array(N_list, dtype=int),
        theta_list=np.array(theta_list, dtype=float),
        Delta_list=np.array(Delta_list, dtype=float),
        M=int(M),
        squeeze_dB=float(squeeze_dB),
        r_squeeze=float(r_squeeze),
        F_avg=F_avg,
        F_state=F_state,
    )
    print(f"Saved CNOT fidelity data to {out_path}\n")


if __name__ == "__main__":
    main()
