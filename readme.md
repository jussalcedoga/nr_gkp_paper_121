# Nonreciprocal GKP gates in circuit QED

This repository contains the numerical code and plotting scripts used to generate the results and figures for the manuscript

> **Hardware-Efficient Universal Gate Set for GKP Qubits in Circuit QED**  
> Juan Sebastián Salcedo-Gallo, *et al.*

The code constructs finite-energy GKP codewords using a 1D \(q\)-comb in a truncated Fock basis, evaluates logical overlaps as a function of squeezing and cutoff, and simulates a gyrator-based SUM gate that acts as a logical CNOT between two GKP-encoded oscillators.

---

## Repository structure

A typical layout (your tree may be a superset) is:

- `gkp_overlap_qcomb_sweep_squeezing.py`  
  Generates data for the logical overlap \(|\langle 0_L^{(\Delta)} | 1_L^{(\Delta)} \rangle|^2\) as a function of Fock cutoff \(N\), squeezing \(s\) (in dB), and comb width \(\Delta\).

- `gkp_overlap_qcomb_sweep_squeezing_plot_from_data.py`  
  Re-plots the overlap data (including log-scale versions) and produces the heatmaps used for the overlap panel in Fig. 2(a).

- `gkp_cnot_gyrator_sweep.py`  
  Performs two-mode simulations of the gyrator-based SUM interaction, computes the effective logical unitary on the GKP subspace, and evaluates the average logical gate fidelity \(F_{\mathrm{avg}}(N,\theta)\).

- `gkp_cnot_gyrator_sweep_plot_from_data.py`  
  Loads the stored CNOT/SUM scan data and generates the fidelity maps and Wigner panels used for Fig. 2.

- `gkp_qcomb_stateprep_panels.py`  
  Generates the single-mode Wigner function panels for the approximate logical GKP eigenstates and logical superpositions shown in Fig. 1.

- `data_gkp_overlap_qcomb/`  
  NPZ files with precomputed overlap data.

- `data_gkp_cnot_gyrator_sweep/`  
  NPZ files with precomputed CNOT/SUM gate data.

- `figs_gkp_*`  
  Output directories where the figure scripts save PNGs used in the manuscript.

---

## Environment and dependencies

The simulations use a standard scientific Python stack:

- Python 3.9+
- `numpy`
- `scipy`
- `matplotlib`
- `qutip`
- `tqdm`

You can install the dependencies with:

```bash
pip install numpy scipy matplotlib qutip tqdm


## Parameters used in the manuscript

Unless otherwise stated, the main results and figures are obtained with:

- Fock cutoff: $N = 60$–$70$
- Comb width: $\Delta = 0.300$
- Single-mode squeezing (in dB): $s = 12.0~\mathrm{dB}$
- Optimal SUM interaction angle (at largest $N$): $\theta_{\mathrm{opt}} \approx 0.967~\mathrm{rad}$

These values can be adjusted directly in the configuration sections at the top of each script.
