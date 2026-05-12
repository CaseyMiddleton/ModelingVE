# Incorporating vaccine effects into epidemiological models: common pitfalls and solutions

This repository contains all code needed to reproduce the analyses and figures in:

> Middleton CE, Eales O, McCaw JM, Shearer FM. *Incorporating vaccine effects into epidemiological models: common pitfalls and solutions.*

---

## Repository structure

```
├── env/
│   ├── transmission_functions.py   # Core model library (see below)
│   ├── prettyplotlib.py            # Matplotlib styling helpers
│   ├── generate_figures.ipynb      # Generates all manuscript figures
│   └── output/
│       ├── heatmap_data.npz        # Pre-computed leaky correction accuracy grids
│       └── heterogeneous_concentration_heatmap.npz  # Pre-computed heterogeneous correction grids
```

---

## Dependencies

Python 3.8+ with the following packages:

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.6
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Reproducing the figures

All figures are generated from a single notebook:

```bash
cd env
jupyter notebook generate_figures.ipynb
```

Run the cells top to bottom. Two cells involve long-running simulations:

| Cell label | Runtime | Output file |
|---|---|---|
| Leaky correction accuracy | ~4 hours | `output/heatmap_data.npz` |
| Heterogeneous correction accuracy | ~4 minutes | `output/heterogeneous_concentration_heatmap.npz` |

Pre-computed versions of both `.npz` files are included in the repository so these cells can be skipped and the figure cells run directly from the saved data.

---

## Model library (`transmission_functions.py`)

The core library implements three vaccine mechanisms in a deterministic SIRV framework and provides supporting analytical tools.

**Vaccine models**

| Mechanism | Description |
|---|---|
| All-or-nothing (AoN) | A fixed fraction of vaccinees are fully protected; the rest receive no protection |
| Leaky | All vaccinees receive the same per-exposure reduction in susceptibility |
| Heterogeneous / heterogeneous | Vaccine effects are drawn from a Beta distribution, creating a spectrum of protection levels |

**Key functions**

| Function | Purpose |
|---|---|
| `run_SIRV` | Simulate SIRV dynamics for AoN or leaky vaccine |
| `run_SIRV_heterogeneous` | Simulate SIRV dynamics with heterogeneous vaccine effects |
| `solve_attack_rates` | Compute final-size attack rates analytically (leaky model) |
| `solve_attack_rates_heterogeneous` | Compute final-size attack rates analytically (heterogeneous model) |
| `solve_sigma_for_target_VE` | Find the corrected vaccine effect parameter that reproduces a measured VE |
| `solve_distribution_params_for_target_VE_heterogeneous` | Analogous correction for the heterogeneous model |
| `generate_vaccine_responder_classes` | Discretise a Beta distribution into K responder classes |
| `get_herd_immunity_threshold` | Compute the vaccination coverage required for herd immunity |
| `estimate_VE_cox_tvi` | Estimate VE from simulated trial data using a Cox time-varying interaction model |
| `get_VE_binomial` | Compute observed VE under a binomial challenge study design |

---

## Figure index

| Figure | Description | Saved to |
|---|---|---|
| Fig 1 | Time-varying VE under AoN, leaky, and heterogeneous models | `figs/raw/time-varying-VE.jpg` |
| Fig 2 | Schematic of compounding bias | `figs/raw/schematic.jpg` |
| Fig 3 | Heterogeneous model isoclines: VE_modeled as a function of mean effect and concentration | `figs/raw/heterogeneous_isoclines.jpg` |
| Fig 4 | Leaky correction accuracy across vaccination coverages | `figs/raw/correction_accuracy.jpg` |
| Fig 5 | Heterogeneous correction accuracy across concentration parameters | `figs/raw/heterogeneous_concentration_heatmap.jpg` |
| Fig 6 | Impact of parameterisation on predicted herd immunity threshold | `figs/raw/HIT.jpg` |
| Supp | Supplemental figures | `figs/` |