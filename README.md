# Simulation-Based Inference for a Leaky Integrate-and-Fire (LIF) Neuron
A modular implementation of simulation-based inference (SBI) for a single-neuron leaky integrate-and-fire (LIF) model using BayesFlow. The project demonstrates amortized neural posterior estimation for mechanistic models without explicit likelihood evaluation.

## Why this matters
Mechanistic neural models often have intractable likelihoods, making classical parameter estimation difficult. This project demonstrates how simulation-based inference enables efficient, reusable posterior estimation for complex biophysical systems.


This repository packages the full workflow:
- prior specification,
- mechanistic simulation,
- amortized posterior training,
- and posterior diagnostics.

It is designed for clarity and extensibility, while preserving the structure of the original research notebook.

---

## Project Overview

The project estimates latent biophysical parameters of a LIF neuron from observed voltage and spike traces. Instead of evaluating an explicit likelihood, it uses **amortized neural posterior estimation** through BayesFlow, where a neural network is trained on simulated datasets and then reused for fast inference.

### Inferred parameters

The workflow infers six parameters:

- `V_th` — spike threshold
- `tau_m` — membrane time constant
- `g_L` — leak conductance
- `V_reset` — reset potential after a spike
- `V_init` — initial membrane potential
- `E_L` — resting (leak reversal) potential

### Observation model

Each simulation produces:

- a membrane voltage time series (`voltage`),
- a binary spike train (`spikes`),

with optional additive Gaussian observation noise on voltage.

---

## Repository Layout

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py             # Simulation and training configuration dataclasses
│   ├── priors.py             # Prior distributions and constrained sampling
│   ├── simulator.py          # LIF dynamics + observation generation
│   ├── bayesflow_setup.py    # BayesFlow simulator, adapter, workflow, training helpers
│   └── diagnostics.py        # Loss/recovery/calibration/PPC helpers
├── notebooks/
│   └── 01_lif_bayesflow_sbi_main.ipynb
├── figures/                  # Example outputs (posterior, recovery, PPC, calibration, loss)
├── requirements.txt
├── ENVIRONMENT.md            # Environment and reproducibility notes
└── README.md
```

---

## Method at a Glance

1. **Sample parameters from the prior** (`src/priors.py`).
2. **Simulate synthetic observations** with the LIF forward model (`src/simulator.py`).
3. **Train BayesFlow offline** on many simulated datasets (`src/bayesflow_setup.py`).
4. **Evaluate posterior quality** with recovery, calibration histograms, and PPC-lite (`src/diagnostics.py`).

---

## Installation

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `bayesflow` is installed from a pinned Git commit in `requirements.txt`.

---

## Quickstart

### Build simulator and workflow

```python
from src.config import SimConfig
from src.bayesflow_setup import make_bayesflow_simulator, make_workflow

sim_cfg = SimConfig()
sim = make_bayesflow_simulator(sim_cfg, seed=45)
workflow = make_workflow(sim)
```

### Train offline

```python
from src.config import TrainConfig
from src.bayesflow_setup import train_offline

train_cfg = TrainConfig(
    n_train=20_000,
    n_val=1_000,
    epochs=50,
    batch_size=128,
    seed=45,
)

history, train_data, val_data = train_offline(workflow, train_cfg)
```

### Plot training loss

```python
from src.diagnostics import plot_loss

fig = plot_loss(history)
fig.show()
```

---

## Core Configuration

### `SimConfig` (`src/config.py`)

Controls simulation and observation settings, including:
- total duration `T` and timestep `dt`,
- refractory period `tref`,
- noisy current injection statistics (`I_mean`, `I_std`),
- observation noise (`obs_noise_std`),
- numerical safety voltage clipping (`v_clip`).

### `TrainConfig` (`src/config.py`)

Controls offline training:
- number of training and validation simulations,
- epochs,
- batch size,
- random seed.

### `PriorConfig` (`src/priors.py`)

Defines prior families and bounds:
- truncated normals for voltage-related parameters,
- clipped log-normal priors for `tau_m` and `g_L`,
- dependency-aware constraints (`V_init < E_L`, `V_reset < V_init`).

---

## Diagnostics and Evaluation

The diagnostics module provides wrappers around BayesFlow plotting utilities and a lightweight PPC implementation.

### Available diagnostics

- `plot_loss(...)` — training/validation loss trajectory.
- `plot_recovery(...)` — posterior recovery against known parameters.
- `plot_calibration_hist(...)` — rank histogram for calibration checks.
- `compute_ppc_lite(...)` — posterior predictive summary metrics:
  - 90% pointwise coverage,
  - observed spike-count percentile,
  - subthreshold MAE against predictive median.
- `plot_ppc_voltage(...)` and `plot_ppc_spikecount(...)` — visual PPC summaries.

---

## Example Outputs

Representative figures from prior runs are included in `figures/`:

- `loss.png`
- `posterior.png`
- `recovery.png`
- `calibration_sbc.png`
- `ppc_voltage.png`
- `ppc_spikecount.png`

These are provided as a reference snapshot and may not be exactly reproducible without recreating the original environment.

---

## Reproducibility Notes

This codebase is a cleaned and modularized refactor of the original notebook workflow. Exact numerical parity with the original experimental environment is not guaranteed due to dependency drift; however, the workflow and diagnostics remain fully functional.

Please read [`ENVIRONMENT.md`](ENVIRONMENT.md) before attempting strict reproduction.

---

## Running the Notebook

The notebook at:

- `notebooks/01_lif_bayesflow_sbi_main.ipynb`

captures the last known working end-to-end run and can be used for exploratory analysis, diagnostics, and figure regeneration.

---

## Extending the Project

Suggested directions:

- add richer summary networks or normalizing-flow architectures,
- include multi-condition stimuli and hierarchical priors,
- compare alternative simulators (e.g., adaptive-exponential IF),
- package a reproducible Docker/Conda environment,
- add benchmark scripts for posterior quality and runtime.

---

## Citation

If you use this repository in research or teaching, please cite:

- the BayesFlow framework,
- and this repository (URL and commit hash).

---

## Acknowledgments
This project builds on the BayesFlow ecosystem and the broader SBI community.
