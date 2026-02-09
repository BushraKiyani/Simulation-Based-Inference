# Simulation-Based Inference for LIF Neuron Models

This repository contains a simulation-based inference (SBI) workflow for a
leaky integrate-and-fire (LIF) neuron model using BayesFlow. It provides a
clean, modular implementation of the simulator, prior specification, and
BayesFlow training pipeline, along with diagnostics to evaluate posterior
quality.

## Highlights

- **LIF simulator** with noisy current injection and observation noise.
- **Prior sampling** for biologically plausible parameters.
- **BayesFlow workflow** for amortized posterior inference.
- **Diagnostics** for recovery, calibration, and posterior predictive checks.

## Repository Structure

- `src/`
  - `config.py` — simulation and training configuration dataclasses.
  - `priors.py` — prior specification and sampling utilities.
  - `simulator.py` — LIF simulator + observation model.
  - `bayesflow_setup.py` — BayesFlow simulator/workflow helpers.
  - `diagnostics.py` — plotting and PPC-lite utilities.
- `notebooks/` — exploratory and reproducibility notebooks.
- `figures/` — generated figures and plots.
- `ENVIRONMENT.md` — details about the original environment and reproducibility.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

> The original results were produced in a now-unavailable research
> environment. See `ENVIRONMENT.md` for details and reproducibility notes.

## Quickstart

### 1) Create a BayesFlow simulator and workflow

```python
from src.config import SimConfig, TrainConfig
from src.bayesflow_setup import make_bayesflow_simulator, make_workflow

sim_cfg = SimConfig()
sim = make_bayesflow_simulator(sim_cfg)
workflow = make_workflow(sim)
```

### 2) Train offline

```python
from src.config import TrainConfig
from src.bayesflow_setup import train_offline

train_cfg = TrainConfig(epochs=10)
history, train_data, val_data = train_offline(workflow, train_cfg)
```

### 3) Diagnostics (optional)

```python
from src.diagnostics import plot_loss

fig = plot_loss(history)
fig.show()
```

## Notes on Reproducibility

The repository contains refactored code and a notebook snapshot of the last
known working version. Full retraining may require recreating the original
software environment. See `ENVIRONMENT.md` for details.

## Acknowledgments

This project is built around the BayesFlow framework for simulation-based
inference.
