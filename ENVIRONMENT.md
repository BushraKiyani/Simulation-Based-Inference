# Environment and reproducibility notes

This project was originally developed and executed in a dedicated research
environment with specific versions of BayesFlow, JAX, and Keras that are
no longer available on the author’s local machine.

The results shown in this repository (loss curves, posterior distributions,
calibration plots, and posterior predictive checks) were generated in that
original environment and are provided here as a frozen snapshot.

## Approximate original environment

The original setup included:

- Python ≥ 3.11
- BayesFlow (circa 2023–2024)
- JAX and jaxlib (CPU/GPU)
- TensorFlow / Keras backend
- NumPy, SciPy, Matplotlib, Seaborn

Exact package versions were managed on a separate machine and may need to
be recreated manually for full reproducibility.

## Current status

- The notebook in `notebooks/01_lif_bayesflow_sbi_main.ipynb` reflects the
  last known working version.
- The code in `src/` is a cleaned and modularized refactor of the original
  implementation, provided for clarity and extensibility.
- Full retraining is not guaranteed without recreating the original
  environment.

Future updates may include containerization (e.g. Docker) to restore
full reproducibility.
