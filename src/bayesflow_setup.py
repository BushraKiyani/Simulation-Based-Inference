from __future__ import annotations
import numpy as np
import bayesflow
from .config import SimConfig, TrainConfig
from .priors import sample_prior, PriorConfig
from .simulator import simulate_observation


def make_bayesflow_simulator(cfg: SimConfig, pcfg: PriorConfig = PriorConfig(), seed: int = 45):
    """
    Create a BayesFlow simulator object with the same signature as in your notebook.

    Returns
    -------
    sim : bayesflow.Simulator
        Simulator with prior and simulator functions registered.
    """
    rng = np.random.default_rng(seed)

    def _prior():
        return sample_prior(rng=rng, pcfg=pcfg)

    def _simulator(V_th, tau_m, g_L, V_reset, V_init, E_L):
        theta = {
            "V_th": float(V_th),
            "tau_m": float(tau_m),
            "g_L": float(g_L),
            "V_reset": float(V_reset),
            "V_init": float(V_init),
            "E_L": float(E_L),
        }
        # IMPORTANT: use an independent RNG stream per call to avoid subtle correlations
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        return simulate_observation(theta=theta, cfg=cfg, rng=local_rng)

    return bayesflow.make_simulator([_prior, _simulator])


def make_workflow(sim, summary_dim: int = 32, hidden_dim: int = 64) -> bayesflow.BasicWorkflow:
    """
    Construct the adapter, networks, and workflow.
    Mirrors the notebook architecture:
      - TimeSeriesNetwork(hidden_dim=64, summary_dim=32)
      - CouplingFlow(num_params=6)
    """
    adapter = (
        bayesflow.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["V_th", "tau_m", "g_L", "V_reset", "V_init", "E_L"], into="inference_variables")
        .concatenate(["voltage", "spikes"], into="summary_variables")
    )

    summary_network = bayesflow.networks.TimeSeriesNetwork(hidden_dim=hidden_dim, summary_dim=summary_dim)
    inference_network = bayesflow.networks.CouplingFlow(num_params=6)

    workflow = bayesflow.BasicWorkflow(
        simulator=sim,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        standardize=["inference_variables", "summary_variables"],
    )
    return workflow


def train_offline(workflow: bayesflow.BasicWorkflow, tcfg: TrainConfig):
    """
    Simulate training/validation data and fit offline.

    Returns
    -------
    history : keras History-like
    training_data, validation_data : dict
    """
    np.random.seed(tcfg.seed)
    training_data = workflow.simulate(tcfg.n_train)
    validation_data = workflow.simulate(tcfg.n_val)

    history = workflow.fit_offline(
        data=training_data,
        epochs=tcfg.epochs,
        batch_size=tcfg.batch_size,
        validation_data=validation_data,
    )
    return history, training_data, validation_data
