from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import bayesflow


@dataclass(frozen=True)
class PPCResult:
    coverage_90: float
    spike_percentile: float
    subthreshold_mae: float
    v_lo: np.ndarray
    v_hi: np.ndarray
    v_med: np.ndarray
    pp_spike_counts: np.ndarray


def plot_loss(history, figsize=(10, 4)):
    """Thin wrapper around BayesFlow loss plot."""
    fig = bayesflow.diagnostics.plots.loss(history)
    for ax in fig.axes:
        ax.set_title("Loss Trajectory")
        ax.grid(True, alpha=0.2)
    return fig


def plot_recovery(posterior_samples: Dict[str, np.ndarray], test_sims: Dict[str, np.ndarray]):
    """BayesFlow recovery plot."""
    fig = bayesflow.diagnostics.plots.recovery(posterior_samples, test_sims)
    return fig


def plot_calibration_hist(posterior_samples: Dict[str, np.ndarray], test_sims: Dict[str, np.ndarray]):
    """BayesFlow calibration / rank histogram plot."""
    fig = bayesflow.diagnostics.plots.calibration_histogram(posterior_samples, test_sims)
    return fig


def compute_ppc_lite(workflow,
                     observed_voltage: np.ndarray,
                     observed_spikes: np.ndarray,
                     posterior_samples: Dict[str, np.ndarray],
                     K: int = 200,
                     rng: Optional[np.random.Generator] = None) -> PPCResult:
    """
    Posterior predictive check "lite" mirroring your notebook:
      - simulate K datasets from posterior
      - compute 90% pointwise band and median
      - compute:
          coverage in 90% band,
          spike-count percentile,
          subthreshold deviation vs predictive median
    """
    if rng is None:
        rng = np.random.default_rng(45)

    param_names = ["V_th", "tau_m", "g_L", "V_reset", "V_init", "E_L"]

    obs_v = observed_voltage.reshape(-1)
    obs_spike_count = int(observed_spikes.sum())

    # choose K posterior draws
    n_post = posterior_samples[param_names[0]].reshape(-1).shape[0]
    post_idx = rng.integers(0, n_post, size=K)

    pp_volt = []
    pp_spike_counts = []

    for i in post_idx:
        pars = {p: float(posterior_samples[p].reshape(-1)[i]) for p in param_names}
        sim = workflow.simulator.simulator_fun(**pars)  # direct sim call
        pp_volt.append(sim["voltage"].reshape(-1))
        pp_spike_counts.append(int(sim["spikes"].sum()))

    pp_volt = np.vstack(pp_volt)
    pp_spike_counts = np.asarray(pp_spike_counts)

    v_lo = np.percentile(pp_volt, 5, axis=0)
    v_hi = np.percentile(pp_volt, 95, axis=0)
    v_med = np.median(pp_volt, axis=0)

    coverage = float(np.mean((obs_v >= v_lo) & (obs_v <= v_hi)))
    spike_percentile = float(np.mean(pp_spike_counts <= obs_spike_count))

    Vth_med = float(np.median(posterior_samples["V_th"]))
    mask_sub = obs_v < Vth_med
    sub_mae = float(np.mean(np.abs(obs_v[mask_sub] - v_med[mask_sub])))

    return PPCResult(
        coverage_90=coverage,
        spike_percentile=spike_percentile,
        subthreshold_mae=sub_mae,
        v_lo=v_lo, v_hi=v_hi, v_med=v_med,
        pp_spike_counts=pp_spike_counts,
    )


def plot_ppc_voltage(t: np.ndarray, obs_v: np.ndarray, ppc: PPCResult, figsize=(10, 4)):
    fig = plt.figure(figsize=figsize)
    plt.fill_between(t, ppc.v_lo, ppc.v_hi, alpha=0.3, label="Posterior predictive 90% band")
    plt.plot(t, ppc.v_med, "k-", lw=1.5, label="Posterior predictive median")
    plt.plot(t, obs_v, "r-", lw=1.0, alpha=0.8, label="Observed voltage")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("PPC-lite: Voltage Trace")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_ppc_spikecount(pp_spike_counts: np.ndarray, obs_spike_count: int, figsize=(6, 4)):
    fig = plt.figure(figsize=figsize)
    bins = np.arange(pp_spike_counts.min() - 0.5, pp_spike_counts.max() + 1.5, 1)
    plt.hist(pp_spike_counts, bins=bins, alpha=0.7, edgecolor="black")
    plt.axvline(obs_spike_count, color="red", linestyle="--", lw=2, label=f"Observed: {obs_spike_count}")
    plt.xlabel("Spike count")
    plt.ylabel("Frequency")
    plt.title("PPC-lite: Spike Count")
    plt.legend()
    plt.tight_layout()
    return fig
