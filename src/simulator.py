from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from .config import SimConfig


def default_pars(cfg: SimConfig, **kwargs) -> Dict[str, float]:
    """Create a parameter dict for the simulator with a precomputed time grid."""
    pars = {
        "tref": cfg.tref,
        "T": cfg.T,
        "dt": cfg.dt,
    }
    pars["range_t"] = cfg.time_grid()
    pars.update(kwargs)
    return pars


def generate_noisy_current(cfg: SimConfig, rng: np.random.Generator, Lt: int) -> np.ndarray:
    """Generate a noisy input current for the LIF model."""
    return rng.normal(loc=cfg.I_mean, scale=cfg.I_std, size=Lt)


def run_LIF(pars: Dict[str, float], cfg: SimConfig, rng: np.random.Generator,
            Iinj: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a leaky integrate-and-fire neuron.

    Parameters
    ----------
    pars : dict
        Must contain V_th, tau_m, g_L, V_reset, V_init, E_L and time info.
    cfg : SimConfig
        Simulation / observation config (dt, T, refractory, etc.).
    rng : np.random.Generator
        RNG for stimulus generation.
    Iinj : optional np.ndarray
        If None, a noisy current is generated.

    Returns
    -------
    v : np.ndarray, shape (Lt,)
        Membrane potential trace.
    spike_train : np.ndarray, shape (Lt,)
        Binary spike train aligned to dt grid (1 at spike bin).
    """
    # Basic validation
    if pars["tau_m"] <= 0 or pars["g_L"] <= 0:
        raise ValueError("tau_m and g_L must be positive")
    if pars["V_th"] <= pars["V_reset"]:
        raise ValueError("V_th must be greater than V_reset")

    V_th, V_reset = pars["V_th"], pars["V_reset"]
    tau_m, g_L = pars["tau_m"], pars["g_L"]
    V_init, E_L = pars["V_init"], pars["E_L"]
    dt, range_t = pars["dt"], pars["range_t"]
    Lt = range_t.size
    tref = pars["tref"]

    if Iinj is None:
        Iinj = generate_noisy_current(cfg, rng=rng, Lt=Lt)
    if Iinj.shape[0] != Lt:
        raise ValueError(f"Iinj must have length {Lt}, got {Iinj.shape[0]}")

    v = np.zeros(Lt, dtype=float)
    v[0] = V_init

    rec_spikes = []
    tr = 0.0  # refractory counter in steps (not ms)
    for it in range(Lt - 1):
        if tr > 0:
            v[it] = V_reset
            tr -= 1
        elif v[it] >= V_th:
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt

        dv = (-(v[it] - E_L) + Iinj[it] / g_L) * (dt / tau_m)
        v[it + 1] = v[it] + dv
        v[it + 1] = np.clip(v[it + 1], cfg.v_clip[0], cfg.v_clip[1])

    spike_train = np.zeros(Lt, dtype=float)
    if rec_spikes:
        spike_train[np.array(rec_spikes, dtype=int)] = 1.0
    return v, spike_train


def simulate_observation(theta: Dict[str, float], cfg: SimConfig,
                         rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Produce observation dict expected by BayesFlow adapter.

    Returns dict with:
      - "voltage": (Lt, 1)
      - "spikes":  (Lt, 1)
    """
    pars = default_pars(cfg, **theta)
    v, spikes = run_LIF(pars, cfg=cfg, rng=rng, Iinj=None)

    # observation noise
    if cfg.obs_noise_std > 0:
        v = v + rng.normal(0.0, cfg.obs_noise_std, size=v.shape)

    return {
        "voltage": v[..., np.newaxis].astype(np.float32),
        "spikes": spikes[..., np.newaxis].astype(np.float32),
    }
