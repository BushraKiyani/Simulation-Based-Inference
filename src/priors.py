from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy.stats import truncnorm


def truncated_normal(rng: np.random.Generator, mean: float, std: float, low: float, high: float) -> float:
    """Sample from a truncated normal distribution."""
    a, b = (low - mean) / std, (high - mean) / std
    return float(truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))


@dataclass(frozen=True)
class PriorConfig:
    """
    Prior configuration.

    These are set to mirror your std5 notebook:
    - Voltages via truncated normals with loose dependencies.
    - tau_m and g_L via clipped log-normal.
    """
    # V_th
    V_th_mean: float = -55.0
    V_th_std: float = 2.0
    V_th_low: float = -60.0
    V_th_high: float = -50.0

    # E_L
    E_L_mean: float = -70.0
    E_L_std: float = 2.0
    E_L_low: float = -75.0
    E_L_high: float = -60.0

    # V_init bounds depend on E_L
    V_init_mean: float = -70.0
    V_init_std: float = 2.0
    V_init_low: float = -80.0
    V_init_high_cap: float = -65.0   # upper cap; also must be < E_L - 0.5

    # V_reset bounds depend on V_init
    V_reset_mean: float = -72.0
    V_reset_std: float = 2.0
    V_reset_low: float = -85.0
    V_reset_high_cap: float = -65.0  # upper cap; also must be < V_init - 0.5

    # log-normal (clipped)
    tau_m_logmean: float = np.log(10.0)
    tau_m_logsigma: float = 0.2
    tau_m_low: float = 5.0
    tau_m_high: float = 20.0

    g_L_logmean: float = np.log(10.0)
    g_L_logsigma: float = 0.2
    g_L_low: float = 5.0
    g_L_high: float = 15.0


def sample_prior(rng: np.random.Generator, pcfg: PriorConfig = PriorConfig()) -> Dict[str, float]:
    """Sample theta = {V_th, tau_m, g_L, V_reset, V_init, E_L} from the prior."""
    V_th = truncated_normal(rng, pcfg.V_th_mean, pcfg.V_th_std, pcfg.V_th_low, pcfg.V_th_high)
    E_L = truncated_normal(rng, pcfg.E_L_mean, pcfg.E_L_std, pcfg.E_L_low, pcfg.E_L_high)

    V_init_high = min(E_L - 0.5, pcfg.V_init_high_cap)
    V_init = truncated_normal(rng, pcfg.V_init_mean, pcfg.V_init_std, pcfg.V_init_low, V_init_high)

    V_reset_high = min(V_init - 0.5, pcfg.V_reset_high_cap)
    V_reset = truncated_normal(rng, pcfg.V_reset_mean, pcfg.V_reset_std, pcfg.V_reset_low, V_reset_high)

    tau_m = float(np.clip(rng.lognormal(mean=pcfg.tau_m_logmean, sigma=pcfg.tau_m_logsigma),
                          pcfg.tau_m_low, pcfg.tau_m_high))
    g_L = float(np.clip(rng.lognormal(mean=pcfg.g_L_logmean, sigma=pcfg.g_L_logsigma),
                        pcfg.g_L_low, pcfg.g_L_high))

    return {
        "V_th": V_th,
        "tau_m": tau_m,
        "g_L": g_L,
        "V_reset": V_reset,
        "V_init": V_init,
        "E_L": E_L,
    }
