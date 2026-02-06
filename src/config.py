from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """Configuration for LIF simulation and observation model."""
    T: float = 200.0          # ms
    dt: float = 0.2           # ms
    tref: float = 2.0         # ms refractory
    # stimulus for Iinj ~ N(I_mean, I_std)
    I_mean: float = 200.0
    I_std: float = 50.0
    # observation noise (added to voltage trace)
    obs_noise_std: float = 1.0
    # safety clamp for voltages to avoid numerical overflow
    v_clip: Tuple[float, float] = (-100.0, 100.0)

    def time_grid(self) -> np.ndarray:
        return np.arange(0.0, self.T, self.dt, dtype=float)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for BayesFlow training."""
    n_train: int = 20_000
    n_val: int = 1_000
    epochs: int = 50
    batch_size: int = 128
    seed: int = 45
