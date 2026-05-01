"""
Merton (1976) jump-diffusion model simulator.

    dS_t / S_t = (mu - lambda * k) * dt + sigma * dW_t + (J - 1) * dN_t

where:
    - dW_t is a standard Brownian increment
    - dN_t is a Poisson increment with intensity lambda
    - log(J) ~ Normal(mu_J, sigma_J^2): jump size is log-normal
    - k = E[J - 1] = exp(mu_J + 0.5 * sigma_J^2) - 1: drift compensator
      so that the risk-neutral expected return remains mu

Calibrated to capture fat tails and crash risk that GBM cannot represent.
The threshold-based calibration in calibrators.py separates "jump days"
(returns beyond a 3-sigma threshold) from "normal days" and fits each
component independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MertonParams:
    """Calibrated parameters for a Merton jump-diffusion model."""
    mu: float           # Annualized drift (total)
    sigma: float        # Annualized diffusion volatility (excl. jumps)
    lambda_j: float     # Annualized jump intensity (jumps per year)
    mu_j: float         # Mean of log jump size
    sigma_j: float      # Std of log jump size

    @property
    def k(self) -> float:
        """Drift compensator: E[J - 1]."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0


def simulate_merton(
    s0: float,
    params: MertonParams,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Merton jump-diffusion price paths.

    The discretization is exact in distribution:
        log(S_{t+dt} / S_t) = (mu - 0.5*sigma^2 - lambda*k) * dt
                            + sigma * sqrt(dt) * Z
                            + sum over Poisson(lambda*dt) jumps of N(mu_j, sigma_j)

    Parameters
    ----------
    s0 : float
        Initial price.
    params : MertonParams
        Calibrated parameters.
    horizon_days : int
        Number of trading days to simulate.
    n_paths : int
        Number of independent paths.
    trading_days : int
        Trading-day convention.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_paths, horizon_days + 1).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days

    # Diffusion component
    drift = (params.mu - 0.5 * params.sigma ** 2 - params.lambda_j * params.k) * dt
    diffusion_std = params.sigma * np.sqrt(dt)
    z = rng.standard_normal((n_paths, horizon_days))
    diffusion_part = drift + diffusion_std * z

    # Jump component: Poisson number of jumps per step, each log-normal sized
    n_jumps = rng.poisson(params.lambda_j * dt, size=(n_paths, horizon_days))

    # For each cell, sample n_jumps[i,j] jump sizes and sum them.
    # Equivalent: sum of N normals = Normal(N*mu_j, sqrt(N)*sigma_j)
    # This vectorizes cleanly.
    jump_part = np.where(
        n_jumps > 0,
        n_jumps * params.mu_j + np.sqrt(np.maximum(n_jumps, 1)) * params.sigma_j * rng.standard_normal(n_jumps.shape),
        0.0,
    )
    # Zero out the random portion when there are no jumps
    jump_part = np.where(n_jumps > 0, jump_part, 0.0)

    log_increments = diffusion_part + jump_part
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(log_paths)
    return paths
