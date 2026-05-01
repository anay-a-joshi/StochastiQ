"""
Geometric Brownian Motion (GBM) simulator.

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Closed-form solution:
    S_t = S_0 * exp((mu - 0.5*sigma^2) * t + sigma * W_t)

This is the simplest of the equity SDEs and serves as the baseline against
which Merton, CEV, and Heston are compared. GBM assumes constant drift,
constant volatility, log-normal returns, and continuous paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GBMParams:
    """Calibrated parameters for a Geometric Brownian Motion."""
    mu: float       # Annualized drift
    sigma: float    # Annualized volatility


def simulate_gbm(
    s0: float,
    params: GBMParams,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM price paths using the exact (log-space) discretization.

    Parameters
    ----------
    s0 : float
        Initial price.
    params : GBMParams
        Calibrated drift and volatility (annualized).
    horizon_days : int
        Number of trading days to simulate forward.
    n_paths : int
        Number of independent paths.
    trading_days : int
        Trading-day convention (default 252).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, horizon_days + 1). Column 0 is s0.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days

    drift = (params.mu - 0.5 * params.sigma ** 2) * dt
    diffusion_std = params.sigma * np.sqrt(dt)

    shocks = rng.standard_normal((n_paths, horizon_days))
    log_increments = drift + diffusion_std * shocks

    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(log_paths)

    return paths
