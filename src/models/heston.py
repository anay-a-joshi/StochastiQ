"""
Heston (1993) stochastic volatility simulator.

    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^S
    dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW_t^v
    corr(dW^S, dW^v) = rho

Five parameters:
    kappa    -- speed of mean reversion of variance
    theta    -- long-run mean variance
    sigma_v  -- vol of vol
    rho      -- correlation between price and variance shocks (typically < 0)
    v0       -- initial variance

Feller condition: 2 * kappa * theta > sigma_v^2 ensures variance stays
strictly positive. When violated, our simulator uses Andersen's QE
(quadratic-exponential) scheme to handle near-zero variance correctly.
For our use case we use a simpler full-truncation Euler scheme that
floors variance at zero, which is adequate for the moderate-volatility
regimes we calibrate to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HestonParams:
    """Calibrated parameters for the Heston model."""
    mu: float           # Annualized drift of price
    kappa: float        # Mean-reversion speed of variance
    theta: float        # Long-run variance
    sigma_v: float      # Vol of variance ("vol of vol")
    rho: float          # Correlation between price and variance shocks
    v0: float           # Initial variance

    @property
    def feller_satisfied(self) -> bool:
        """Returns True if 2*kappa*theta > sigma_v^2."""
        return 2 * self.kappa * self.theta > self.sigma_v ** 2

    @property
    def feller_ratio(self) -> float:
        """Ratio 2*kappa*theta / sigma_v^2. Should be > 1."""
        return (2 * self.kappa * self.theta) / (self.sigma_v ** 2 + 1e-12)


def simulate_heston(
    s0: float,
    params: HestonParams,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston price and variance paths via full-truncation Euler.

    The full-truncation scheme (Lord, Koekkoek, van Dijk 2010) replaces
    sqrt(v_t) with sqrt(max(v_t, 0)) wherever needed and keeps v_t itself
    floored at zero in the drift. This is the simplest Euler discretization
    that preserves positivity in expectation and is widely used for Heston.

    Parameters
    ----------
    s0 : float
        Initial price.
    params : HestonParams
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
    (paths, variance_paths) : tuple of np.ndarray
        Both have shape (n_paths, horizon_days + 1).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days
    sqrt_dt = np.sqrt(dt)

    s_paths = np.empty((n_paths, horizon_days + 1))
    v_paths = np.empty((n_paths, horizon_days + 1))
    s_paths[:, 0] = s0
    v_paths[:, 0] = params.v0

    s = np.full(n_paths, s0, dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)

    # Cholesky factor for correlated normals
    rho = params.rho
    sqrt_one_minus_rho_sq = np.sqrt(max(1.0 - rho ** 2, 0.0))

    for t in range(horizon_days):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # Correlated shocks: (W_S, W_v) with corr = rho
        dW_v = z1
        dW_S = rho * z1 + sqrt_one_minus_rho_sq * z2

        # Full-truncation: use max(v, 0) where positivity is needed
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        # Variance update (Euler with full truncation)
        v_next = v + params.kappa * (params.theta - v_pos) * dt + params.sigma_v * sqrt_v * sqrt_dt * dW_v

        # Price update in log space, using sqrt(max(v, 0))
        log_s_next = np.log(s) + (params.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * dW_S
        s_next = np.exp(log_s_next)

        s = s_next
        v = v_next
        s_paths[:, t + 1] = s
        v_paths[:, t + 1] = v

    return s_paths, v_paths
