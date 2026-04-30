"""
Constant Elasticity of Variance (CEV) simulator.

    dS_t = mu * S_t * dt + sigma * S_t^gamma * dW_t

The CEV model generalizes GBM by letting the diffusion coefficient depend on
the price level. When gamma = 1 it reduces to GBM. When gamma < 1, volatility
rises as the price falls -- this is the "leverage effect" empirically observed
in equities (high vol on the way down, low vol on the way up).

Equivalent local-volatility form:
    sigma_local(S) = sigma * S^(gamma - 1)

When gamma < 1 the SDE can reach zero in finite time. We use a log-Euler
discretization with a small positive floor (PRICE_FLOOR) to prevent numerical
issues from log(0) and S^(gamma-1) at S = 0. This is equivalent to imposing
a reflecting boundary at a near-zero level, a standard treatment for CEV
simulation with gamma < 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Floor on the price to prevent log(0) and division-by-zero in S^(gamma-1).
# At any plausible asset scale, 1e-8 is effectively zero.
PRICE_FLOOR: float = 1e-8


@dataclass
class CEVParams:
    """Calibrated parameters for the CEV model."""
    mu: float       # Annualized drift
    sigma: float    # Volatility scale parameter
    gamma: float    # Elasticity (1.0 reduces to GBM)


def simulate_cev(
    s0: float,
    params: CEVParams,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate CEV price paths via log-space Euler-Maruyama with positive floor.

    Applying Ito's lemma to X = log(S):
        dX = (mu - 0.5 * sigma^2 * S^(2*gamma - 2)) * dt + sigma * S^(gamma - 1) * dW

    We integrate this in log-space and exponentiate. To prevent numerical
    issues when S is near zero (which can happen for gamma < 1), the price
    is floored at PRICE_FLOOR before each step's local-vol computation.

    Parameters
    ----------
    s0 : float
        Initial price.
    params : CEVParams
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
        Shape (n_paths, horizon_days + 1). All entries are strictly positive.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days
    sqrt_dt = np.sqrt(dt)

    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = s0
    s = np.full(n_paths, s0, dtype=float)

    for t in range(horizon_days):
        z = rng.standard_normal(n_paths)
        # Floor before computing local vol to prevent S^(gamma-1) blowup
        s_safe = np.maximum(s, PRICE_FLOOR)
        local_vol = params.sigma * np.power(s_safe, params.gamma - 1.0)
        drift_term = (params.mu - 0.5 * local_vol ** 2) * dt
        diffusion_term = local_vol * sqrt_dt * z
        log_s = np.log(s_safe) + drift_term + diffusion_term
        s = np.exp(log_s)
        paths[:, t + 1] = s

    return paths
