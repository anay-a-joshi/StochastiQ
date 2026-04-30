"""
Joint multi-asset Monte Carlo simulation under calibrated stochastic models.

Each stochastic model in StochastiQ (GBM, Merton, CEV, Heston) is calibrated
marginally per asset in Phase 3. To use these calibrations for portfolio
analysis, we need to simulate the *joint* evolution of all 7 assets
preserving their cross-asset correlation structure.

Approach: Cholesky correlation injection
----------------------------------------
For each model, we replace the i.i.d. standard normal shocks z_t (one per
asset per time step) with a vector of correlated shocks:

    Z_t = L @ epsilon_t

where epsilon_t ~ N(0, I_n) is i.i.d. standard normal across assets, and L
is the Cholesky factor of the empirical correlation matrix R from training
data: L L^T = R.

This preserves each model's marginal distribution per asset while
injecting realistic cross-asset dependence. It's the standard approach
used in production risk systems (Glasserman, "Monte Carlo Methods in
Financial Engineering", 2003).

For Heston, the price-variance correlation rho_i is already built into
each asset's simulator. The cross-asset correlation matrix R is applied
to the *price-side* shocks; variance shocks remain idiosyncratic.

For Merton, jump arrivals are independent across assets (each asset has
its own Poisson process); jump magnitudes are also independent. Only the
diffusion shocks are correlated across assets via R.

For CEV, the local volatility depends on the price level, but the shock
structure is unchanged.

This module exposes one function per model that simulates joint paths
across all assets, returning a (n_paths, n_steps + 1, n_assets) array
of price paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.gbm import GBMParams
from src.models.merton import MertonParams
from src.models.cev import CEVParams, PRICE_FLOOR as CEV_PRICE_FLOOR
from src.models.heston import HestonParams


# ============================================================
# Correlation infrastructure
# ============================================================

def cholesky_factor(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky factor of a correlation matrix.

    Adds a tiny diagonal regularization if the matrix is not strictly
    positive definite (which can happen due to numerical noise in the
    empirical correlation estimate).
    """
    R = np.asarray(correlation_matrix, dtype=float)
    n = R.shape[0]
    # Symmetrize to ensure exact symmetry
    R = 0.5 * (R + R.T)
    # Try Cholesky; on failure regularize and retry
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        eps = 1e-8
        for _ in range(10):
            try:
                L = np.linalg.cholesky(R + eps * np.eye(n))
                break
            except np.linalg.LinAlgError:
                eps *= 10
        else:
            raise RuntimeError("Failed to factor correlation matrix even with regularization")
    return L


def correlated_normals(
    rng: np.random.Generator,
    L: np.ndarray,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
    Generate correlated normal shocks of shape (n_paths, n_steps, n_assets).

    Uses the Cholesky factor L: independent normals epsilon are transformed
    to correlated shocks Z = epsilon @ L.T (using row vectors for samples).
    """
    n_assets = L.shape[0]
    eps = rng.standard_normal((n_paths, n_steps, n_assets))
    # Apply Cholesky transform along the last axis
    return eps @ L.T


# ============================================================
# Joint GBM simulation
# ============================================================

def simulate_joint_gbm(
    s0: np.ndarray,
    params_list: list[GBMParams],
    L: np.ndarray,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate joint GBM paths for n_assets with cross-asset correlation L L^T.

    Parameters
    ----------
    s0 : np.ndarray
        Initial prices, shape (n_assets,).
    params_list : list of GBMParams
        Calibrated parameters, one per asset.
    L : np.ndarray
        Cholesky factor of the cross-asset correlation matrix.
    horizon_days, n_paths, trading_days : int
        Simulation parameters.
    seed : int, optional
        Random seed.

    Returns
    -------
    paths : np.ndarray
        Shape (n_paths, horizon_days + 1, n_assets).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params_list)
    dt = 1.0 / trading_days
    sqrt_dt = np.sqrt(dt)

    mu = np.array([p.mu for p in params_list])           # (n_assets,)
    sigma = np.array([p.sigma for p in params_list])     # (n_assets,)

    # Drift in log space: (mu - 0.5 * sigma^2) * dt
    log_drift = (mu - 0.5 * sigma ** 2) * dt             # (n_assets,)

    # Generate all correlated shocks at once
    Z = correlated_normals(rng, L, n_paths, horizon_days)  # (n_paths, n_steps, n_assets)
    log_increments = log_drift + sigma * sqrt_dt * Z       # (n_paths, n_steps, n_assets)
    log_paths_increments = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, horizon_days + 1, n_assets))
    paths[:, 0, :] = s0
    paths[:, 1:, :] = s0 * np.exp(log_paths_increments)

    return paths


# ============================================================
# Joint Merton simulation
# ============================================================

def simulate_joint_merton(
    s0: np.ndarray,
    params_list: list[MertonParams],
    L: np.ndarray,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate joint Merton paths.

    Diffusion shocks are correlated across assets via L. Jump arrivals
    (Poisson) and jump magnitudes (log-normal) are independent across
    assets, reflecting the typical assumption that asset-specific shocks
    are idiosyncratic.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params_list)
    dt = 1.0 / trading_days
    sqrt_dt = np.sqrt(dt)

    mu       = np.array([p.mu for p in params_list])
    sigma    = np.array([p.sigma for p in params_list])
    lambda_j = np.array([p.lambda_j for p in params_list])
    mu_j     = np.array([p.mu_j for p in params_list])
    sigma_j  = np.array([p.sigma_j for p in params_list])
    k        = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    # Diffusion drift: (mu - lambda_j * k - 0.5 * sigma^2) * dt
    diff_drift = (mu - lambda_j * k - 0.5 * sigma ** 2) * dt  # (n_assets,)

    # Correlated diffusion shocks
    Z = correlated_normals(rng, L, n_paths, horizon_days)     # (n_paths, n_steps, n_assets)
    diff_log_inc = diff_drift + sigma * sqrt_dt * Z           # (n_paths, n_steps, n_assets)

    # Independent Poisson jump counts per asset per step
    n_jumps = rng.poisson(lambda_j * dt, size=(n_paths, horizon_days, n_assets))

    # Jump contribution: sum of n_jumps log-normals with mean mu_j, std sigma_j
    # E[sum of N i.i.d. Normal(m, s^2)] has mean N*m, var N*s^2
    # We sample directly: jump_log_inc ~ Normal(n*mu_j, sqrt(n)*sigma_j) when n >= 1, else 0
    jump_means = n_jumps * mu_j                                # (n_paths, n_steps, n_assets)
    jump_stds = np.sqrt(n_jumps) * sigma_j                     # (n_paths, n_steps, n_assets)
    jump_eps = rng.standard_normal((n_paths, horizon_days, n_assets))
    jump_log_inc = jump_means + jump_stds * jump_eps           # (n_paths, n_steps, n_assets)

    log_increments = diff_log_inc + jump_log_inc
    log_paths_increments = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, horizon_days + 1, n_assets))
    paths[:, 0, :] = s0
    paths[:, 1:, :] = s0 * np.exp(log_paths_increments)

    return paths


# ============================================================
# Joint CEV simulation
# ============================================================

def simulate_joint_cev(
    s0: np.ndarray,
    params_list: list[CEVParams],
    L: np.ndarray,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate joint CEV paths with correlated shocks.

    Unlike GBM/Merton, CEV's local volatility depends on the current price
    level, so we cannot vectorize across time -- each step needs the
    previous price. We loop over time but keep paths and assets vectorized.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params_list)
    dt = 1.0 / trading_days
    sqrt_dt = np.sqrt(dt)

    mu     = np.array([p.mu for p in params_list])      # (n_assets,)
    sigma  = np.array([p.sigma for p in params_list])   # (n_assets,)
    gamma  = np.array([p.gamma for p in params_list])   # (n_assets,)
    gamma_minus_1 = gamma - 1.0

    # Pre-generate correlated shocks
    Z = correlated_normals(rng, L, n_paths, horizon_days)  # (n_paths, n_steps, n_assets)

    paths = np.empty((n_paths, horizon_days + 1, n_assets))
    paths[:, 0, :] = s0
    s = np.tile(s0, (n_paths, 1)).astype(float)            # (n_paths, n_assets)

    for t in range(horizon_days):
        s_safe = np.maximum(s, CEV_PRICE_FLOOR)
        # Local vol: sigma * S^(gamma - 1), broadcast across (n_paths, n_assets)
        local_vol = sigma * np.power(s_safe, gamma_minus_1)
        drift_term = (mu - 0.5 * local_vol ** 2) * dt
        diffusion_term = local_vol * sqrt_dt * Z[:, t, :]
        log_s = np.log(s_safe) + drift_term + diffusion_term
        s = np.exp(log_s)
        paths[:, t + 1, :] = s

    return paths


# ============================================================
# Joint Heston simulation
# ============================================================

def simulate_joint_heston(
    s0: np.ndarray,
    params_list: list[HestonParams],
    L: np.ndarray,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate joint Heston paths with cross-asset correlation on price shocks.

    Each asset has its own price-variance correlation rho_i (the leverage
    correlation from the asset's calibration). The cross-asset correlation
    matrix L L^T is applied to the price-side shocks. Variance shocks are
    asset-specific (one per asset per step) to avoid double-counting
    correlation structure.

    Uses full-truncation Euler scheme for variance positivity.

    Returns
    -------
    (paths, variance_paths) : tuple of np.ndarray
        Both shape (n_paths, horizon_days + 1, n_assets).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params_list)
    dt = 1.0 / trading_days
    sqrt_dt = np.sqrt(dt)

    mu       = np.array([p.mu for p in params_list])
    kappa    = np.array([p.kappa for p in params_list])
    theta    = np.array([p.theta for p in params_list])
    sigma_v  = np.array([p.sigma_v for p in params_list])
    rho      = np.array([p.rho for p in params_list])
    v0       = np.array([p.v0 for p in params_list])
    sqrt_one_minus_rho_sq = np.sqrt(np.maximum(1.0 - rho ** 2, 0.0))

    # Cross-asset correlated price-side shocks
    Z_S = correlated_normals(rng, L, n_paths, horizon_days)  # (n_paths, n_steps, n_assets)
    # Independent variance shocks per asset
    Z_V = rng.standard_normal((n_paths, horizon_days, n_assets))

    # Build correlated (W_S, W_V) per asset:
    # W_V = Z_V (independent)
    # W_S = rho * Z_V + sqrt(1 - rho^2) * Z_S
    # This way W_S has correlation rho with W_V *per asset*, and the
    # cross-asset correlation comes through Z_S.
    dW_V = Z_V
    dW_S = rho * Z_V + sqrt_one_minus_rho_sq * Z_S          # (n_paths, n_steps, n_assets)

    paths = np.empty((n_paths, horizon_days + 1, n_assets))
    var_paths = np.empty((n_paths, horizon_days + 1, n_assets))
    paths[:, 0, :] = s0
    var_paths[:, 0, :] = v0

    s = np.tile(s0, (n_paths, 1)).astype(float)
    v = np.tile(v0, (n_paths, 1)).astype(float)

    for t in range(horizon_days):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        v_next = v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * dW_V[:, t, :]
        log_s_next = np.log(s) + (mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * dW_S[:, t, :]
        s_next = np.exp(log_s_next)

        s = s_next
        v = v_next
        paths[:, t + 1, :] = s
        var_paths[:, t + 1, :] = v

    return paths, var_paths


# ============================================================
# High-level orchestrator
# ============================================================

def simulate_all_models(
    s0: np.ndarray,
    calibrated: dict,
    tickers: list[str],
    correlation_matrix: np.ndarray,
    horizon_days: int,
    n_paths: int,
    trading_days: int = 252,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run joint Monte Carlo simulation under all four calibrated models.

    Parameters
    ----------
    s0 : np.ndarray
        Initial prices, shape (n_assets,).
    calibrated : dict
        Nested dict from Phase 3: calibrated[ticker][model_name] -> Params.
    tickers : list[str]
        Asset order matching s0 and correlation_matrix.
    correlation_matrix : np.ndarray
        Empirical cross-asset correlation matrix, shape (n_assets, n_assets).
    horizon_days, n_paths, trading_days : int
        Simulation parameters.
    seed : int, optional
        Base random seed (each model gets a different offset).

    Returns
    -------
    dict[str, np.ndarray]
        Keys: 'GBM', 'Merton', 'CEV', 'Heston'. Each value is shape
        (n_paths, horizon_days + 1, n_assets).
    """
    L = cholesky_factor(correlation_matrix)
    base_seed = 0 if seed is None else seed

    gbm_params    = [calibrated[t]["GBM"]    for t in tickers]
    merton_params = [calibrated[t]["Merton"] for t in tickers]
    cev_params    = [calibrated[t]["CEV"]    for t in tickers]
    heston_params = [calibrated[t]["Heston"] for t in tickers]

    paths_gbm    = simulate_joint_gbm(s0, gbm_params,    L, horizon_days, n_paths, trading_days, seed=base_seed + 1)
    paths_merton = simulate_joint_merton(s0, merton_params, L, horizon_days, n_paths, trading_days, seed=base_seed + 2)
    paths_cev    = simulate_joint_cev(s0, cev_params,    L, horizon_days, n_paths, trading_days, seed=base_seed + 3)
    paths_heston, _ = simulate_joint_heston(s0, heston_params, L, horizon_days, n_paths, trading_days, seed=base_seed + 4)

    return {
        "GBM":    paths_gbm,
        "Merton": paths_merton,
        "CEV":    paths_cev,
        "Heston": paths_heston,
    }


# ============================================================
# Path -> return conversion
# ============================================================

def paths_to_returns(paths: np.ndarray) -> np.ndarray:
    """
    Convert price paths to daily log returns.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_paths, n_steps + 1, n_assets).

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps, n_assets).
    """
    return np.diff(np.log(paths), axis=1)


def paths_to_terminal_returns(paths: np.ndarray) -> np.ndarray:
    """
    Convert price paths to total log returns over the full horizon.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_assets). Total log return: log(S_T / S_0).
    """
    return np.log(paths[:, -1, :] / paths[:, 0, :])
