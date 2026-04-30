"""
Robust portfolio optimization across multiple stochastic models.

In Phase 2 we optimized portfolios using historical mean-variance
analysis. In Phase 4 we generalize: each of the four calibrated stochastic
models gives us a different distribution of forward returns, and the
"optimal" portfolio differs across models. We address this in three ways:

1. **Per-model optimization** (`optimize_max_sharpe`) -- solve for the
   max-Sharpe portfolio under each model's simulated distribution
   independently. This produces 4 portfolios that may disagree.

2. **Min-max Sharpe robust** (`optimize_minmax_sharpe`) -- find weights
   that maximize the *worst-case* Sharpe across all 4 models. This is
   robust optimization in the Ben-Tal/Nemirovski sense: we hedge against
   model misspecification by guaranteeing acceptable performance no
   matter which model is correct.

3. **Equal-blend** (`optimize_blended`) -- pool the 4 model distributions
   with equal weight (1/4 each) and optimize on the blended distribution.
   This is the simplest robustification.

4. **KS-weighted blend** (`optimize_ks_weighted`) -- like equal-blend but
   weight each model by its empirical KS p-value from Phase 3 (better-fitting
   models get more influence). This is a Bayesian model averaging approach.

All optimizations enforce realistic IPS constraints:
    - Long-only: w_i >= 0
    - Concentration cap: w_i <= MAX_WEIGHT
    - Fully invested: sum(w_i) = 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# Configuration
# ============================================================

# Maximum weight per asset (concentration limit).
# 30% is a common institutional guideline for diversified mandates.
MAX_WEIGHT: float = 0.30

# Risk-free rate for Sharpe ratio computation
RISK_FREE_RATE: float = 0.04


# ============================================================
# Portfolio statistics
# ============================================================

def portfolio_stats(
    weights: np.ndarray,
    terminal_returns: np.ndarray,
    rf: float = RISK_FREE_RATE,
    horizon_years: float = 1.0,
) -> dict:
    """
    Compute portfolio statistics from simulated terminal returns.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, shape (n_assets,).
    terminal_returns : np.ndarray
        Per-path total log returns, shape (n_paths, n_assets).
    rf : float
        Risk-free rate (annualized).
    horizon_years : float
        Length of the simulation horizon in years (used to annualize).

    Returns
    -------
    dict with keys:
        mean       -- annualized mean return (geometric, log-space)
        vol        -- annualized volatility
        sharpe     -- annualized Sharpe ratio
        sortino    -- annualized Sortino ratio
        var_95     -- 95% Value at Risk (negative number; loss)
        cvar_95    -- 95% Conditional Value at Risk
        max_loss   -- worst-case path return
    """
    # Per-path portfolio log return = w^T * (log return vector)
    port_log_returns = terminal_returns @ weights  # (n_paths,)

    # Annualize
    mean_log = port_log_returns.mean() / horizon_years
    vol_log = port_log_returns.std(ddof=1) / np.sqrt(horizon_years)

    # Convert log mean to arithmetic mean for Sharpe (more standard)
    # E[exp(X)] = exp(mean + 0.5 * var) for log-normal
    arith_mean = np.exp(mean_log + 0.5 * vol_log ** 2) - 1.0

    sharpe = (mean_log - rf) / vol_log if vol_log > 1e-10 else 0.0

    # Sortino: divide by downside deviation
    downside = port_log_returns[port_log_returns < 0]
    if len(downside) > 1:
        downside_dev = np.sqrt(np.mean(downside ** 2)) / np.sqrt(horizon_years)
        sortino = (mean_log - rf) / downside_dev if downside_dev > 1e-10 else 0.0
    else:
        sortino = float("inf")

    # VaR and CVaR at 95% (5% worst losses)
    var_95 = float(np.percentile(port_log_returns, 5))
    cvar_95 = float(port_log_returns[port_log_returns <= var_95].mean())
    max_loss = float(port_log_returns.min())

    return {
        "mean":     float(mean_log),
        "arith_mean": float(arith_mean),
        "vol":      float(vol_log),
        "sharpe":   float(sharpe),
        "sortino":  float(sortino),
        "var_95":   var_95,
        "cvar_95":  cvar_95,
        "max_loss": max_loss,
    }


# ============================================================
# Per-model optimization
# ============================================================

def _negative_sharpe(weights: np.ndarray, terminal_returns: np.ndarray, rf: float, horizon_years: float) -> float:
    """Objective for max-Sharpe: minimize -Sharpe."""
    stats = portfolio_stats(weights, terminal_returns, rf, horizon_years)
    return -stats["sharpe"]


def _build_constraints(n_assets: int, max_weight: float) -> tuple:
    """Standard IPS constraints: long-only, capped, fully invested."""
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # weights sum to 1
    ]
    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    return constraints, bounds


def optimize_max_sharpe(
    terminal_returns: np.ndarray,
    rf: float = RISK_FREE_RATE,
    horizon_years: float = 1.0,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Find max-Sharpe portfolio under a single model's distribution.
    """
    n_assets = terminal_returns.shape[1]
    constraints, bounds = _build_constraints(n_assets, max_weight)
    x0 = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        _negative_sharpe,
        x0,
        args=(terminal_returns, rf, horizon_years),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        # Fall back to equal-weight if optimizer fails (shouldn't happen with good data)
        return x0
    return result.x


# ============================================================
# Min-max Sharpe robust optimization
# ============================================================

def _negative_min_sharpe(weights: np.ndarray, returns_by_model: list, rf: float, horizon_years: float) -> float:
    """
    Objective for min-max Sharpe: minimize -(worst-case Sharpe).

    Equivalently: maximize the minimum Sharpe across the model set.
    """
    sharpes = [
        portfolio_stats(weights, ret, rf, horizon_years)["sharpe"]
        for ret in returns_by_model
    ]
    return -min(sharpes)


def optimize_minmax_sharpe(
    returns_by_model: list,
    rf: float = RISK_FREE_RATE,
    horizon_years: float = 1.0,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Find weights that maximize the worst-case Sharpe across model distributions.

    This is robust optimization: regardless of which model is "correct",
    the resulting portfolio is guaranteed to achieve at least this Sharpe.

    Parameters
    ----------
    returns_by_model : list of np.ndarray
        List of (n_paths, n_assets) terminal-return arrays, one per model.
    """
    n_assets = returns_by_model[0].shape[1]
    constraints, bounds = _build_constraints(n_assets, max_weight)
    x0 = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        _negative_min_sharpe,
        x0,
        args=(returns_by_model, rf, horizon_years),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        return x0
    return result.x


# ============================================================
# Blended-distribution optimization
# ============================================================

def blend_returns(
    returns_by_model: list,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pool terminal-return samples from multiple models with given weights.

    If `weights` is None, uses equal weights (1/M each).

    The pooling is done by sampling: from M arrays of shape (n_paths_i, n_assets),
    we concatenate. Optionally we could resample with given weights, but the
    simplest and most defensible approach is to treat the M model distributions
    as equally informative (or weighted by a prior) and concatenate.
    """
    M = len(returns_by_model)
    if weights is None:
        weights = np.full(M, 1.0 / M)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()  # normalize

    # Sample from each model proportionally to its weight
    n_total = returns_by_model[0].shape[0]  # use the same n_paths
    rng = np.random.default_rng(42)

    pieces = []
    for i, ret in enumerate(returns_by_model):
        n_take = int(round(weights[i] * n_total))
        if n_take == 0:
            continue
        # Sample without replacement if possible
        if n_take <= ret.shape[0]:
            idx = rng.choice(ret.shape[0], size=n_take, replace=False)
        else:
            idx = rng.choice(ret.shape[0], size=n_take, replace=True)
        pieces.append(ret[idx])

    if not pieces:
        # Degenerate case: all weights effectively zero; return equal blend
        return np.concatenate(returns_by_model, axis=0)
    return np.concatenate(pieces, axis=0)


def optimize_blended(
    returns_by_model: list,
    blend_weights: np.ndarray | None = None,
    rf: float = RISK_FREE_RATE,
    horizon_years: float = 1.0,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Find max-Sharpe portfolio on the blended distribution.

    `blend_weights` controls the model-blending proportions:
      - None or [1/M, ..., 1/M] = equal blend (uniform prior)
      - Bayesian: weight by KS p-values from Phase 3 (model evidence)
    """
    blended = blend_returns(returns_by_model, blend_weights)
    return optimize_max_sharpe(blended, rf, horizon_years, max_weight)


def optimize_ks_weighted(
    returns_by_model: list,
    ks_pvalues: np.ndarray,
    rf: float = RISK_FREE_RATE,
    horizon_years: float = 1.0,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Optimize on a KS-weighted blend of model distributions.

    Each model's contribution to the blend is proportional to its average
    out-of-sample KS p-value across assets. Models that fit the data
    better (higher p-value) get more weight in the optimization.
    """
    weights = np.maximum(np.asarray(ks_pvalues, dtype=float), 0.0)
    if weights.sum() < 1e-12:
        # All p-values near zero: fall back to equal blend
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    return optimize_blended(returns_by_model, weights, rf, horizon_years, max_weight)


# ============================================================
# Multi-model evaluation
# ============================================================

def evaluate_under_all_models(
    weights: np.ndarray,
    returns_by_model: dict,
    rf: float = RISK_FREE_RATE,
    horizon_years: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate one portfolio's statistics under every model's distribution.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, shape (n_assets,).
    returns_by_model : dict
        Keys are model names; values are (n_paths, n_assets) terminal returns.

    Returns
    -------
    pd.DataFrame
        Rows are model names; columns are statistics.
    """
    rows = {}
    for model_name, ret in returns_by_model.items():
        rows[model_name] = portfolio_stats(weights, ret, rf, horizon_years)
    return pd.DataFrame(rows).T
