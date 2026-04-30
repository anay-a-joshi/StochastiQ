"""
Portfolio optimization module for the StochastiQ project.

Implements four optimal portfolios under different objective functions:
    1. Maximum Sharpe ratio (tangency portfolio)
    2. Minimum variance
    3. Maximum Sortino ratio (downside-deviation-adjusted)
    4. Minimum Conditional Value at Risk (CVaR) at 95%
    5. Risk parity (equal risk contribution)

Also computes the full efficient frontier for visualization.

All optimizations use long-only constraints (weights >= 0, sum to 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# Data classes
# ============================================================

@dataclass
class Portfolio:
    """Container for a single optimized portfolio's results."""
    name: str
    weights: pd.Series
    expected_return: float    # annualized
    volatility: float         # annualized
    sharpe_ratio: float
    sortino_ratio: float
    cvar_95: float            # 95% Conditional Value at Risk (positive = loss)
    var_95: float             # 95% Value at Risk (positive = loss)


# ============================================================
# Helper: portfolio statistics
# ============================================================

def _portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252,
) -> tuple[float, float]:
    """Compute annualized expected return and volatility for given weights."""
    annual_return = np.dot(weights, mean_returns) * trading_days
    annual_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)) * trading_days)
    return annual_return, annual_vol


def _downside_deviation(
    weights: np.ndarray,
    returns: np.ndarray,
    target: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualized downside deviation: std of returns below the target."""
    portfolio_returns = returns @ weights
    downside = np.minimum(portfolio_returns - target, 0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(trading_days)


def _historical_var_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Historical (non-parametric) VaR and CVaR at the given confidence.

    Returns positive numbers representing losses (e.g. 0.03 = 3% loss).
    """
    portfolio_returns = returns @ weights
    var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
    tail_returns = portfolio_returns[portfolio_returns <= -var]
    cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var
    return var, cvar


# ============================================================
# Objective functions (to minimize)
# ============================================================

def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free, trading_days):
    """Negative Sharpe (minimized = Sharpe maximized)."""
    ret, vol = _portfolio_stats(weights, mean_returns, cov_matrix, trading_days)
    return -(ret - risk_free) / vol if vol > 0 else 1e6


def _portfolio_variance(weights, mean_returns, cov_matrix, trading_days):
    """Portfolio variance (minimized = min-variance)."""
    return np.dot(weights, np.dot(cov_matrix, weights)) * trading_days


def _neg_sortino(weights, returns_array, mean_returns, risk_free, trading_days):
    """Negative Sortino (minimized = Sortino maximized)."""
    annual_return = np.dot(weights, mean_returns) * trading_days
    daily_target = risk_free / trading_days
    dd = _downside_deviation(weights, returns_array, target=daily_target, trading_days=trading_days)
    return -(annual_return - risk_free) / dd if dd > 0 else 1e6


def _cvar_objective(weights, returns_array, confidence):
    """CVaR at the given confidence level (minimized = min-CVaR)."""
    _, cvar = _historical_var_cvar(weights, returns_array, confidence)
    return cvar


# ============================================================
# Risk parity objective
# ============================================================

def _risk_parity_objective(weights, cov_matrix):
    """
    Sum of squared deviations of each asset's risk contribution from the equal share.

    Each asset's risk contribution: w_i * (Σw)_i / sqrt(w'Σw)
    Equal share = total portfolio vol / N
    """
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    if portfolio_vol < 1e-10:
        return 1e6
    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / portfolio_vol
    target = portfolio_vol / len(weights)
    return np.sum((risk_contrib - target) ** 2)


# ============================================================
# Generic optimizer wrapper
# ============================================================

def _optimize(
    objective: Callable,
    n_assets: int,
    args: tuple,
    initial_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run scipy.optimize.minimize with long-only, fully-invested constraints.

    Constraints:
        - sum(weights) = 1
        - 0 <= weights <= 1 for each asset
    """
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    result = minimize(
        objective,
        initial_weights,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )

    if not result.success:
        # Try once more with a perturbed starting point
        perturbed = initial_weights + np.random.uniform(-0.05, 0.05, n_assets)
        perturbed = np.clip(perturbed, 0, 1)
        perturbed = perturbed / perturbed.sum()
        result = minimize(
            objective, perturbed, args=args,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

    return result.x


# ============================================================
# Build a Portfolio object from raw weights
# ============================================================

def _build_portfolio(
    name: str,
    weights: np.ndarray,
    asset_names: list[str],
    log_returns: pd.DataFrame,
    risk_free_rate: float,
    trading_days: int,
) -> Portfolio:
    """Compute all summary statistics and package into a Portfolio object."""
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    returns_array = log_returns.values

    annual_return, annual_vol = _portfolio_stats(weights, mean_returns, cov_matrix, trading_days)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    daily_target = risk_free_rate / trading_days
    dd = _downside_deviation(weights, returns_array, target=daily_target, trading_days=trading_days)
    sortino = (annual_return - risk_free_rate) / dd if dd > 0 else 0.0

    var_95, cvar_95 = _historical_var_cvar(weights, returns_array, confidence=0.95)

    return Portfolio(
        name=name,
        weights=pd.Series(weights, index=asset_names, name=name),
        expected_return=annual_return,
        volatility=annual_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        cvar_95=cvar_95,
        var_95=var_95,
    )


# ============================================================
# Public API: portfolio constructors
# ============================================================

def max_sharpe_portfolio(
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> Portfolio:
    """Tangency portfolio: maximizes (E[r] - rf) / sigma."""
    n = log_returns.shape[1]
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values

    weights = _optimize(
        _neg_sharpe,
        n,
        args=(mean_returns, cov_matrix, risk_free_rate, trading_days),
    )
    return _build_portfolio(
        "Max Sharpe", weights, log_returns.columns.tolist(),
        log_returns, risk_free_rate, trading_days,
    )


def min_variance_portfolio(
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> Portfolio:
    """Minimum-variance portfolio."""
    n = log_returns.shape[1]
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values

    weights = _optimize(
        _portfolio_variance,
        n,
        args=(mean_returns, cov_matrix, trading_days),
    )
    return _build_portfolio(
        "Min Variance", weights, log_returns.columns.tolist(),
        log_returns, risk_free_rate, trading_days,
    )


def max_sortino_portfolio(
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> Portfolio:
    """Maximizes (E[r] - rf) / downside_deviation."""
    n = log_returns.shape[1]
    mean_returns = log_returns.mean().values
    returns_array = log_returns.values

    weights = _optimize(
        _neg_sortino,
        n,
        args=(returns_array, mean_returns, risk_free_rate, trading_days),
    )
    return _build_portfolio(
        "Max Sortino", weights, log_returns.columns.tolist(),
        log_returns, risk_free_rate, trading_days,
    )


def min_cvar_portfolio(
    log_returns: pd.DataFrame,
    confidence: float = 0.95,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> Portfolio:
    """Minimizes CVaR at the given confidence level."""
    n = log_returns.shape[1]
    returns_array = log_returns.values

    weights = _optimize(
        _cvar_objective,
        n,
        args=(returns_array, confidence),
    )
    return _build_portfolio(
        f"Min CVaR ({int(confidence*100)}%)", weights, log_returns.columns.tolist(),
        log_returns, risk_free_rate, trading_days,
    )


def risk_parity_portfolio(
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> Portfolio:
    """Risk parity: each asset contributes equal risk."""
    n = log_returns.shape[1]
    cov_matrix = log_returns.cov().values

    # Inverse-volatility starting point gives the optimizer a head start
    inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
    initial = inv_vol / inv_vol.sum()

    weights = _optimize(
        _risk_parity_objective,
        n,
        args=(cov_matrix,),
        initial_weights=initial,
    )
    return _build_portfolio(
        "Risk Parity", weights, log_returns.columns.tolist(),
        log_returns, risk_free_rate, trading_days,
    )


# ============================================================
# Efficient Frontier
# ============================================================

def efficient_frontier(
    log_returns: pd.DataFrame,
    n_points: int = 50,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by solving min-variance for a sweep of
    target returns between min-asset-return and max-asset-return.

    Returns a DataFrame with columns: target_return, volatility, sharpe.
    """
    mean_returns = log_returns.mean().values * trading_days
    cov_matrix = log_returns.cov().values * trading_days
    n = len(mean_returns)

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier_vols = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        bounds = tuple((0.0, 1.0) for _ in range(n))
        x0 = np.ones(n) / n

        result = minimize(
            lambda w: np.sqrt(w @ cov_matrix @ w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 300},
        )
        frontier_vols.append(result.fun if result.success else np.nan)

    return pd.DataFrame({
        "target_return": target_returns,
        "volatility": frontier_vols,
    }).dropna()


# ============================================================
# Convenience: build all portfolios at once
# ============================================================

def build_all_portfolios(
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> dict[str, Portfolio]:
    """Construct all five portfolios and return as a dict."""
    return {
        "max_sharpe":   max_sharpe_portfolio(log_returns, risk_free_rate, trading_days),
        "min_variance": min_variance_portfolio(log_returns, risk_free_rate, trading_days),
        "max_sortino":  max_sortino_portfolio(log_returns, risk_free_rate, trading_days),
        "min_cvar":     min_cvar_portfolio(log_returns, 0.95, risk_free_rate, trading_days),
        "risk_parity":  risk_parity_portfolio(log_returns, risk_free_rate, trading_days),
    }


def portfolios_summary_table(portfolios: dict[str, Portfolio]) -> pd.DataFrame:
    """Build a side-by-side comparison table of all portfolios."""
    rows = []
    for portfolio in portfolios.values():
        row = {
            "Portfolio": portfolio.name,
            "Annual Return": portfolio.expected_return,
            "Annual Volatility": portfolio.volatility,
            "Sharpe Ratio": portfolio.sharpe_ratio,
            "Sortino Ratio": portfolio.sortino_ratio,
            "VaR 95%": portfolio.var_95,
            "CVaR 95%": portfolio.cvar_95,
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("Portfolio")


def weights_comparison_table(portfolios: dict[str, Portfolio]) -> pd.DataFrame:
    """Build a table comparing weights across all portfolios."""
    return pd.DataFrame({
        portfolio.name: portfolio.weights
        for portfolio in portfolios.values()
    })
