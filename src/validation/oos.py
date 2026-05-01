"""
Out-of-sample (OOS) validation utilities for Phase 6.

Provides functions to evaluate Phase 4 portfolio weights and Phase 5 overlay
strategies against historical test-window data. The central design principle:
load saved Phase 3/4/5 outputs and check whether their predictions
materialized on data the calibration never saw.

Functions
---------
compute_portfolio_nav
    Buy-and-hold daily NAV from initial weights and price series.
compute_realized_metrics
    Sharpe, max drawdown, VaR, CVaR, hit rate from a NAV series.
compute_overlay_realized_pnl
    Realized terminal P&L for a portfolio with per-asset overlays.
forecast_realized_percentile
    Where does the realized value fall in a simulated distribution?
out_of_sample_ks_test
    Two-sample KS test between simulated and realized return distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.options.black_scholes import call_price, put_price
from src.options.strategies import CoveredCall, ProtectivePut, Collar


# ============================================================
# Portfolio NAV construction (buy-and-hold)
# ============================================================

def compute_portfolio_nav(
    weights: np.ndarray,
    prices: pd.DataFrame,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Compute buy-and-hold portfolio NAV from initial weights and price series.

    Initial shares are set such that weights * initial_value is allocated
    to each asset on the first date. Holdings are NOT rebalanced -- weights
    drift naturally with relative price moves, which is the realistic
    institutional benchmark for an annual or multi-year portfolio.

    Parameters
    ----------
    weights : np.ndarray, shape (n_assets,)
        Portfolio weights at inception. Must sum to 1.0 (or close).
    prices : pd.DataFrame
        Daily price series. Columns must match the order of ``weights``.
    initial_value : float
        Initial portfolio dollar value.

    Returns
    -------
    pd.Series
        Daily portfolio NAV indexed by date.
    """
    weights = np.asarray(weights, dtype=float)
    if prices.empty:
        return pd.Series([], dtype=float, name="NAV")
    initial_prices = prices.iloc[0].values
    shares = (weights * initial_value) / initial_prices
    nav_values = (prices.values * shares).sum(axis=1)
    return pd.Series(nav_values, index=prices.index, name="NAV")


# ============================================================
# Realized performance metrics
# ============================================================

def compute_realized_metrics(
    nav_series: pd.Series,
    rf: float,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute realized performance metrics from a daily NAV series.

    All metrics are *realized* (not forecast). Sharpe is annualized using
    daily log returns (geometric basis), drawdown is peak-to-trough on
    the NAV path, and VaR/CVaR are computed at the daily-return level.

    Parameters
    ----------
    nav_series : pd.Series
        Daily NAV (length n; the NAV on the first date is the starting value).
    rf : float
        Annualized risk-free rate (used in Sharpe computation).
    periods_per_year : int
        Trading days per year for annualization (default 252).

    Returns
    -------
    dict with keys:
        cumulative_return, annualized_return, annualized_vol,
        sharpe, max_drawdown, var_5, cvar_5, hit_rate, n_obs
    """
    daily_log_rets = np.log(nav_series / nav_series.shift(1)).dropna()
    n = len(daily_log_rets)
    if n == 0:
        return {k: np.nan for k in [
            "cumulative_return", "annualized_return", "annualized_vol",
            "sharpe", "max_drawdown", "var_5", "cvar_5", "hit_rate", "n_obs",
        ]}

    cumulative_log = float(daily_log_rets.sum())
    cumulative_return = float(np.exp(cumulative_log) - 1.0)
    annualized_return = float(daily_log_rets.mean() * periods_per_year)
    annualized_vol = float(daily_log_rets.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = (
        (annualized_return - rf) / annualized_vol
        if annualized_vol > 0 else float("nan")
    )

    # Drawdown
    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1.0
    max_drawdown = float(drawdown.min())

    # Tail metrics (daily basis)
    var_5 = float(np.percentile(daily_log_rets, 5))
    cvar_5 = float(daily_log_rets[daily_log_rets <= var_5].mean())
    hit_rate = float((daily_log_rets > 0).mean())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_vol":    annualized_vol,
        "sharpe":            float(sharpe),
        "max_drawdown":      max_drawdown,
        "var_5":             var_5,
        "cvar_5":            cvar_5,
        "hit_rate":          hit_rate,
        "n_obs":             n,
    }


# ============================================================
# Options overlay realized P&L
# ============================================================

def compute_overlay_realized_pnl(
    S0_vec: np.ndarray,
    S_T_vec: np.ndarray,
    weights: np.ndarray,
    strategy_per_asset: list,
    sigma_vec: np.ndarray,
    T: float,
    r: float,
    q_vec: np.ndarray | None = None,
) -> dict:
    """
    Realized terminal portfolio P&L under a per-asset overlay assignment.

    For each asset, the realized terminal value per share is:
      - None (unhedged):     S_T_i
      - Covered call:        min(S_T_i, K_call) + accrued call premium
      - Protective put:      max(S_T_i, K_put) - accrued put premium
      - Collar:              max(min(S_T_i, K_call), K_put) - accrued net premium

    Premiums are computed via BSM using ``sigma_vec`` (typically Phase 3
    GBM sigma) and grown at the risk-free rate to T to put them on the
    same time-of-money basis as the terminal asset price.

    Parameters
    ----------
    S0_vec : np.ndarray, shape (n_assets,)
        Initial prices on the strategy entry date.
    S_T_vec : np.ndarray, shape (n_assets,)
        Realized prices at expiry.
    weights : np.ndarray, shape (n_assets,)
        Portfolio weights (must sum to 1).
    strategy_per_asset : list of length n_assets
        Each entry is None or one of CoveredCall, ProtectivePut, Collar.
    sigma_vec : np.ndarray, shape (n_assets,)
        BSM volatility input per asset.
    T : float
        Tenor in years.
    r : float
        Risk-free rate.
    q_vec : np.ndarray, optional
        Continuous dividend yield per asset (default zeros).

    Returns
    -------
    dict with keys:
        portfolio_terminal_value : weighted basket terminal value (V_T / V_0)
        portfolio_log_return     : log of the above
        per_asset_terminal       : realized terminal value per share, by asset
    """
    n = len(S0_vec)
    if q_vec is None:
        q_vec = np.zeros(n)

    per_asset_value = np.empty(n)
    for i in range(n):
        S0 = float(S0_vec[i])
        S_T = float(S_T_vec[i])
        sig = float(sigma_vec[i])
        q = float(q_vec[i])
        strat = strategy_per_asset[i]

        if strat is None:
            per_asset_value[i] = S_T
        elif isinstance(strat, CoveredCall):
            K_c = strat.call_strike_mult * S0
            c_p = float(call_price(S0, K_c, T, r, sig, q))
            per_asset_value[i] = min(S_T, K_c) + c_p * float(np.exp(r * T))
        elif isinstance(strat, ProtectivePut):
            K_p = strat.put_strike_mult * S0
            p_p = float(put_price(S0, K_p, T, r, sig, q))
            per_asset_value[i] = max(S_T, K_p) - p_p * float(np.exp(r * T))
        elif isinstance(strat, Collar):
            K_c = strat.call_strike_mult * S0
            K_p = strat.put_strike_mult * S0
            c_p = float(call_price(S0, K_c, T, r, sig, q))
            p_p = float(put_price(S0, K_p, T, r, sig, q))
            net_premium = p_p - c_p
            per_asset_value[i] = max(min(S_T, K_c), K_p) - net_premium * float(np.exp(r * T))
        else:
            raise ValueError(f"Unknown strategy type: {type(strat).__name__}")

    price_ratios = per_asset_value / S0_vec
    portfolio_terminal_value = float((weights * price_ratios).sum())
    portfolio_log_return = float(np.log(max(portfolio_terminal_value, 1e-12)))

    return {
        "portfolio_terminal_value": portfolio_terminal_value,
        "portfolio_log_return":     portfolio_log_return,
        "per_asset_terminal":       per_asset_value,
    }


# ============================================================
# Distribution diagnostics
# ============================================================

def forecast_realized_percentile(
    realized: float,
    simulated_distribution: np.ndarray,
) -> float:
    """
    Percentile of the realized value within a simulated distribution.

    A well-calibrated forecast distribution should produce realized
    percentiles uniformly distributed on [0, 100] across many independent
    forecasts. Systematic deviations indicate bias (mean wrong) or
    miscalibrated dispersion (variance wrong).

    Returns
    -------
    Percentile in [0, 100], or 0 / 100 at the bounds.
    """
    return float((np.asarray(simulated_distribution) <= float(realized)).mean() * 100.0)


def out_of_sample_ks_test(
    realized_returns: np.ndarray,
    simulated_returns: np.ndarray,
) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test between simulated and realized
    return distributions.

    Null hypothesis: the two samples are drawn from the same distribution.
    A small p-value (< 0.05) rejects the null -- the calibrated model is
    statistically distinguishable from the realized data on this window.

    Returns
    -------
    (statistic, p_value)
    """
    realized = np.asarray(realized_returns)
    simulated = np.asarray(simulated_returns)
    result = ks_2samp(realized, simulated)
    return float(result.statistic), float(result.pvalue)


# ============================================================
# Convenience: prediction interval coverage
# ============================================================

def prediction_interval_coverage(
    realized_returns: np.ndarray,
    simulated_returns: np.ndarray,
    levels: tuple = (0.50, 0.80, 0.90, 0.95),
) -> dict:
    """
    For each confidence level, compute the empirical coverage of the
    simulated-distribution prediction interval over the realized data.

    A well-calibrated model should produce coverage close to the nominal
    level. e.g., the 90% prediction interval should contain 90% of the
    realized observations.

    Parameters
    ----------
    realized_returns : np.ndarray
        Observed daily/weekly/etc. returns on the test window.
    simulated_returns : np.ndarray
        Simulated returns from the model (any sample size).
    levels : tuple of float
        Confidence levels to evaluate.

    Returns
    -------
    dict mapping level -> empirical_coverage (fraction in [0, 1]).
    """
    realized = np.asarray(realized_returns)
    simulated = np.asarray(simulated_returns)
    coverage = {}
    for lvl in levels:
        alpha = (1.0 - lvl) / 2.0
        lo = float(np.quantile(simulated, alpha))
        hi = float(np.quantile(simulated, 1.0 - alpha))
        coverage[lvl] = float(((realized >= lo) & (realized <= hi)).mean())
    return coverage
