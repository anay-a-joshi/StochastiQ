"""
src/validation/bootstrap.py — Statistical-power tools for OOS validation.

This module provides:
  - Politis-Romano (1994) stationary block bootstrap.
  - Bootstrap confidence intervals on annualized Sharpe ratios.
  - Bootstrap difference-of-Sharpes test for two correlated portfolios.
  - Train-vs-test KS p-value comparison (Spearman rank, Wilcoxon signed-rank).

These tools quantify the statistical power of single-window OOS comparisons.
With a typical OOS window of 200-400 trading days, the standard error on a
realized Sharpe is roughly 0.7-1.0, meaning many "headline" Sharpe gaps
between portfolios are within bootstrap noise.

References:
  Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
    Journal of the American Statistical Association, 89(428), 1303-1313.
  DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive
    diversification: How inefficient is the 1/N portfolio strategy?
    Review of Financial Studies, 22(5), 1915-1953.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


# ---------------------------------------------------------------------------
# Stationary block bootstrap
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(
    n: int,
    block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Politis-Romano (1994) stationary block bootstrap index sequence.

    Generates an index sequence of length n where each new block starts
    with probability p = 1 / block_length, otherwise continues the
    previous block (with circular wrap-around). This preserves serial
    dependence in financial returns better than i.i.d. bootstrap and
    avoids the boundary issues of fixed-block bootstrap.

    Parameters
    ----------
    n : int
        Length of the sequence to generate.
    block_length : float
        Expected block length. Larger values preserve more serial
        dependence; rule-of-thumb is ~ n^(1/3).
    rng : np.random.Generator
        Numpy random generator (for reproducibility).

    Returns
    -------
    indices : np.ndarray of shape (n,)
        Integer indices in [0, n).
    """
    if block_length < 1.0:
        raise ValueError("block_length must be >= 1")

    p = 1.0 / float(block_length)
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)

    # Vectorize the "new block?" decisions for speed.
    new_block = rng.random(n - 1) < p
    new_starts = rng.integers(0, n, size=n - 1)

    for i in range(1, n):
        if new_block[i - 1]:
            indices[i] = new_starts[i - 1]
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def _annualized_sharpe(daily_returns: np.ndarray, rf_daily: float, trading_days: int) -> float:
    """Annualized Sharpe from a daily return series."""
    excess = daily_returns - rf_daily
    sd = daily_returns.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(trading_days))


def bootstrap_sharpe_ci(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
    n_boot: int = 10_000,
    block_length: Optional[float] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Stationary block-bootstrap confidence interval on the annualized Sharpe.

    Parameters
    ----------
    daily_returns : array-like of shape (T,)
        Daily simple or log returns of the portfolio.
    risk_free_rate : float
        Annualized risk-free rate (default 4%).
    trading_days : int
        Trading days per year for annualization (default 252).
    n_boot : int
        Number of bootstrap resamples (default 10,000).
    block_length : float, optional
        Stationary-bootstrap expected block length. Defaults to
        max(2, T^(1/3)).
    alpha : float
        Significance level (default 0.05 -> 95% CI).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    point : float
        Point-estimate annualized Sharpe (computed from the original sample).
    lo, hi : float
        Lower and upper bounds of the (1-alpha) percentile CI.
    boot_dist : np.ndarray
        Full bootstrap distribution of Sharpes (for plotting/diagnostics).
    """
    rets = np.asarray(daily_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    if n < 30:
        raise ValueError(f"Series too short for bootstrap (T={n}).")

    if block_length is None:
        block_length = max(2.0, float(n) ** (1.0 / 3.0))

    rf_daily = risk_free_rate / trading_days
    rng = np.random.default_rng(seed)

    point = _annualized_sharpe(rets, rf_daily, trading_days)

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        boot[b] = _annualized_sharpe(rets[idx], rf_daily, trading_days)

    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), boot


def bootstrap_sharpe_diff(
    daily_returns_a: np.ndarray,
    daily_returns_b: np.ndarray,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
    n_boot: int = 10_000,
    block_length: Optional[float] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Bootstrap distribution of (Sharpe_A - Sharpe_B) for two return series
    measured over the *same* dates. Uses paired resampling (same indices
    on both series) to preserve cross-portfolio correlation, which yields
    a much tighter CI than independent bootstraps for two highly correlated
    portfolios (the standard case here, since both are weighted combinations
    of the same 7 underlyings).

    Parameters
    ----------
    daily_returns_a, daily_returns_b : array-like of shape (T,)
        Aligned daily returns for the two portfolios.
    risk_free_rate, trading_days, n_boot, block_length, alpha, seed
        See `bootstrap_sharpe_ci`.

    Returns
    -------
    point_diff : float
        Sharpe_A - Sharpe_B from the original (un-resampled) data.
    lo, hi : float
        (1-alpha) percentile CI for the difference.
    p_value : float
        Two-sided bootstrap p-value for H0: Sharpe_A == Sharpe_B.
    boot_diffs : np.ndarray
        Full bootstrap distribution of the differences.
    """
    a = np.asarray(daily_returns_a, dtype=float)
    b = np.asarray(daily_returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Series must have identical length.")

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 30:
        raise ValueError(f"Series too short for bootstrap (T={n}).")

    if block_length is None:
        block_length = max(2.0, float(n) ** (1.0 / 3.0))

    rf_daily = risk_free_rate / trading_days
    rng = np.random.default_rng(seed)

    sharpe_a = _annualized_sharpe(a, rf_daily, trading_days)
    sharpe_b = _annualized_sharpe(b, rf_daily, trading_days)
    point_diff = sharpe_a - sharpe_b

    boot_diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        boot_diffs[k] = (
            _annualized_sharpe(a[idx], rf_daily, trading_days)
            - _annualized_sharpe(b[idx], rf_daily, trading_days)
        )

    lo, hi = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided p-value: 2 * min(P(diff <= 0), P(diff >= 0)) under the
    # bootstrap distribution centered at the observed diff. Equivalent to
    # asking how often a recentered-at-zero bootstrap exceeds |observed|.
    centered = boot_diffs - point_diff
    p_value = float(np.mean(np.abs(centered) >= abs(point_diff)))
    return float(point_diff), float(lo), float(hi), p_value, boot_diffs


# ---------------------------------------------------------------------------
# KS train-vs-test comparison
# ---------------------------------------------------------------------------

def ks_train_test_comparison(
    oos_ks_df: pd.DataFrame,
    train_ks_df: pd.DataFrame,
    asset_col: str = "Asset",
    model_col: str = "Model",
    oos_pval_col: str = "OOS_KS_pvalue",
    train_pval_col: str = "Train_KS_pvalue",
) -> dict:
    """
    Quantify whether OOS KS p-values represent a degradation from the
    training-window KS p-values, or merely reflect a structural property
    of the calibrated models that holds across both windows.

    Two complementary tests:

    1. Spearman rank correlation between train and OOS p-values across all
       (asset, model) pairs. A positive correlation indicates that models
       which fit poorly in-sample also fit poorly OOS — the *ranking* is
       preserved, which is what we expect under a stable calibration that
       has structural limits (e.g., parametric continuous-path models
       cannot reproduce empirical fat tails — Cont, 2001).

    2. Wilcoxon signed-rank test on the paired differences (OOS - Train).
       The null is that the median paired difference is zero — i.e., that
       OOS p-values are not systematically lower (no degradation). A high
       p-value supports stability.

    Parameters
    ----------
    oos_ks_df : DataFrame
        Long-form table with columns [Asset, Model, OOS_KS_pvalue].
    train_ks_df : DataFrame
        Long-form table with columns [Asset, Model, Train_KS_pvalue].

    Returns
    -------
    dict with keys:
        spearman_rho, spearman_p,
        wilcoxon_stat, wilcoxon_p,
        n_pairs,
        mean_train_p, mean_oos_p,
        merged (DataFrame).
    """
    merged = oos_ks_df.merge(train_ks_df, on=[asset_col, model_col])
    if merged.empty:
        raise ValueError("Train/OOS KS tables share no (Asset, Model) pairs.")

    train_p = merged[train_pval_col].astype(float).to_numpy()
    oos_p = merged[oos_pval_col].astype(float).to_numpy()

    rho, rho_p = spearmanr(train_p, oos_p)
    diff = oos_p - train_p

    # Wilcoxon errors out if all diffs are zero; guard for that edge case.
    if np.allclose(diff, 0.0):
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_stat, w_p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "n_pairs": int(len(merged)),
        "mean_train_p": float(train_p.mean()),
        "mean_oos_p": float(oos_p.mean()),
        "merged": merged,
    }


# ---------------------------------------------------------------------------
# Return attribution
# ---------------------------------------------------------------------------

def return_attribution(
    weights_a: pd.Series,
    weights_b: pd.Series,
    asset_returns: pd.Series,
) -> pd.DataFrame:
    """
    Decompose the realized return gap (Portfolio A - Portfolio B) into
    per-asset contributions arising from differences in weight allocation.

    Under buy-and-hold, the period return of a portfolio with initial
    weights w is approximately sum_i w_i * R_i (exact to first order, with
    drift effects of order R_i^2). The gap between two portfolios over the
    same period therefore decomposes cleanly as:

        Gap = R_A - R_B = sum_i (w_a_i - w_b_i) * R_i

    Each term is the "contribution" of asset i: positive when A overweights
    a winning asset relative to B, or underweights a losing one.

    Parameters
    ----------
    weights_a, weights_b : Series indexed by asset
        Initial portfolio weights. Must share the same index.
    asset_returns : Series indexed by asset
        Realized total returns over the buy-and-hold period (e.g.,
        cumulative simple return).

    Returns
    -------
    DataFrame with columns:
        weight_a, weight_b, delta_weight, asset_return, contribution.
        Sorted by absolute contribution (largest drivers first).
    """
    idx = weights_a.index
    if not idx.equals(weights_b.index) or not idx.equals(asset_returns.index):
        raise ValueError("weights_a, weights_b, asset_returns must share the same index.")

    delta_w = weights_a - weights_b
    contrib = delta_w * asset_returns

    out = pd.DataFrame({
        "weight_a": weights_a,
        "weight_b": weights_b,
        "delta_weight": delta_w,
        "asset_return": asset_returns,
        "contribution": contrib,
    })
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
