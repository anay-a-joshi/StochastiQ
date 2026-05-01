"""
Regime-conditional portfolio evaluation.

Given a set of portfolio weights and a regime-label series, this module:

1. Reconstructs each portfolio's daily-rebalanced NAV path against actual
   historical returns (`portfolio_nav_from_weights`).
2. Computes per-regime performance metrics -- annualized return, vol,
   Sharpe, Sortino, CVaR, max drawdown, hit rate (`regime_conditional_metrics`).
3. Provides bootstrap inference on regime-conditional Sharpes
   (`bootstrap_regime_sharpe_ci`, `bootstrap_regime_sharpe_diff`).
4. Compares tail risk across portfolios within a regime
   (`regime_conditional_cvar`).

The bootstrap implementation reuses the Politis-Romano stationary
block-bootstrap from src/validation/bootstrap.py (introduced in Phase 6).
We resample only WITHIN each regime, treating regime-conditional returns
as a (potentially non-contiguous) stationary series. The block bootstrap
preserves serial dependence within the regime; the discontinuities at
regime transitions are ignored, consistent with the within-regime
inference target.

A small adapter (`_call_with_signature`) is used so this module is robust
to differences between the parameter names that bootstrap.py exposes and
the names used here. It introspects the bootstrap function's signature
and forwards only the kwargs it accepts.

The return-value of bootstrap.py functions can be either a dict or a
tuple (different versions of the project have used both). `_parse_bootstrap_result`
normalizes both shapes into a dict downstream code can rely on.

References
----------
Politis, D. N. & Romano, J. P. (1994). "The stationary bootstrap." JASA.
DeMiguel, Garlappi, & Uppal (2009). "Optimal versus naive diversification:
    how inefficient is the 1/N portfolio strategy?" Review of Financial
    Studies. (Establishes that estimation noise often dominates optimization
    gains; regime conditioning is one way to reduce that noise.)
Artzner, Delbaen, Eber & Heath (1999). "Coherent measures of risk."
    Mathematical Finance. (CVaR's coherence properties.)
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import pandas as pd

# Reuse Phase 6's stationary block-bootstrap utilities
from src.validation.bootstrap import (
    bootstrap_sharpe_ci as _bootstrap_sharpe_ci,
    bootstrap_sharpe_diff as _bootstrap_sharpe_diff,
)


# ============================================================
# Signature-tolerant bootstrap adapter
# ============================================================

# Maps from "canonical" Phase 7 parameter name to plausible alternate names.
# Adapter forwards the value to whichever alias the target function actually
# accepts. ``alpha`` is treated as a separate canonical entry because its
# *value* is different from ``confidence`` (alpha = 1 - confidence).
_PARAM_ALIASES: dict[str, list[str]] = {
    "n_boot": ["n_boot", "n_bootstrap", "n_samples", "B"],
    "block_size": ["block_size", "block_length", "expected_block_length", "L"],
    "confidence": ["confidence", "confidence_level", "conf_level"],
    "alpha": ["alpha", "alpha_level", "tail_alpha"],
    "random_state": ["random_state", "seed", "rng_seed"],
    "rf": ["rf", "risk_free_rate", "risk_free"],
    "trading_days": ["trading_days", "periods_per_year", "annualization_factor"],
}


def _call_with_signature(
    fn: Callable,
    positional: tuple,
    canonical_kwargs: dict,
):
    """Invoke ``fn`` with only those kwargs whose names (or aliases) it accepts."""
    sig = inspect.signature(fn)
    accepted_names = set(sig.parameters.keys())

    adapted: dict = {}
    for canonical_name, value in canonical_kwargs.items():
        aliases = _PARAM_ALIASES.get(canonical_name, [canonical_name])
        for alias in aliases:
            if alias in accepted_names:
                adapted[alias] = value
                break

    return fn(*positional, **adapted)


def _sharpe_local(returns: np.ndarray, rf: float, trading_days: int) -> float:
    """Annualized Sharpe of a return series, used to backfill sharpe_a/sharpe_b
    when bootstrap_sharpe_diff doesn't return them in its tuple."""
    if returns is None or len(returns) < 2:
        return float("nan")
    arr = np.asarray(returns, dtype=float)
    std = float(arr.std(ddof=1))
    if std < 1e-12:
        return float("nan")
    return float((arr.mean() - rf / trading_days) / std * np.sqrt(trading_days))


def _parse_bootstrap_result(
    result,
    kind: str,
    returns_a=None,
    returns_b=None,
    rf: float = 0.04,
    trading_days: int = 252,
) -> dict:
    """
    Normalize the return value of bootstrap_sharpe_{ci,diff} into a dict.

    Accepted shapes:
      - dict with keys point_estimate / ci_lower / ci_upper (and optionally
        p_value, sharpe_a, sharpe_b, boot_mean, boot_std)
      - tuple (point, ci_lower, ci_upper) for ci
      - tuple (point, ci_lower, ci_upper, distribution) for ci
      - tuple (point, ci_lower, ci_upper, p_value) for diff
      - tuple (point, ci_lower, ci_upper, p_value, distribution) for diff

    For diff results that don't include sharpe_a / sharpe_b, we backfill
    them locally from the input arrays so the downstream notebook code
    has a uniform interface.
    """
    # Dict path: pass through, with legacy-key normalization
    if isinstance(result, dict):
        out = dict(result)
        if "point_estimate" not in out:
            for alt in ("estimate", "sharpe", "point"):
                if alt in out:
                    out["point_estimate"] = out[alt]
                    break
        if "ci_lower" not in out:
            for alt in ("lower", "lo", "lower_bound"):
                if alt in out:
                    out["ci_lower"] = out[alt]
                    break
        if "ci_upper" not in out:
            for alt in ("upper", "hi", "upper_bound"):
                if alt in out:
                    out["ci_upper"] = out[alt]
                    break
        if kind == "diff":
            if "sharpe_a" not in out:
                out["sharpe_a"] = _sharpe_local(returns_a, rf, trading_days)
            if "sharpe_b" not in out:
                out["sharpe_b"] = _sharpe_local(returns_b, rf, trading_days)
        return out

    # Tuple path
    if isinstance(result, tuple):
        if kind == "ci":
            if len(result) >= 3:
                point, lo, hi = result[0], result[1], result[2]
                dist = result[3] if len(result) >= 4 else None
                out = {
                    "point_estimate": float(point),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                }
                if dist is not None and len(dist) > 0:
                    out["boot_mean"] = float(np.mean(dist))
                    out["boot_std"] = float(np.std(dist, ddof=1))
                return out
        elif kind == "diff":
            if len(result) >= 4:
                point, lo, hi, p_val = result[0], result[1], result[2], result[3]
                dist = result[4] if len(result) >= 5 else None
                out = {
                    "point_estimate": float(point),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "p_value": float(p_val),
                    "sharpe_a": _sharpe_local(returns_a, rf, trading_days),
                    "sharpe_b": _sharpe_local(returns_b, rf, trading_days),
                }
                if dist is not None and len(dist) > 0:
                    out["boot_mean"] = float(np.mean(dist))
                    out["boot_std"] = float(np.std(dist, ddof=1))
                return out
            elif len(result) == 3:
                # CI-style tuple but called from diff — user's diff function
                # might omit p_value. Back-compute a 2-sided p-value would
                # require the bootstrap distribution, which we don't have.
                point, lo, hi = result
                return {
                    "point_estimate": float(point),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "p_value": float("nan"),
                    "sharpe_a": _sharpe_local(returns_a, rf, trading_days),
                    "sharpe_b": _sharpe_local(returns_b, rf, trading_days),
                }

    raise TypeError(
        f"bootstrap result has unexpected shape (kind={kind}, type={type(result).__name__}, "
        f"len={len(result) if hasattr(result, '__len__') else '?'}). "
        f"Update _parse_bootstrap_result to handle this return signature."
    )


# ============================================================
# NAV construction
# ============================================================

def portfolio_nav_from_weights(
    weights: pd.Series,
    log_returns: pd.DataFrame,
    initial_value: float = 1.0,
    rebalance: str = "daily",
) -> pd.Series:
    """
    Construct a portfolio's NAV (net asset value) path from fixed weights
    and observed asset log returns under daily rebalancing.

    Parameters
    ----------
    weights : pd.Series
        Asset weights, indexed by ticker. Aligned to log_returns.columns;
        any tickers in log_returns missing from weights default to 0.
    log_returns : pd.DataFrame
        Daily log returns, columns = tickers, index = dates.
    initial_value : float
        Starting NAV (default 1.0, so terminal value is the cumulative
        growth multiple).
    rebalance : str
        "daily" : weights are rebalanced to target every day. Portfolio
                  daily simple return = sum(w_i * (exp(log_r_i) - 1)),
                  the textbook formulation for fixed-weight daily-rebalanced
                  portfolios. NAV evolves as a compound product of (1 + r).
        Other rebalancing schedules are not implemented in this version.

    Returns
    -------
    pd.Series
        NAV indexed by the same dates as log_returns. Each value is NAV at
        the end of that trading day, after the day's return.
    """
    if rebalance != "daily":
        raise NotImplementedError(
            f"rebalance='{rebalance}' is not implemented; only 'daily' is supported."
        )

    w = weights.reindex(log_returns.columns).fillna(0.0).values
    simple_returns = np.exp(log_returns.values) - 1.0
    port_simple = simple_returns @ w

    # NAV at end of day t = initial_value * prod_{s<=t} (1 + port_simple[s])
    nav_values = initial_value * np.cumprod(1.0 + port_simple)
    return pd.Series(nav_values, index=log_returns.index, name="NAV")


def portfolio_returns_from_weights(
    weights: pd.Series,
    log_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute the portfolio's daily LOG returns under fixed-weight daily
    rebalancing.

    The portfolio's daily simple return is sum(w_i * (exp(log_r_i) - 1));
    we then take log(1 + r_simple) so downstream code (Sharpe, Sortino,
    KS) can treat the series as log returns.
    """
    w = weights.reindex(log_returns.columns).fillna(0.0).values
    simple_returns = np.exp(log_returns.values) - 1.0
    port_simple = simple_returns @ w
    port_log = np.log(1.0 + port_simple)
    return pd.Series(port_log, index=log_returns.index, name="port_log_return")


# ============================================================
# Regime-conditional metrics
# ============================================================

def _compute_metrics(
    returns: np.ndarray,
    nav: np.ndarray | None,
    rf: float,
    trading_days: int,
) -> dict:
    """Inner helper: compute headline metrics on a returns / NAV slice."""
    n = len(returns)
    if n < 2:
        return {
            "n_obs": n,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "var_5": np.nan,
            "cvar_5": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }

    mean_log = float(returns.mean())
    std_log = float(returns.std(ddof=1))

    annual_return = mean_log * trading_days
    annual_vol = std_log * np.sqrt(trading_days)
    daily_rf = rf / trading_days
    sharpe = (
        (mean_log - daily_rf) / std_log * np.sqrt(trading_days)
        if std_log > 1e-12
        else np.nan
    )

    downside = returns[returns < daily_rf] - daily_rf
    if len(downside) > 1:
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        sortino = (
            (mean_log - daily_rf) / downside_std * np.sqrt(trading_days)
            if downside_std > 1e-12
            else np.nan
        )
    else:
        sortino = np.nan

    var_5 = float(np.percentile(returns, 5))
    tail = returns[returns <= var_5]
    cvar_5 = float(tail.mean()) if len(tail) > 0 else var_5

    if nav is not None and len(nav) > 1:
        running_max = np.maximum.accumulate(nav)
        drawdown = nav / running_max - 1.0
        max_dd = float(drawdown.min())
    else:
        max_dd = np.nan

    hit_rate = float((returns > 0).mean())

    return {
        "n_obs": n,
        "annualized_return": annual_return,
        "annualized_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
    }


def regime_conditional_metrics(
    portfolio_returns: pd.Series,
    portfolio_nav: pd.Series,
    regime_labels: pd.Series,
    rf: float = 0.04,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute headline performance metrics conditional on regime.

    Notes
    -----
    Regime-conditional max drawdown is computed against a *regime-local*
    NAV that resets to 1.0 at the start of each regime spell, by way of
    cumulating only the regime-conditional returns. This isolates the
    drawdown experienced *within* that regime, independent of cross-regime
    cumulative drift.
    """
    df = pd.concat(
        [
            portfolio_returns.rename("ret"),
            portfolio_nav.rename("nav"),
            regime_labels.rename("regime"),
        ],
        axis=1,
    ).dropna(subset=["ret", "regime"])

    rows: dict[str, dict] = {}
    rows["All"] = _compute_metrics(
        df["ret"].values, df["nav"].values, rf, trading_days
    )
    for regime in ["Calm", "Stress"]:
        slice_ = df[df["regime"] == regime]
        if len(slice_) > 1:
            regime_returns = slice_["ret"].values
            regime_nav = np.exp(np.cumsum(regime_returns))  # local NAV starting at 1
            rows[regime] = _compute_metrics(
                regime_returns, regime_nav, rf, trading_days
            )
        else:
            rows[regime] = _compute_metrics(
                np.array([]), None, rf, trading_days
            )

    return pd.DataFrame(rows).T


# ============================================================
# Bootstrap inference within a regime
# ============================================================

def bootstrap_regime_sharpe_ci(
    portfolio_returns: pd.Series,
    regime_labels: pd.Series,
    regime: str,
    rf: float = 0.04,
    trading_days: int = 252,
    n_boot: int = 5000,
    block_size: int | None = None,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Stationary block-bootstrap CI for the Sharpe ratio of one portfolio
    within one regime.

    Filters portfolio_returns to the regime-conditional subseries, then
    delegates the actual block-bootstrap to bootstrap.py's
    ``bootstrap_sharpe_ci`` using a signature-tolerant adapter.

    Block size defaults to len(slice)^(1/3) (Politis-Romano rule of thumb).
    """
    df = pd.concat(
        [portfolio_returns.rename("ret"), regime_labels.rename("regime")],
        axis=1,
    ).dropna()
    slice_returns = df.loc[df["regime"] == regime, "ret"].values

    if len(slice_returns) < 30:
        return {
            "regime": regime,
            "n_obs": int(len(slice_returns)),
            "point_estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "boot_mean": np.nan,
            "boot_std": np.nan,
        }

    if block_size is None:
        block_size = max(2, int(round(len(slice_returns) ** (1.0 / 3))))

    canonical_kwargs = {
        "rf": rf,
        "trading_days": trading_days,
        "n_boot": n_boot,
        "block_size": block_size,
        "confidence": confidence,
        "alpha": 1.0 - confidence,
        "random_state": random_state,
    }
    result = _call_with_signature(
        _bootstrap_sharpe_ci,
        positional=(slice_returns,),
        canonical_kwargs=canonical_kwargs,
    )

    out = _parse_bootstrap_result(
        result,
        kind="ci",
        rf=rf,
        trading_days=trading_days,
    )
    out["regime"] = regime
    out["n_obs"] = int(len(slice_returns))
    return out


def bootstrap_regime_sharpe_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    regime_labels: pd.Series,
    regime: str,
    rf: float = 0.04,
    trading_days: int = 252,
    n_boot: int = 5000,
    block_size: int | None = None,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Paired stationary block-bootstrap test for Sharpe(A) - Sharpe(B) within
    a regime.

    Uses the same time-aligned bootstrap indices for both portfolios on each
    iteration so that the resulting CI on the Sharpe DIFFERENCE is correctly
    pair-correlated. This is the headline statistical test for Phase 7.
    """
    df = pd.concat(
        [
            returns_a.rename("a"),
            returns_b.rename("b"),
            regime_labels.rename("regime"),
        ],
        axis=1,
    ).dropna()
    slice_ = df[df["regime"] == regime]
    a = slice_["a"].values
    b = slice_["b"].values

    if len(a) < 30:
        return {
            "regime": regime,
            "n_obs": int(len(a)),
            "point_estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "sharpe_a": np.nan,
            "sharpe_b": np.nan,
        }

    if block_size is None:
        block_size = max(2, int(round(len(a) ** (1.0 / 3))))

    canonical_kwargs = {
        "rf": rf,
        "trading_days": trading_days,
        "n_boot": n_boot,
        "block_size": block_size,
        "confidence": confidence,
        "alpha": 1.0 - confidence,
        "random_state": random_state,
    }
    result = _call_with_signature(
        _bootstrap_sharpe_diff,
        positional=(a, b),
        canonical_kwargs=canonical_kwargs,
    )

    out = _parse_bootstrap_result(
        result,
        kind="diff",
        returns_a=a,
        returns_b=b,
        rf=rf,
        trading_days=trading_days,
    )
    out["regime"] = regime
    out["n_obs"] = int(len(a))
    return out


# ============================================================
# Tail risk comparison
# ============================================================

def regime_conditional_cvar(
    portfolio_returns_dict: dict[str, pd.Series],
    regime_labels: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compare CVaR across multiple portfolios within each regime.

    For each portfolio, computes empirical VaR_alpha and CVaR_alpha within
    All / Calm / Stress slices, where alpha = 1 - confidence (e.g. 5% tail).

    CVaR is the conditional expectation of losses beyond VaR -- a coherent
    risk measure (Artzner et al. 1999) and the natural risk metric to
    evaluate when stochastic models predict tail behavior.
    """
    alpha_pct = (1 - confidence) * 100
    rows = []
    for name, returns in portfolio_returns_dict.items():
        df = pd.concat(
            [returns.rename("ret"), regime_labels.rename("regime")], axis=1
        ).dropna()
        for regime in ["All", "Calm", "Stress"]:
            if regime == "All":
                r = df["ret"].values
            else:
                r = df.loc[df["regime"] == regime, "ret"].values
            if len(r) < 2:
                rows.append(
                    {
                        "portfolio": name,
                        "regime": regime,
                        "n_obs": int(len(r)),
                        "VaR": np.nan,
                        "CVaR": np.nan,
                    }
                )
                continue
            var_val = float(np.percentile(r, alpha_pct))
            tail = r[r <= var_val]
            cvar_val = float(tail.mean()) if len(tail) > 0 else var_val
            rows.append(
                {
                    "portfolio": name,
                    "regime": regime,
                    "n_obs": int(len(r)),
                    "VaR": var_val,
                    "CVaR": cvar_val,
                }
            )

    return pd.DataFrame(rows).set_index(["portfolio", "regime"])
