"""
Options overlay strategies built on Black-Scholes-Merton pricing.

Three strategies:

    1. Covered Call    -- long stock + short OTM call
                         Income generation; caps upside above strike.

    2. Protective Put  -- long stock + long OTM put
                         Insurance against downside; pay premium.

    3. Collar          -- long stock + short OTM call + long OTM put
                         Bounded P&L; often near-zero net premium.

Each strategy is parameterized by:
    - Underlying price S0 and weight in the portfolio
    - Strike multipliers (e.g., 1.05 * S0 for OTM call, 0.95 * S0 for OTM put)
    - Tenor T in years
    - GBM volatility sigma (from Phase 3 calibration)
    - Risk-free rate r and dividend yield q

Outputs include:
    - Strategy entry cost / income (option premiums)
    - Terminal payoff distribution given simulated stock paths
    - Net Greeks at strategy inception (portfolio-level if weights provided)

Strategies are evaluated at the asset level: each holding can have its
own overlay. This is more rigorous than index-level proxies and lets us
compute proper portfolio Greeks by aggregating asset-level Greeks
weighted by holding sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.options.black_scholes import (
    call_price, put_price, delta, gamma, vega, theta, rho,
)


# ============================================================
# Strategy definitions
# ============================================================

@dataclass
class CoveredCall:
    """
    Long stock + short OTM call.

    At inception:
        Cost = S0 (stock) - call_premium (income from short call)

    At expiry:
        Payoff = min(S_T, K_call) + (call_premium accrued)
        Effectively: capped at K_call, with collected premium as income.
    """
    call_strike_mult: float = 1.05  # OTM call strike = 1.05 * S0


@dataclass
class ProtectivePut:
    """
    Long stock + long OTM put.

    At inception:
        Cost = S0 (stock) + put_premium (insurance cost)

    At expiry:
        Payoff = max(S_T, K_put) - put_premium (paid up front)
        Effectively: floored at K_put, downside protection at the cost
        of the put premium.
    """
    put_strike_mult: float = 0.95   # OTM put strike = 0.95 * S0


@dataclass
class Collar:
    """
    Long stock + short OTM call + long OTM put.

    At inception:
        Cost = S0 + put_premium - call_premium

    At expiry:
        Payoff bounded:  max(min(S_T, K_call), K_put)
        Both upside and downside are capped. Often structured as a
        zero-cost collar by choosing strikes such that the call premium
        funds the put.
    """
    call_strike_mult: float = 1.05
    put_strike_mult: float = 0.95


# ============================================================
# Pricing per asset
# ============================================================

def price_covered_call(
    S0: float,
    sigma: float,
    T: float,
    r: float,
    strategy: CoveredCall,
    q: float = 0.0,
) -> dict:
    """
    Price a covered call and return its Greeks.

    The covered call POSITION = +1 stock - 1 call.
    Greeks: stock contributes (delta=1, gamma=0, vega=0, theta=0, rho=0).
    Short call contributes the negation of long-call Greeks.
    """
    K_call = strategy.call_strike_mult * S0
    call_p = call_price(S0, K_call, T, r, sigma, q)

    # Position Greeks: stock + (-1) * call
    pos_delta = 1.0 - delta(S0, K_call, T, r, sigma, q, "call")
    pos_gamma = -gamma(S0, K_call, T, r, sigma, q)
    pos_vega  = -vega(S0, K_call, T, r, sigma, q)
    pos_theta = -theta(S0, K_call, T, r, sigma, q, "call")
    pos_rho   = -rho(S0, K_call, T, r, sigma, q, "call")

    return {
        "S0":               S0,
        "K_call":           K_call,
        "call_premium":     float(call_p),
        "net_cost":         float(S0 - call_p),  # what you pay to enter
        "delta":            float(pos_delta),
        "gamma":            float(pos_gamma),
        "vega":             float(pos_vega),
        "theta":            float(pos_theta),
        "rho":              float(pos_rho),
    }


def price_protective_put(
    S0: float,
    sigma: float,
    T: float,
    r: float,
    strategy: ProtectivePut,
    q: float = 0.0,
) -> dict:
    """
    Price a protective put and return its Greeks.

    The protective put POSITION = +1 stock + 1 put.
    """
    K_put = strategy.put_strike_mult * S0
    put_p = put_price(S0, K_put, T, r, sigma, q)

    # Position Greeks: stock + put
    pos_delta = 1.0 + delta(S0, K_put, T, r, sigma, q, "put")
    pos_gamma = gamma(S0, K_put, T, r, sigma, q)
    pos_vega  = vega(S0, K_put, T, r, sigma, q)
    pos_theta = theta(S0, K_put, T, r, sigma, q, "put")
    pos_rho   = rho(S0, K_put, T, r, sigma, q, "put")

    return {
        "S0":               S0,
        "K_put":            K_put,
        "put_premium":      float(put_p),
        "net_cost":         float(S0 + put_p),  # what you pay to enter
        "delta":            float(pos_delta),
        "gamma":            float(pos_gamma),
        "vega":             float(pos_vega),
        "theta":            float(pos_theta),
        "rho":              float(pos_rho),
    }


def price_collar(
    S0: float,
    sigma: float,
    T: float,
    r: float,
    strategy: Collar,
    q: float = 0.0,
) -> dict:
    """
    Price a collar and return its Greeks.

    Position = +1 stock + 1 put - 1 call.
    """
    K_call = strategy.call_strike_mult * S0
    K_put = strategy.put_strike_mult * S0
    call_p = call_price(S0, K_call, T, r, sigma, q)
    put_p = put_price(S0, K_put, T, r, sigma, q)

    # Position Greeks
    pos_delta = 1.0 + delta(S0, K_put, T, r, sigma, q, "put") - delta(S0, K_call, T, r, sigma, q, "call")
    pos_gamma = gamma(S0, K_put, T, r, sigma, q) - gamma(S0, K_call, T, r, sigma, q)
    pos_vega  = vega(S0, K_put, T, r, sigma, q) - vega(S0, K_call, T, r, sigma, q)
    pos_theta = theta(S0, K_put, T, r, sigma, q, "put") - theta(S0, K_call, T, r, sigma, q, "call")
    pos_rho   = rho(S0, K_put, T, r, sigma, q, "put") - rho(S0, K_call, T, r, sigma, q, "call")

    return {
        "S0":               S0,
        "K_call":           K_call,
        "K_put":            K_put,
        "call_premium":     float(call_p),
        "put_premium":      float(put_p),
        "net_premium":      float(put_p - call_p),  # >0 means net debit
        "net_cost":         float(S0 + put_p - call_p),
        "delta":            float(pos_delta),
        "gamma":            float(pos_gamma),
        "vega":             float(pos_vega),
        "theta":            float(pos_theta),
        "rho":              float(pos_rho),
    }


# ============================================================
# Terminal payoff under simulated paths
# ============================================================

def covered_call_payoff(
    S_T: np.ndarray,
    S0: float,
    strategy: CoveredCall,
    sigma: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """
    Terminal payoff per share for a covered call held to expiry.

    Payoff = min(S_T, K_call) + call_premium * (1 + r)^T
                                             ^---- premium grows at risk-free rate

    We compound the premium received at the risk-free rate to get the
    fair comparison with the underlying-only return at maturity.

    Returns
    -------
    np.ndarray of shape S_T.shape
        Terminal portfolio value per share.
    """
    K_call = strategy.call_strike_mult * S0
    call_p = call_price(S0, K_call, T, r, sigma, q)
    capped_stock = np.minimum(S_T, K_call)
    accrued_premium = call_p * np.exp(r * T)  # continuous compounding for consistency
    return capped_stock + accrued_premium


def protective_put_payoff(
    S_T: np.ndarray,
    S0: float,
    strategy: ProtectivePut,
    sigma: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """
    Terminal payoff per share for a protective put held to expiry.

    Payoff = max(S_T, K_put) - put_premium * (1 + r)^T
                                            ^---- premium paid grows at rf
    """
    K_put = strategy.put_strike_mult * S0
    put_p = put_price(S0, K_put, T, r, sigma, q)
    floored_stock = np.maximum(S_T, K_put)
    cost_of_put = put_p * np.exp(r * T)
    return floored_stock - cost_of_put


def collar_payoff(
    S_T: np.ndarray,
    S0: float,
    strategy: Collar,
    sigma: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> np.ndarray:
    """
    Terminal payoff per share for a collar held to expiry.

    Payoff = max(min(S_T, K_call), K_put) - net_premium * (1+r)^T
    """
    K_call = strategy.call_strike_mult * S0
    K_put = strategy.put_strike_mult * S0
    call_p = call_price(S0, K_call, T, r, sigma, q)
    put_p = put_price(S0, K_put, T, r, sigma, q)
    bounded_stock = np.maximum(np.minimum(S_T, K_call), K_put)
    net_premium = put_p - call_p  # positive = net debit
    cost_of_collar = net_premium * np.exp(r * T)
    return bounded_stock - cost_of_collar


# ============================================================
# Portfolio-level overlay
# ============================================================

def portfolio_overlay_payoff(
    S_T_paths: np.ndarray,
    S0_vec: np.ndarray,
    weights: np.ndarray,
    strategy_per_asset: list,
    sigma_vec: np.ndarray,
    T: float,
    r: float,
    q_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute portfolio-level terminal returns under a per-asset overlay strategy.

    For each path, the portfolio value at T is:
        V_T = sum_i w_i * (overlay_payoff_i(S_T_i) / S0_i)

    where overlay_payoff_i depends on the strategy assigned to asset i
    (None = unhedged, plain stock).

    Parameters
    ----------
    S_T_paths : np.ndarray, shape (n_paths, n_assets)
        Simulated terminal asset prices.
    S0_vec : np.ndarray, shape (n_assets,)
        Initial asset prices.
    weights : np.ndarray, shape (n_assets,)
        Portfolio weights (must sum to 1).
    strategy_per_asset : list of len n_assets
        Each entry is None (unhedged) or one of CoveredCall, ProtectivePut, Collar.
    sigma_vec : np.ndarray, shape (n_assets,)
        BSM volatility per asset (typically GBM-calibrated sigma).
    T : float
        Tenor in years.
    r : float
        Risk-free rate.
    q_vec : np.ndarray, optional
        Continuous dividend yields. Defaults to zero for all assets.

    Returns
    -------
    portfolio_terminal_returns : np.ndarray, shape (n_paths,)
        Per-path log returns of the overlaid portfolio.
    """
    n_paths, n_assets = S_T_paths.shape
    if q_vec is None:
        q_vec = np.zeros(n_assets)

    per_asset_terminal_value = np.empty_like(S_T_paths)  # value-per-share at T

    for i in range(n_assets):
        S0 = float(S0_vec[i])
        sigma = float(sigma_vec[i])
        q = float(q_vec[i])
        S_T_i = S_T_paths[:, i]
        strat = strategy_per_asset[i]

        if strat is None:
            per_asset_terminal_value[:, i] = S_T_i
        elif isinstance(strat, CoveredCall):
            per_asset_terminal_value[:, i] = covered_call_payoff(S_T_i, S0, strat, sigma, T, r, q)
        elif isinstance(strat, ProtectivePut):
            per_asset_terminal_value[:, i] = protective_put_payoff(S_T_i, S0, strat, sigma, T, r, q)
        elif isinstance(strat, Collar):
            per_asset_terminal_value[:, i] = collar_payoff(S_T_i, S0, strat, sigma, T, r, q)
        else:
            raise ValueError(f"Unknown strategy type: {type(strat).__name__}")

    # Per-path total log return = log(sum_i w_i * P_i_T / S0_i)
    price_ratios = per_asset_terminal_value / S0_vec  # (n_paths, n_assets)
    portfolio_terminal_value = (price_ratios * weights).sum(axis=1)  # (n_paths,)
    return np.log(np.maximum(portfolio_terminal_value, 1e-12))


# ============================================================
# Portfolio-level Greeks aggregator
# ============================================================

def portfolio_greeks(
    S0_vec: np.ndarray,
    weights: np.ndarray,
    strategy_per_asset: list,
    sigma_vec: np.ndarray,
    T: float,
    r: float,
    q_vec: np.ndarray | None = None,
    portfolio_value: float = 1.0,
) -> dict:
    """
    Aggregate portfolio-level Greeks from per-asset overlay strategies.

    Greeks are dollar-Greeks scaled by holding sizes. Specifically, for a
    portfolio of value V allocated by weights w_i, the dollar holding in
    asset i is V * w_i, and the number of shares is (V * w_i) / S_i.

    Position Greeks (per share) are computed by the price_* functions
    above; we multiply by share counts to get dollar Greeks.

    Returns
    -------
    dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho' summed
    across all assets.
    """
    n_assets = len(weights)
    if q_vec is None:
        q_vec = np.zeros(n_assets)

    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    for i in range(n_assets):
        S0 = float(S0_vec[i])
        sigma = float(sigma_vec[i])
        q = float(q_vec[i])
        strat = strategy_per_asset[i]
        # Number of shares of asset i = portfolio_value * w_i / S0_i
        n_shares = portfolio_value * float(weights[i]) / S0

        if strat is None:
            # Plain stock: only delta = 1 per share, all other Greeks zero
            totals["delta"] += n_shares * 1.0
            continue

        if isinstance(strat, CoveredCall):
            g = price_covered_call(S0, sigma, T, r, strat, q)
        elif isinstance(strat, ProtectivePut):
            g = price_protective_put(S0, sigma, T, r, strat, q)
        elif isinstance(strat, Collar):
            g = price_collar(S0, sigma, T, r, strat, q)
        else:
            raise ValueError(f"Unknown strategy type: {type(strat).__name__}")

        for greek_name in totals:
            totals[greek_name] += n_shares * g[greek_name]

    return totals
