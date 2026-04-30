"""
Black-Scholes-Merton (BSM) European option pricing and Greeks.

This module implements the BSM model for European calls and puts, with
all five primary Greeks computed in closed form:

    Delta    -- dV/dS    (sensitivity to underlying price)
    Gamma    -- d2V/dS2  (convexity / hedge ratio change)
    Vega     -- dV/dsigma (sensitivity to volatility)
    Theta    -- dV/dt    (time decay)
    Rho      -- dV/dr    (rate sensitivity)

The pricer assumes:
  - Continuous compounding for the risk-free rate
  - Continuous dividend yield q (set q=0 for non-dividend-paying assets)
  - European exercise (no American optionality)
  - Constant volatility (Black-Scholes assumption)

These assumptions are appropriate for our class-project framework. In
production, equity options would use forward vol surfaces and discrete
dividend handling; this module provides the textbook foundation that
demonstrates BSM mechanics cleanly.

All functions are vectorized: S, K, T, sigma can be scalars or arrays
of compatible shape, and the output broadcasts accordingly.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ============================================================
# Core BSM primitives
# ============================================================

def d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> tuple:
    """
    Compute the d1 and d2 quantities used throughout BSM.

        d1 = [log(S/K) + (r - q + 0.5*sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Returns
    -------
    (d1, d2) : tuple of float or np.ndarray
    """
    sqrt_T = np.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


# ============================================================
# Prices
# ============================================================

def call_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    BSM European call price.

        C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    BSM European put price.

        P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ============================================================
# Greeks -- closed-form
# ============================================================

def delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
    option_type: str = "call",
) -> float | np.ndarray:
    """
    Delta = dV/dS.

    Call:  e^(-qT) * N(d1)
    Put:   e^(-qT) * (N(d1) - 1)
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-q * T)
    if option_type == "call":
        return discount * norm.cdf(d1)
    elif option_type == "put":
        return discount * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError(f"Unknown option type: {option_type!r}")


def gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    Gamma = d2V/dS2 (same for calls and puts).

        Gamma = e^(-qT) * phi(d1) / (S * sigma * sqrt(T))

    where phi is the standard normal PDF.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    Vega = dV/dsigma (same for calls and puts).

        Vega = S * e^(-qT) * phi(d1) * sqrt(T)

    Quoted per 1.0 unit change in sigma. Divide by 100 for change-per-1%.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
    option_type: str = "call",
) -> float | np.ndarray:
    """
    Theta = dV/dt (sign convention: time-decay, typically negative).

    Call:
      Theta = -[S * e^(-qT) * phi(d1) * sigma / (2*sqrt(T))]
              - r * K * e^(-rT) * N(d2)
              + q * S * e^(-qT) * N(d1)

    Put:
      Theta = -[S * e^(-qT) * phi(d1) * sigma / (2*sqrt(T))]
              + r * K * e^(-rT) * N(-d2)
              - q * S * e^(-qT) * N(-d1)

    Quoted in price-units per year. Divide by 365 for per-day.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    discount_q = np.exp(-q * T)
    discount_r = np.exp(-r * T)

    common = -S * discount_q * pdf_d1 * sigma / (2.0 * np.sqrt(T))

    if option_type == "call":
        return common - r * K * discount_r * norm.cdf(d2) + q * S * discount_q * norm.cdf(d1)
    elif option_type == "put":
        return common + r * K * discount_r * norm.cdf(-d2) - q * S * discount_q * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {option_type!r}")


def rho(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
    option_type: str = "call",
) -> float | np.ndarray:
    """
    Rho = dV/dr (sensitivity to risk-free rate).

    Call:  K * T * e^(-rT) * N(d2)
    Put:  -K * T * e^(-rT) * N(-d2)
    """
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    discount_r = np.exp(-r * T)
    if option_type == "call":
        return K * T * discount_r * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * discount_r * norm.cdf(-d2)
    else:
        raise ValueError(f"Unknown option type: {option_type!r}")


# ============================================================
# Convenience: full Greeks dictionary
# ============================================================

def all_greeks(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
    option_type: str = "call",
) -> dict:
    """
    Compute price and all five Greeks in one call.

    Returns a dict with keys: 'price', 'delta', 'gamma', 'vega',
    'theta', 'rho'.
    """
    if option_type == "call":
        price = call_price(S, K, T, r, sigma, q)
    elif option_type == "put":
        price = put_price(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"Unknown option type: {option_type!r}")

    return {
        "price":  price,
        "delta":  delta(S, K, T, r, sigma, q, option_type),
        "gamma":  gamma(S, K, T, r, sigma, q),
        "vega":   vega(S, K, T, r, sigma, q),
        "theta":  theta(S, K, T, r, sigma, q, option_type),
        "rho":    rho(S, K, T, r, sigma, q, option_type),
    }


# ============================================================
# Put-call parity check (sanity utility)
# ============================================================

def put_call_parity_violation(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    Compute the violation of put-call parity:

        C - P  ?=  S * e^(-qT) - K * e^(-rT)

    Returns the absolute residual. Should be effectively zero
    (numerical noise only) if BSM pricing is correct.
    """
    c = call_price(S, K, T, r, sigma, q)
    p = put_price(S, K, T, r, sigma, q)
    parity_lhs = c - p
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return np.abs(parity_lhs - parity_rhs)
