"""
Calibration procedures for the four stochastic models in StochastiQ.

Each function fits a model's parameters to a time series of historical
log returns. The fitting procedures are deliberately well-known and
defensible -- this module is the technical heart of the project, and
clarity is more valuable than cleverness.

Procedures implemented:

    GBM     -- Maximum likelihood: sample mean and standard deviation
               of log returns, annualized.

    Merton  -- Threshold method: returns beyond N sigma are flagged as
               jumps; jump distribution fitted as log-normal; diffusion
               re-estimated on cleaned-of-jumps returns.

    CEV     -- Two-step OLS: estimate elasticity gamma via regression of
               log|delta S| on log S, then estimate the volatility scale
               sigma given gamma.

    Heston  -- Method of moments on rolling realized variance:
                  theta   = sample mean of realized variance
                  kappa   = -log(rho_AR1) of AR(1) on realized variance
                  sigma_v = volatility of innovations to realized variance
                  rho     = correlation between price returns and changes
                            in realized variance
                  v0      = most recent realized variance estimate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.gbm import GBMParams
from src.models.merton import MertonParams
from src.models.cev import CEVParams
from src.models.heston import HestonParams


# ============================================================
# GBM
# ============================================================

def calibrate_gbm(
    log_returns: pd.Series,
    trading_days: int = 252,
) -> GBMParams:
    """
    Maximum-likelihood calibration of GBM.

    Under GBM, log returns are i.i.d. normal with mean (mu - 0.5*sigma^2)*dt
    and variance sigma^2 * dt. The MLEs are simply the sample mean and
    sample variance of log returns, then annualized.
    """
    daily_mean = log_returns.mean()
    daily_var = log_returns.var(ddof=1)

    sigma = np.sqrt(daily_var * trading_days)
    # Inverting the drift relationship: daily_mean = (mu - 0.5*sigma^2)*dt
    mu = daily_mean * trading_days + 0.5 * sigma ** 2

    return GBMParams(mu=mu, sigma=sigma)


# ============================================================
# Merton
# ============================================================

def calibrate_merton(
    log_returns: pd.Series,
    trading_days: int = 252,
    jump_threshold_sigma: float = 3.0,
) -> MertonParams:
    """
    Threshold-based calibration of Merton jump-diffusion.

    Procedure:
      1. Standardize returns and flag those with |z| > threshold as jumps.
      2. Fit log-normal jump distribution to the flagged returns
         (mean and std of those returns).
      3. Re-estimate diffusion mu and sigma from the *non-jump* returns.
      4. lambda = (number of jumps) / (years of data).

    The threshold method is simple and transparent, well-suited to a class
    project. More sophisticated alternatives (Bayesian filtering, EM
    algorithm, characteristic-function MLE) exist but are not required.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns.
    trading_days : int
        Trading-day convention.
    jump_threshold_sigma : float
        Returns beyond this many standard deviations are flagged as jumps.
        3.0 captures the obvious tail events while keeping a clean
        diffusion sample.
    """
    returns = log_returns.dropna().values

    # Step 1: Identify jumps
    daily_std = returns.std(ddof=1)
    daily_mean = returns.mean()
    z_scores = (returns - daily_mean) / daily_std
    jump_mask = np.abs(z_scores) > jump_threshold_sigma
    n_jumps = jump_mask.sum()

    # Step 2: Jump distribution
    if n_jumps >= 2:
        jump_returns = returns[jump_mask]
        mu_j = float(jump_returns.mean())
        sigma_j = float(jump_returns.std(ddof=1))
    else:
        # Not enough jumps to fit; fall back to small jumps
        mu_j = 0.0
        sigma_j = 2.0 * daily_std

    # Step 3: Diffusion from non-jump returns
    diffusion_returns = returns[~jump_mask]
    diff_daily_mean = diffusion_returns.mean()
    diff_daily_var = diffusion_returns.var(ddof=1)
    sigma = float(np.sqrt(diff_daily_var * trading_days))

    # Step 4: Annualized jump intensity
    n_observations = len(returns)
    years_of_data = n_observations / trading_days
    lambda_j = float(n_jumps / years_of_data)

    # Total drift mu including jump compensation
    # Daily drift before jump effect: (mu - lambda*k - 0.5*sigma^2)*dt = diff_daily_mean
    k = float(np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0)
    mu = float(diff_daily_mean * trading_days + 0.5 * sigma ** 2 + lambda_j * k)

    return MertonParams(
        mu=mu,
        sigma=sigma,
        lambda_j=lambda_j,
        mu_j=mu_j,
        sigma_j=sigma_j,
    )


# ============================================================
# CEV
# ============================================================

# Bounds for the CEV elasticity parameter.
# Values outside [0.1, 1.5] are economically implausible for equities and
# typically indicate a poor regression fit (low signal-to-noise in the data).
# When the unconstrained OLS estimate falls outside this range, we fall back
# to gamma = 1.0 (CEV degenerates to GBM with sigma = empirical sigma).
CEV_GAMMA_MIN: float = 0.1
CEV_GAMMA_MAX: float = 1.5


def calibrate_cev(
    prices: pd.Series,
    log_returns: pd.Series,
    trading_days: int = 252,
) -> CEVParams:
    """
    OLS-based calibration of the CEV model with robustness fallbacks.

    Starting from the SDE dS = mu*S*dt + sigma*S^gamma*dW, the absolute
    increment satisfies:
        |dS| ~ sigma * S^gamma * sqrt(dt)
    Taking logs:
        log|dS| ~ log(sigma) + gamma * log(S) + 0.5 * log(dt)

    OLS regression of log|delta S| on log S yields gamma directly, and
    the intercept gives sigma (after subtracting the dt term).

    Robustness: For low-volatility series (e.g., broad-market ETFs) the
    log|dS| signal is weak relative to noise, and the OLS estimate of gamma
    can land outside any economically reasonable range. We bound the
    estimate to [CEV_GAMMA_MIN, CEV_GAMMA_MAX]; if it falls outside, we
    fall back to gamma = 1.0 with sigma equal to the GBM sigma. This keeps
    CEV simulation numerically stable across all assets.

    Parameters
    ----------
    prices : pd.Series
        Price series (NOT returns).
    log_returns : pd.Series
        Daily log returns (used for drift estimate).
    trading_days : int
        Trading-day convention.
    """
    prices = prices.dropna()
    delta_s = prices.diff().dropna()

    # Align price levels at t-1 with delta_s (forward differences)
    s_lag = prices.shift(1).dropna()
    s_lag = s_lag.loc[delta_s.index]

    # Filter out zero-increment days to avoid log(0)
    nonzero = delta_s.abs() > 1e-12
    log_abs_ds = np.log(delta_s[nonzero].abs().values)
    log_s = np.log(s_lag[nonzero].values)

    # OLS: log|dS| = a + gamma * log(S)
    A = np.vstack([np.ones_like(log_s), log_s]).T
    coeffs, *_ = np.linalg.lstsq(A, log_abs_ds, rcond=None)
    intercept, gamma_raw = float(coeffs[0]), float(coeffs[1])

    dt = 1.0 / trading_days

    # GBM sigma from log returns, used as fallback and as a sanity reference
    gbm_sigma = float(log_returns.std(ddof=1) * np.sqrt(trading_days))

    # Annualized drift estimate (same as GBM)
    mu = float(log_returns.mean() * trading_days + 0.5 * gbm_sigma ** 2)

    # Robustness check on gamma
    if CEV_GAMMA_MIN <= gamma_raw <= CEV_GAMMA_MAX:
        gamma = gamma_raw
        # Recover sigma from intercept
        sigma_raw = float(np.exp(intercept - 0.5 * np.log(dt)))
        # Sanity check sigma: if it's more than 10x the GBM sigma, the
        # regression is producing inconsistent estimates -- fall back to GBM.
        # The 10x threshold is generous but catches obvious failures.
        if sigma_raw > 10.0 * gbm_sigma or sigma_raw < 0.1 * gbm_sigma:
            gamma = 1.0
            sigma = gbm_sigma
        else:
            sigma = sigma_raw
    else:
        # gamma estimate is outside economic plausible range -- fall back to GBM
        gamma = 1.0
        sigma = gbm_sigma

    return CEVParams(mu=mu, sigma=sigma, gamma=gamma)


# ============================================================
# Heston
# ============================================================

def calibrate_heston(
    log_returns: pd.Series,
    trading_days: int = 252,
    rv_window: int = 21,
) -> HestonParams:
    """
    Method-of-moments calibration of Heston using rolling realized variance.

    Procedure:
      1. Compute rolling-window realized variance:
            RV_t = (sum of squared returns over rv_window days) * (trading_days / rv_window)
         This is an annualized variance proxy for v_t.
      2. theta = sample mean of RV.
      3. Fit AR(1): RV_t = a + b * RV_{t-1} + e_t.
            kappa = -log(b)             (continuous-time mean reversion speed)
            (or kappa = (1-b)*trading_days for daily-step interpretation)
         We use the continuous-time form.
      4. sigma_v = std(e_t) / sqrt(mean(RV_t)) * sqrt(trading_days)
         (standardized so that the units match dv = sigma_v*sqrt(v)*dW).
      5. rho = correlation between log returns and changes in RV.
      6. v0 = most recent RV value (front-loads simulations toward today's regime).

    This procedure is documented in standard quant references (Mikhailov &
    Nogel 2003, Aits-Sahalia & Kimmel 2007). It is the "first-pass"
    calibration -- production desks would refine via option-implied
    surfaces, but for historical-only equity calibration this is appropriate.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns.
    trading_days : int
        Trading-day convention.
    rv_window : int
        Rolling window for realized variance estimation. 21 days
        approximates one calendar month, a common choice.
    """
    returns = log_returns.dropna()

    # Step 1: Annualized realized variance, rolling window
    squared = returns ** 2
    rv = squared.rolling(window=rv_window).sum() * (trading_days / rv_window)
    rv = rv.dropna()

    # Step 2: long-run mean
    theta = float(rv.mean())

    # Step 3: AR(1) on RV
    rv_curr = rv.values[1:]
    rv_lag = rv.values[:-1]
    # OLS: RV_t = a + b * RV_{t-1}
    X = np.vstack([np.ones_like(rv_lag), rv_lag]).T
    coeffs, *_ = np.linalg.lstsq(X, rv_curr, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    residuals = rv_curr - (a + b * rv_lag)

    # Continuous-time kappa from AR(1) coefficient
    # If b = exp(-kappa * dt) then kappa = -log(b) / dt
    dt = 1.0 / trading_days
    if 0 < b < 1:
        kappa = float(-np.log(b) / dt)
    else:
        # AR(1) coefficient outside (0,1) -- fall back to (1-b)/dt
        # which is the small-dt approximation, and floor at a sensible value.
        kappa = float(max((1.0 - b) / dt, 0.5))

    # Step 4: sigma_v from residual std
    # Heston: dv ~ sigma_v * sqrt(v) * dW => std(dv) ~ sigma_v * sqrt(v) * sqrt(dt)
    # Using mean(v) = theta as the level:
    sigma_v = float(np.std(residuals, ddof=1) / np.sqrt(theta * dt))

    # Step 5: correlation between price returns and RV changes
    rv_changes = rv.diff().dropna()
    aligned_returns = returns.loc[rv_changes.index]
    rho = float(aligned_returns.corr(rv_changes))
    if not np.isfinite(rho):
        rho = -0.5  # sensible default for equities

    # Clip rho to avoid numerical issues at the boundary
    rho = float(np.clip(rho, -0.95, 0.95))

    # Step 6: initial variance = most recent RV
    v0 = float(rv.iloc[-1])

    # Drift estimate from log returns
    mu = float(returns.mean() * trading_days + 0.5 * theta)

    return HestonParams(
        mu=mu,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
    )


# ============================================================
# Convenience: calibrate all models for one asset
# ============================================================

def calibrate_all_models(
    prices: pd.Series,
    log_returns: pd.Series,
    trading_days: int = 252,
) -> dict:
    """
    Calibrate all four models for a single asset.

    Returns a dict keyed by model name with the calibrated Params object.
    """
    return {
        "GBM":    calibrate_gbm(log_returns, trading_days),
        "Merton": calibrate_merton(log_returns, trading_days),
        "CEV":    calibrate_cev(prices, log_returns, trading_days),
        "Heston": calibrate_heston(log_returns, trading_days),
    }
