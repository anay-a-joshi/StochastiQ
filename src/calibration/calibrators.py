"""
Calibration procedures for the four stochastic models in StochastiQ.

This module fits each model's parameters to a time series of historical
log returns. All procedures are documented and defensible -- the math is
the technical heart of the project, and clarity is more valuable than
cleverness.

Procedures implemented (per model):

    GBM     -- Maximum likelihood: sample mean and standard deviation
               of log returns, annualized.

    Merton  -- Threshold method: returns beyond N sigma are flagged as
               jumps; jump distribution fitted as log-normal; diffusion
               re-estimated on cleaned-of-jumps returns.

    CEV     -- Two methods provided:
                 (a) `calibrate_cev`     -- Two-step OLS on log|delta S|
                                            vs log S (fast, can be fragile
                                            for low-vol smooth series).
                 (b) `calibrate_cev_nls` -- Nonlinear least squares fitting
                                            CEV-implied local vol to rolling
                                            realized vol. Robust across all
                                            asset types (recommended).

    Heston  -- Two variants provided:
                 (a) `calibrate_heston`              -- Unconstrained method
                                                        of moments on rolling
                                                        21-day realized
                                                        variance.
                 (b) `calibrate_heston_constrained`  -- Same MoM but enforces
                                                        Feller condition by
                                                        capping sigma_v.

Both Heston variants are presented in the notebook with a comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp

from src.models.gbm import GBMParams
from src.models.merton import MertonParams
from src.models.cev import CEVParams
from src.models.heston import HestonParams


# ============================================================
# Module-level constants for calibration robustness
# ============================================================

# Bounds for the CEV elasticity parameter. Values outside [0.1, 1.5] are
# economically implausible for equities; when the OLS estimate falls outside
# this range we fall back to gamma = 1.0 (CEV degenerates to GBM).
CEV_GAMMA_MIN: float = 0.1
CEV_GAMMA_MAX: float = 1.5

# Margin used by the Feller-constrained Heston calibrator.
# We enforce 2*kappa*theta >= (1 + FELLER_MARGIN) * sigma_v^2, so the
# Feller condition holds with a small safety buffer rather than at the
# boundary. 0.05 = 5% buffer.
FELLER_MARGIN: float = 0.05


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
      2. Fit log-normal jump distribution to the flagged returns.
      3. Re-estimate diffusion mu and sigma from the non-jump returns.
      4. lambda = (number of jumps) / (years of data).

    The threshold method is simple and transparent. More sophisticated
    alternatives (Bayesian filtering, EM algorithm, characteristic-function
    MLE) exist but are out of scope for this project.
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
# CEV -- Method A: OLS on log|delta S| vs log S
# ============================================================

def calibrate_cev(
    prices: pd.Series,
    log_returns: pd.Series,
    trading_days: int = 252,
) -> CEVParams:
    """
    OLS-based calibration of CEV (Method A: legacy, kept for comparison).

    Starting from the SDE dS = mu*S*dt + sigma*S^gamma*dW:
        |dS| ~ sigma * S^gamma * sqrt(dt)
        log|dS| ~ log(sigma) + gamma * log(S) + 0.5 * log(dt)

    OLS regression of log|delta S| on log S yields gamma directly, and
    the intercept gives sigma. This method is fast but can be fragile for
    smooth low-volatility series (the log|dS| signal is weak relative to
    noise). When the estimate falls outside reasonable bounds, we fall
    back to gamma = 1.0 with sigma = GBM sigma.

    The newer `calibrate_cev_nls` is more robust and is recommended as the
    default in the notebook.
    """
    prices = prices.dropna()
    delta_s = prices.diff().dropna()

    s_lag = prices.shift(1).dropna().loc[delta_s.index]

    nonzero = delta_s.abs() > 1e-12
    log_abs_ds = np.log(delta_s[nonzero].abs().values)
    log_s = np.log(s_lag[nonzero].values)

    A = np.vstack([np.ones_like(log_s), log_s]).T
    coeffs, *_ = np.linalg.lstsq(A, log_abs_ds, rcond=None)
    intercept, gamma_raw = float(coeffs[0]), float(coeffs[1])

    dt = 1.0 / trading_days
    gbm_sigma = float(log_returns.std(ddof=1) * np.sqrt(trading_days))
    mu = float(log_returns.mean() * trading_days + 0.5 * gbm_sigma ** 2)

    if CEV_GAMMA_MIN <= gamma_raw <= CEV_GAMMA_MAX:
        gamma = gamma_raw
        sigma_raw = float(np.exp(intercept - 0.5 * np.log(dt)))
        if sigma_raw > 10.0 * gbm_sigma or sigma_raw < 0.1 * gbm_sigma:
            gamma = 1.0
            sigma = gbm_sigma
        else:
            sigma = sigma_raw
    else:
        gamma = 1.0
        sigma = gbm_sigma

    return CEVParams(mu=mu, sigma=sigma, gamma=gamma)


# ============================================================
# CEV -- Method B: Nonlinear least squares on rolling realized vol
# ============================================================

def calibrate_cev_nls(
    prices: pd.Series,
    log_returns: pd.Series,
    trading_days: int = 252,
    rv_window: int = 21,
) -> CEVParams:
    """
    Nonlinear least squares calibration of CEV (Method B: recommended).

    Idea: under CEV, the local volatility at price S is sigma_local(S) =
    sigma * S^(gamma - 1). We compute rolling realized volatility from the
    return series and fit the parameters (sigma, gamma) by minimizing the
    squared error between observed rolling realized vol and the model-implied
    local vol at each time:

        minimize sum_t [ rv_t - sigma * S_t^(gamma - 1) ]^2

    For a given gamma, the optimal sigma is available in closed form:
        sigma(gamma) = sum(rv_t * S_t^(gamma - 1)) / sum(S_t^(2*(gamma - 1)))

    So we reduce the 2D optimization to a 1D search over gamma in
    [CEV_GAMMA_MIN, CEV_GAMMA_MAX], using bounded scalar minimization.

    This method is more robust than the log|dS| OLS regression because:
      - It uses a smoothed volatility signal (rolling RV) rather than
        single-day |delta S|, which is much noisier.
      - It directly fits the quantity the model is trying to capture
        (the local-vol -- price-level relationship) rather than its log.
      - It naturally enforces gamma bounds via bounded optimization.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    log_returns : pd.Series
        Daily log returns (used for drift estimate).
    trading_days : int
        Trading-day convention.
    rv_window : int
        Rolling window for realized vol estimation. 21 days is one month.
    """
    prices = prices.dropna()

    # Rolling realized vol (annualized, in volatility units, not variance)
    rv = log_returns.rolling(window=rv_window).std() * np.sqrt(trading_days)
    rv = rv.dropna()

    # Align prices with rv
    s = prices.loc[rv.index].values
    rv_vals = rv.values

    # Drop any non-positive prices defensively (should not happen for real data)
    valid = (s > 0) & np.isfinite(rv_vals)
    s = s[valid]
    rv_vals = rv_vals[valid]

    def loss(gamma: float) -> float:
        """Squared error between observed RV and CEV-implied local vol."""
        # Closed-form sigma given gamma
        s_pow = np.power(s, gamma - 1.0)
        denom = np.sum(s_pow ** 2)
        if denom < 1e-12:
            return np.inf
        sigma = np.sum(rv_vals * s_pow) / denom
        if sigma <= 0:
            return np.inf
        residuals = rv_vals - sigma * s_pow
        return float(np.sum(residuals ** 2))

    # 1D bounded search over gamma
    result = minimize_scalar(
        loss,
        bounds=(CEV_GAMMA_MIN, CEV_GAMMA_MAX),
        method="bounded",
        options={"xatol": 1e-4},
    )

    gamma = float(result.x)
    # Recover sigma at the optimum
    s_pow = np.power(s, gamma - 1.0)
    sigma = float(np.sum(rv_vals * s_pow) / np.sum(s_pow ** 2))

    # Drift from log returns
    gbm_sigma = float(log_returns.std(ddof=1) * np.sqrt(trading_days))
    mu = float(log_returns.mean() * trading_days + 0.5 * gbm_sigma ** 2)

    return CEVParams(mu=mu, sigma=sigma, gamma=gamma)


# ============================================================
# Heston -- Method A: unconstrained method of moments
# ============================================================

def calibrate_heston(
    log_returns: pd.Series,
    trading_days: int = 252,
    rv_window: int = 21,
) -> HestonParams:
    """
    Unconstrained method-of-moments calibration of Heston.

    Procedure:
      1. Compute rolling-window annualized realized variance (proxy for v_t).
      2. theta = sample mean of RV.
      3. Fit AR(1) on RV; extract continuous-time kappa from the AR
         coefficient.
      4. sigma_v = std(residuals) standardized by sqrt(theta * dt).
      5. rho = correlation between log returns and changes in RV.
      6. v0 = most recent RV value.

    Reference: Mikhailov & Nogel (2003), Aits-Sahalia & Kimmel (2007).

    Note: this method makes no attempt to enforce the Feller condition
    2*kappa*theta > sigma_v^2. For high-vol-of-vol assets (e.g. equities
    during 2020-2022) this routinely produces Feller-violating estimates.
    The full-truncation Euler scheme handles violations correctly at
    simulation time, but a Feller-compliant alternative is provided in
    `calibrate_heston_constrained`.
    """
    returns = log_returns.dropna()

    # Step 1: annualized realized variance
    squared = returns ** 2
    rv = squared.rolling(window=rv_window).sum() * (trading_days / rv_window)
    rv = rv.dropna()

    # Step 2: long-run mean
    theta = float(rv.mean())

    # Step 3: AR(1) on RV
    rv_curr = rv.values[1:]
    rv_lag = rv.values[:-1]
    X = np.vstack([np.ones_like(rv_lag), rv_lag]).T
    coeffs, *_ = np.linalg.lstsq(X, rv_curr, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    residuals = rv_curr - (a + b * rv_lag)

    dt = 1.0 / trading_days
    if 0 < b < 1:
        kappa = float(-np.log(b) / dt)
    else:
        kappa = float(max((1.0 - b) / dt, 0.5))

    # Step 4: sigma_v from residual std
    sigma_v = float(np.std(residuals, ddof=1) / np.sqrt(theta * dt))

    # Step 5: correlation between price returns and RV changes
    rv_changes = rv.diff().dropna()
    aligned_returns = returns.loc[rv_changes.index]
    rho = float(aligned_returns.corr(rv_changes))
    if not np.isfinite(rho):
        rho = -0.5
    rho = float(np.clip(rho, -0.95, 0.95))

    # Step 6: initial variance
    v0 = float(rv.iloc[-1])

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
# Heston -- Method B: Feller-constrained method of moments
# ============================================================

def calibrate_heston_constrained(
    log_returns: pd.Series,
    trading_days: int = 252,
    rv_window: int = 21,
    feller_margin: float = FELLER_MARGIN,
) -> HestonParams:
    """
    Feller-constrained calibration of Heston.

    This produces parameters identical to `calibrate_heston` but caps
    sigma_v so the Feller condition holds with a small safety margin:

        2 * kappa * theta >= (1 + feller_margin) * sigma_v^2

    Equivalently:

        sigma_v <= sqrt(2 * kappa * theta / (1 + feller_margin))

    When the unconstrained estimate exceeds this cap, we cap it. This
    guarantees the variance process remains strictly positive in the
    continuous-time formulation, at the cost of slightly under-fitting
    the empirical vol-of-vol.

    Use this variant when:
      - Theoretical guarantees on positivity matter (e.g. for analytical
        results or risk calculations sensitive to the boundary).
      - You want a "safer" calibration that avoids the full-truncation
        regime entirely.

    Use the unconstrained `calibrate_heston` when:
      - Empirical fit to realized vol-of-vol matters more than theoretical
        purity (typical for equity option pricing).

    The notebook reports both side-by-side.
    """
    # Get the unconstrained calibration first
    p = calibrate_heston(log_returns, trading_days, rv_window)

    # Compute Feller cap
    feller_cap = float(np.sqrt(2 * p.kappa * p.theta / (1.0 + feller_margin)))

    # Cap sigma_v if it exceeds the Feller-compliant ceiling
    sigma_v_constrained = float(min(p.sigma_v, feller_cap))

    return HestonParams(
        mu=p.mu,
        kappa=p.kappa,
        theta=p.theta,
        sigma_v=sigma_v_constrained,
        rho=p.rho,
        v0=p.v0,
    )


# ============================================================
# Goodness-of-fit tests
# ============================================================

def ks_test_returns(
    empirical_returns: np.ndarray,
    simulated_returns: np.ndarray,
) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test comparing empirical and simulated
    return distributions.

    Returns
    -------
    (ks_statistic, p_value)
        Lower KS statistic = better fit. p-value > 0.05 means we fail to
        reject the null that both samples come from the same distribution
        (i.e. the model fits the data at the 5% level).
    """
    # Suppress the warning about ties (returns can have duplicate values
    # at high precision); the test is still valid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ks_2samp(empirical_returns, simulated_returns)
    return float(result.statistic), float(result.pvalue)


# ============================================================
# Convenience: calibrate all models for one asset
# ============================================================

def calibrate_all_models(
    prices: pd.Series,
    log_returns: pd.Series,
    trading_days: int = 252,
    cev_method: str = "nls",
    heston_constrained: bool = False,
) -> dict:
    """
    Calibrate all four models for a single asset.

    Parameters
    ----------
    prices, log_returns : pd.Series
        Price and log-return series.
    trading_days : int
        Trading-day convention.
    cev_method : {'nls', 'ols'}
        Which CEV calibrator to use. 'nls' is the recommended robust method.
    heston_constrained : bool
        If True, use the Feller-constrained Heston calibrator.

    Returns
    -------
    dict
        Keyed by model name with the calibrated Params object.
    """
    if cev_method == "nls":
        cev_params = calibrate_cev_nls(prices, log_returns, trading_days)
    elif cev_method == "ols":
        cev_params = calibrate_cev(prices, log_returns, trading_days)
    else:
        raise ValueError(f"Unknown cev_method: {cev_method!r}. Use 'nls' or 'ols'.")

    if heston_constrained:
        heston_params = calibrate_heston_constrained(log_returns, trading_days)
    else:
        heston_params = calibrate_heston(log_returns, trading_days)

    return {
        "GBM":    calibrate_gbm(log_returns, trading_days),
        "Merton": calibrate_merton(log_returns, trading_days),
        "CEV":    cev_params,
        "Heston": heston_params,
    }
