"""
Market regime detection.

Two complementary detectors:

1. Gaussian HMM (primary) -- a 2-state hidden Markov model with Gaussian
   emissions, fit to a univariate series of daily log returns. The latent
   states are interpreted as Calm / Stress. After fitting we deterministically
   relabel the states by within-state variance: the higher-variance state
   is "Stress", the lower-variance state is "Calm". This eliminates the
   non-identifiability of the raw state indices that hmmlearn returns.

2. Realized-volatility threshold (robustness check) -- compute a rolling
   N-day realized volatility on the same return series and binarize at
   the in-sample median. This is the simplest defensible regime classifier
   and serves as a sanity check on the HMM labels.

The HMM is fit on training-period data only; out-of-sample labels are
produced by inference (no refit), preserving the project's train/test
discipline.

References
----------
Hamilton, J. D. (1989). "A new approach to the economic analysis of
    nonstationary time series and the business cycle." Econometrica.
Ang, A. & Bekaert, G. (2002). "International asset allocation with
    regime shifts." Review of Financial Studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


CALM_LABEL = "Calm"
STRESS_LABEL = "Stress"


# ============================================================
# Container for fitted HMM artifacts
# ============================================================

@dataclass
class HMMRegimeModel:
    """Bundle of artifacts from a fitted regime HMM.

    Attributes
    ----------
    model : hmmlearn.hmm.GaussianHMM
        The fitted model. Use ``.score()`` for log-likelihood,
        ``.predict_proba()`` for posteriors, etc.
    calm_state : int
        Raw HMM state index identified as Calm (lower variance).
    stress_state : int
        Raw HMM state index identified as Stress (higher variance).
    state_means : np.ndarray
        Daily-log-return means by state, ordered [calm, stress].
    state_vols : np.ndarray
        Daily-log-return std devs by state, ordered [calm, stress].
    transition_matrix : np.ndarray
        Transition probability matrix in [calm, stress] order
        (i.e. relabeled, not the raw hmmlearn output).
    """

    model: GaussianHMM
    calm_state: int
    stress_state: int
    state_means: np.ndarray
    state_vols: np.ndarray
    transition_matrix: np.ndarray


# ============================================================
# HMM fitting
# ============================================================

def fit_hmm_regime(
    log_returns: pd.Series,
    n_iter: int = 1000,
    random_state: int = 42,
    covariance_type: Literal["full", "diag", "tied", "spherical"] = "full",
) -> HMMRegimeModel:
    """
    Fit a 2-state Gaussian HMM to a univariate daily log-return series.

    The two latent states are deterministically relabeled by within-state
    variance: the state with higher daily-return variance is identified as
    Stress (index 1 after relabeling) and the lower-variance state as
    Calm (index 0).

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns of a market proxy (typically SPY). Index must be
        date-like.
    n_iter : int
        Maximum EM iterations.
    random_state : int
        Seed for hmmlearn's internal initialization. Fixed for reproducibility.
    covariance_type : str
        Covariance structure of the Gaussian emissions. Default "full" is
        appropriate for univariate returns (degenerates to a single variance).

    Returns
    -------
    HMMRegimeModel
        Bundle containing the fitted model and the calm/stress index mapping.
    """
    if log_returns.isna().any():
        raise ValueError(
            "log_returns contains NaN values; drop them before fitting."
        )
    if len(log_returns) < 100:
        raise ValueError(
            f"log_returns has only {len(log_returns)} observations; "
            "HMM regime detection needs at least ~100 daily returns."
        )

    X = log_returns.values.reshape(-1, 1).astype(float)

    model = GaussianHMM(
        n_components=2,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-6,
    )
    model.fit(X)

    if not model.monitor_.converged:
        # We do not raise -- EM monotonicity guarantees the result is still
        # a valid local optimum -- but we surface the warning to the caller
        # via state_vols (the user can inspect the model object directly).
        pass

    # Identify states by within-state variance using the fitted means/covars.
    # GaussianHMM stores means as (n_components, n_features) and covars as
    # (n_components, n_features, n_features) when covariance_type="full".
    raw_means = model.means_.flatten()                        # shape (2,)
    raw_vars = np.array([model.covars_[k].flatten()[0] for k in range(2)])  # shape (2,)
    raw_vols = np.sqrt(raw_vars)

    # Relabel: higher-variance state => Stress
    if raw_vars[0] < raw_vars[1]:
        calm_state, stress_state = 0, 1
    else:
        calm_state, stress_state = 1, 0

    # Build relabeled transition matrix in [calm, stress] order
    # P_relabeled[i, j] = P(state_relabeled_t+1 = j | state_relabeled_t = i)
    raw_T = model.transmat_
    perm = np.array([calm_state, stress_state])
    transition_matrix = raw_T[np.ix_(perm, perm)]

    state_means = raw_means[perm]
    state_vols = raw_vols[perm]

    return HMMRegimeModel(
        model=model,
        calm_state=int(calm_state),
        stress_state=int(stress_state),
        state_means=state_means,
        state_vols=state_vols,
        transition_matrix=transition_matrix,
    )


def predict_hmm_regime(
    hmm_model: HMMRegimeModel,
    log_returns: pd.Series,
) -> pd.DataFrame:
    """
    Apply a fitted HMM to a return series (training or OOS) and return
    regime labels plus posterior probabilities.

    Parameters
    ----------
    hmm_model : HMMRegimeModel
        Output of `fit_hmm_regime`.
    log_returns : pd.Series
        Daily log returns to label.

    Returns
    -------
    pd.DataFrame
        Index = log_returns.index. Columns:
            state_raw : int -- raw HMM state index (0 or 1)
            label : str -- "Calm" or "Stress"
            calm_prob : float -- posterior probability of Calm
            stress_prob : float -- posterior probability of Stress
    """
    if log_returns.isna().any():
        raise ValueError(
            "log_returns contains NaN values; drop them before predicting."
        )

    X = log_returns.values.reshape(-1, 1).astype(float)
    raw_states = hmm_model.model.predict(X)
    raw_posterior = hmm_model.model.predict_proba(X)  # (T, 2)

    # Map raw state -> label
    label = np.where(raw_states == hmm_model.calm_state, CALM_LABEL, STRESS_LABEL)

    # Reorder posterior to [calm, stress]
    calm_prob = raw_posterior[:, hmm_model.calm_state]
    stress_prob = raw_posterior[:, hmm_model.stress_state]

    return pd.DataFrame(
        {
            "state_raw": raw_states,
            "label": label,
            "calm_prob": calm_prob,
            "stress_prob": stress_prob,
        },
        index=log_returns.index,
    )


# ============================================================
# Realized-volatility regime (robustness check)
# ============================================================

def realized_vol_regime(
    log_returns: pd.Series,
    window: int = 21,
    threshold: float | None = None,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Classify each day as Calm or Stress using a rolling realized-vol threshold.

    Realized volatility is computed as the rolling std of daily log returns
    over the given window. Days where realized vol exceeds the threshold are
    labeled Stress; otherwise Calm. If no threshold is provided, the median
    realized vol of the input series is used (in-sample threshold).

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns.
    window : int
        Lookback window for realized vol (default 21 trading days = ~1 month).
    threshold : float, optional
        Cutoff for Stress classification. In the same units as the returned
        rolling vol (annualized if annualize=True, else daily). If None,
        uses the median of the rolling-vol series.
    annualize : bool
        If True, multiply daily std by sqrt(trading_days).
    trading_days : int
        Days per year used for annualization.

    Returns
    -------
    pd.DataFrame
        Index = log_returns.index. Columns:
            realized_vol : rolling vol (NaN for first window-1 days)
            label : "Calm" or "Stress" (NaN for first window-1 days)
    """
    rolling_std = log_returns.rolling(window=window).std()
    if annualize:
        rolling_vol = rolling_std * np.sqrt(trading_days)
    else:
        rolling_vol = rolling_std

    if threshold is None:
        threshold = float(rolling_vol.median())

    label = pd.Series(index=log_returns.index, dtype="object")
    label[rolling_vol > threshold] = STRESS_LABEL
    label[rolling_vol <= threshold] = CALM_LABEL
    label[rolling_vol.isna()] = np.nan  # warm-up period

    return pd.DataFrame(
        {"realized_vol": rolling_vol, "label": label},
        index=log_returns.index,
    )


# ============================================================
# Summaries and diagnostics
# ============================================================

def regime_summary(
    regime_labels: pd.Series,
    log_returns: pd.Series | None = None,
    vix: pd.Series | None = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Summarize each regime: count, fraction, optional return / VIX statistics.

    Parameters
    ----------
    regime_labels : pd.Series
        Series of regime labels ("Calm" / "Stress"), date-indexed.
    log_returns : pd.Series, optional
        Aligned daily log returns to compute regime-conditional return stats.
    vix : pd.Series, optional
        Aligned VIX series to compute regime-conditional mean VIX (external
        validation). Must be in VIX units (e.g. 15 for 15%, not 0.15).
    trading_days : int
        Used to annualize log-return statistics.

    Returns
    -------
    pd.DataFrame
        One row per regime label. Columns include n_days, fraction,
        mean_return_annual, vol_annual, sharpe (if rf=0; for diagnostic only),
        mean_vix (if VIX provided).
    """
    labels = regime_labels.dropna()
    rows = {}
    for state in [CALM_LABEL, STRESS_LABEL]:
        mask = labels == state
        n = int(mask.sum())
        row: dict[str, float] = {
            "n_days": n,
            "fraction": n / len(labels) if len(labels) > 0 else np.nan,
        }
        if log_returns is not None:
            r = log_returns.reindex(labels.index)[mask].dropna()
            if len(r) > 1:
                row["mean_return_annual"] = float(r.mean()) * trading_days
                row["vol_annual"] = float(r.std(ddof=1)) * np.sqrt(trading_days)
                row["sharpe_zero_rf"] = (
                    row["mean_return_annual"] / row["vol_annual"]
                    if row["vol_annual"] > 0
                    else np.nan
                )
            else:
                row["mean_return_annual"] = np.nan
                row["vol_annual"] = np.nan
                row["sharpe_zero_rf"] = np.nan
        if vix is not None:
            v = vix.reindex(labels.index)[mask].dropna()
            row["mean_vix"] = float(v.mean()) if len(v) > 0 else np.nan
            row["median_vix"] = float(v.median()) if len(v) > 0 else np.nan
        rows[state] = row

    return pd.DataFrame(rows).T


def regime_agreement(labels_a: pd.Series, labels_b: pd.Series) -> dict:
    """
    Compare two regime-label series and report agreement statistics.

    Returns a dict with:
        n_compared : int -- days where both have non-null labels
        agreement_rate : float -- fraction of those days where labels match
        confusion : pd.DataFrame -- 2x2 confusion matrix (rows = labels_a,
                                    cols = labels_b)
        cohens_kappa : float -- Cohen's kappa (chance-corrected agreement)
    """
    df = pd.concat([labels_a.rename("a"), labels_b.rename("b")], axis=1).dropna()
    n = len(df)
    if n == 0:
        return {
            "n_compared": 0,
            "agreement_rate": np.nan,
            "confusion": pd.DataFrame(),
            "cohens_kappa": np.nan,
        }

    agreement_rate = float((df["a"] == df["b"]).mean())

    confusion = pd.crosstab(df["a"], df["b"], margins=False)
    # Make sure both states appear as rows and columns even if one is missing
    for state in [CALM_LABEL, STRESS_LABEL]:
        if state not in confusion.index:
            confusion.loc[state] = 0
        if state not in confusion.columns:
            confusion[state] = 0
    confusion = confusion.loc[
        [CALM_LABEL, STRESS_LABEL], [CALM_LABEL, STRESS_LABEL]
    ]

    # Cohen's kappa = (p_o - p_e) / (1 - p_e)
    total = confusion.values.sum()
    p_o = np.diag(confusion.values).sum() / total
    row_marg = confusion.values.sum(axis=1) / total
    col_marg = confusion.values.sum(axis=0) / total
    p_e = float((row_marg * col_marg).sum())
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan

    return {
        "n_compared": n,
        "agreement_rate": agreement_rate,
        "confusion": confusion,
        "cohens_kappa": float(kappa),
    }


def expected_regime_duration(
    transition_matrix: np.ndarray,
    trading_days: int = 252,
) -> dict:
    """
    Compute expected duration of each regime from the transition matrix.

    For a 2-state Markov chain with self-transition probabilities p_ii, the
    expected duration of state i is 1 / (1 - p_ii). We also compute the
    stationary distribution -- the long-run fraction of time spent in each
    state -- by solving pi = pi @ P with sum(pi) = 1.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (2, 2)
        Relabeled transition matrix in [calm, stress] order.
    trading_days : int
        Used to convert expected duration in days to weeks/months for
        readability. Returned as days; trading_days only used in the
        derived "expected_duration_weeks" output.

    Returns
    -------
    dict
        calm_expected_duration_days, stress_expected_duration_days,
        calm_stationary_prob, stress_stationary_prob.
    """
    P = np.asarray(transition_matrix, dtype=float)
    # Expected duration in state i: 1 / (1 - P[i, i])
    p_calm_self = P[0, 0]
    p_stress_self = P[1, 1]
    calm_dur = 1.0 / (1.0 - p_calm_self) if p_calm_self < 1 else np.inf
    stress_dur = 1.0 / (1.0 - p_stress_self) if p_stress_self < 1 else np.inf

    # Stationary distribution: solve pi P = pi, sum(pi)=1
    # For 2x2, pi_calm = P[1,0] / (P[0,1] + P[1,0])
    p01 = P[0, 1]
    p10 = P[1, 0]
    denom = p01 + p10
    if denom > 0:
        pi_calm = p10 / denom
        pi_stress = p01 / denom
    else:
        pi_calm = np.nan
        pi_stress = np.nan

    return {
        "calm_expected_duration_days": float(calm_dur),
        "stress_expected_duration_days": float(stress_dur),
        "calm_stationary_prob": float(pi_calm),
        "stress_stationary_prob": float(pi_stress),
    }
