"""
Regime detection and regime-conditional portfolio evaluation.

This package implements Phase 7 of the StochastiQ project. It detects
market regimes (Calm vs Stress) using a Gaussian Hidden Markov Model
on SPY daily log returns, validates the regime labels against the VIX
and against a 21-day realized-volatility threshold, and evaluates the
Phase 4 portfolios' performance conditional on regime.

Modules
-------
detection : HMM and realized-vol regime classifiers
evaluation : regime-conditional portfolio metrics and bootstrap inference
"""

from .detection import (
    fit_hmm_regime,
    predict_hmm_regime,
    realized_vol_regime,
    regime_summary,
    regime_agreement,
    expected_regime_duration,
)
from .evaluation import (
    portfolio_nav_from_weights,
    regime_conditional_metrics,
    bootstrap_regime_sharpe_ci,
    bootstrap_regime_sharpe_diff,
    regime_conditional_cvar,
)

__all__ = [
    "fit_hmm_regime",
    "predict_hmm_regime",
    "realized_vol_regime",
    "regime_summary",
    "regime_agreement",
    "expected_regime_duration",
    "portfolio_nav_from_weights",
    "regime_conditional_metrics",
    "bootstrap_regime_sharpe_ci",
    "bootstrap_regime_sharpe_diff",
    "regime_conditional_cvar",
]
