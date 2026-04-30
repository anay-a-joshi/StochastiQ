"""
Project-wide configuration constants for StochastiQ.

Centralizing these here ensures every notebook and module uses the same
values for the risk-free rate, trading day convention, and other
parameters that appear in multiple phases of the project.
"""

from __future__ import annotations

# ============================================================
# Market conventions
# ============================================================

# Annualized risk-free rate.
# Set to 4.0%, reflecting the prevailing short-term Treasury environment
# at the time of analysis (Q1 2026). Used for Sharpe/Sortino ratios,
# Black-Scholes option pricing, and stochastic-rate model anchoring.
RISK_FREE_RATE: float = 0.04

# Trading days per year. The standard US equity convention.
TRADING_DAYS: int = 252

# ============================================================
# Risk metrics
# ============================================================

# Default confidence level for VaR and CVaR computations.
DEFAULT_CONFIDENCE: float = 0.95

# ============================================================
# Simulation defaults
# ============================================================

# Default number of Monte Carlo paths for forward simulation.
# 5,000 paths gives a stable distribution estimate for portfolio statistics
# without being prohibitively slow on a laptop.
DEFAULT_N_PATHS: int = 5_000

# Default forward simulation horizon (one trading year).
DEFAULT_HORIZON_DAYS: int = 252

# Random seed for reproducibility across simulation runs.
RANDOM_SEED: int = 42
