# StochastiQ

> Multi-model portfolio optimization and derivatives strategy framework using GBM, Merton, CEV, and Heston stochastic processes.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-in--development-orange.svg)]()
![Profile Views](https://komarev.com/ghpvc/?username=anay-a-joshi&color=06B6D4)

## Overview

StochastiQ is a quantitative research framework that constructs an optimal equity portfolio from real market data, projects it forward under four stochastic models (Geometric Brownian Motion, Merton jump-diffusion, Constant Elasticity of Variance, and Heston stochastic volatility), identifies portfolio weights that perform robustly across all models, and overlays options to enhance risk-adjusted returns.

This project was developed for **MGT 6081: Derivative Securities** at the **Georgia Institute of Technology** as part of the **MS in Quantitative & Computational Finance** program.

## Research Questions

1. Given a universe of liquid equities and ETFs, what is the optimal portfolio under Markowitz mean-variance optimization?
2. How does that portfolio behave under different stochastic models of future asset dynamics?
3. Which model best captures the regime ahead, and how can options be used to improve risk-adjusted performance?

## Methodology

| Phase | Description |
|-------|-------------|
| 1. Data Acquisition | Pull 5 years of daily prices for a heterogeneous 7-asset universe via `yfinance` |
| 2. Portfolio Optimization | Solve Markowitz mean-variance, Sortino, CVaR-minimization, and risk-parity portfolios |
| 3. Model Calibration | Calibrate GBM, Merton, CEV, and Heston to historical returns with documented procedures |
| 4. Monte Carlo Simulation | Simulate 5,000 forward paths under each model |
| 5. Robust Portfolio | Identify weights that maximize risk-adjusted return across all models |
| 6. Option Overlay | Apply covered calls, protective puts, and collars; analyze portfolio Greeks |
| 7. Out-of-Sample Validation | Test calibrated models and portfolios on held-out data |
| 8. Regime Analysis | Compare model performance across calm and stress regimes |

## Repository Structure

```
StochastiQ/
├── data/                # Raw and processed datasets
├── notebooks/           # End-to-end Jupyter notebooks
├── src/                 # Reusable Python modules
│   ├── data/            # Data loaders (yfinance, FRED)
│   ├── optimization/    # Portfolio optimization algorithms
│   ├── models/          # Stochastic differential equation simulators
│   ├── calibration/     # Model parameter estimation procedures
│   ├── options/         # Black-Scholes pricing and strategies
│   ├── analysis/        # Performance metrics and regime detection
│   └── utils/           # Plotting and helpers
├── reports/             # Final report (LaTeX) and figures
├── tests/               # Unit tests for models and calibration
└── docs/                # Methodology and references
```

## Setup

```bash
# Clone the repository
git clone https://github.com/anay-a-joshi/StochastiQ.git
cd StochastiQ

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

## Notebooks

Run notebooks in numerical order for full reproducibility:

1. `01_data_pull.ipynb` — Acquire and clean historical price data
2. `02_markowitz_optimization.ipynb` — Mean-variance and alternative-objective portfolios
3. `03_model_calibration.ipynb` — Calibrate GBM, Merton, CEV, Heston
4. `04_monte_carlo_simulation.ipynb` — Forward simulation under each model
5. `05_options_overlay.ipynb` — Option strategies and Greek analysis
6. `06_out_of_sample_validation.ipynb` — Test on held-out data
7. `07_regime_analysis.ipynb` — Calm vs. stress regime comparison
8. `08_final_results.ipynb` — Aggregated results and key plots

## Results

*Results will be populated upon project completion.*

## Team

- **Anay Abhijit Joshi** — MS-QCF, Georgia Institute of Technology

## License

MIT — see [LICENSE](LICENSE) for details.

## References

See [docs/references.md](docs/references.md) for the full bibliography.
