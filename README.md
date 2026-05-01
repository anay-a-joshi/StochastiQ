# StochastiQ

> **Multi-Model Portfolio Optimization with Stochastic Simulation, Options Overlay, and Regime-Conditional Validation.**
> A complete falsifiable empirical pipeline built end-to-end in Python, validated with statistically rigorous out-of-sample inference.

[![Live Site](https://img.shields.io/badge/🌐_Live_Site-anay--a--joshi.github.io%2FStochastiQ-00d4ff?style=for-the-badge)](https://anay-a-joshi.github.io/StochastiQ/)
[![Final Report](https://img.shields.io/badge/📄_Final_Report-PDF_(36_pp)-ff006e?style=for-the-badge)](https://github.com/anay-a-joshi/StochastiQ/blob/main/reports/StochastiQ_Final_Report.pdf)

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-complete-00ff9d.svg)]()
[![Course](https://img.shields.io/badge/MGT_6081--A-Derivative_Securities-a855f7.svg)]()
![Profile Views](https://komarev.com/ghpvc/?username=anay-a-joshi&color=06B6D4)

---

## 👥 Team

| Member | Role |
|--------|------|
| **Anay Abhijit Joshi** | Project architecture, source code under `src/`, validation framework (Phase 6), regime-conditional analysis (Phase 7), final report |
| **Vidhi Chaturvedi** | Markowitz baseline (Phase 2), cross-asset correlation analysis, Q-Q goodness-of-fit (Phase 3) |
| **Sriya Nelluri** | Stochastic-process calibration (Phase 3) — Merton & Heston implementations, Phase 4 contributions |
| **Himanshu Bhatt** | BSM options overlay (Phase 5) — covered call, protective put, collar; Greek profiles |

**Course:** MGT 6081-A: Derivative Securities · **Instructor:** Prof. Satyajit Karnik · **Spring 2026** · **Georgia Tech MS in Quantitative & Computational Finance**

---

## 🎯 What this project does

We built a complete pipeline for **multi-model robust portfolio construction** on a 7-asset universe (AAPL, MSFT, JPM, JNJ, XOM, SPY, GLD), and validated it out-of-sample with statistical rigor. The project is framed as **Project Idea 6** (independent direction, instructor-approved) and incorporates the full blueprint of **Project Idea 1** plus substantial extensions on validation, regime-conditioning, and coherent-risk-measure (CVaR) decomposition.

The central empirical question: *does a model-robust portfolio (min-max worst-case across four stochastic-process models) outperform the naïve 1/N benchmark out-of-sample?*

## 🔑 Headline findings

> **Same data. Different verdict.** A strategy designed to manage state-contingent risk must be evaluated using a state-contingent risk metric.

| Phase 6 — Sharpe Ratio (unconditional) | Phase 7 — CVaR by Regime |
|---|---|
| Δ = **−0.443**, 95% CI = [−1.40, +0.21], *p* = 0.288 | All **7 of 7** optimized portfolios beat EW on Stress CVaR |
| **Verdict: Underpowered** (textbook DGU result, reported honestly) | **Per-Heston: +62 bps**, Per-Merton: +60 bps, Min-max: +53 bps |
| Sharpe averages over both regimes — wrong metric for tail-insurance | Cost of insurance: all 7 are *worse* than EW on Calm CVaR ✓ |

**Direct answer to "which models are most pertaining for the future?"** → **Heston** stochastic volatility (best Stress CVaR), **Merton** jump-diffusion a close second. Both capture features of the empirical return distribution (vol clustering, leverage, jumps) that GBM and CEV cannot.

## 🛠️ Methodology

| Phase | Description | Technique |
|-------|-------------|-----------|
| **1. Data** | 5 years of daily prices for 7 assets, train/test split at 2024-12-31 | yfinance · pandas |
| **2. Markowitz Baseline** | Mean-variance frontier with 5 canonical portfolios | scipy.optimize · SLSQP |
| **3. Stochastic Calibration** | GBM, Merton, CEV, Heston for each ticker (154 parameters total) | MLE · method-of-moments · moment-matching |
| **4. Robust Optimization** | 5,000 correlated MC paths/model → per-model + 3 robust portfolios | Cholesky correlation · Ben-Tal-Nemirovski min-max |
| **5. Options Overlay** | BSM pricing for covered call, protective put, collar | Black-Scholes-Merton · Greeks |
| **6. OOS Validation** | Paired stationary block-bootstrap on Sharpe difference | Politis-Romano · pre-registered decision rules |
| **7. Regime Analysis** | 2-state HMM, VIX validation, regime-conditional CVaR | hmmlearn · Holm-Bonferroni across 14 tests |
| **8. Synthesis & Report** | Cross-phase aggregation + 34-page LaTeX report | LaTeX · pdflatex + bibtex |

## 📊 Results at a glance

- **8** notebooks orchestrating the full pipeline
- **30** generated figures across all phases
- **34**-page LaTeX report with **21** academic references
- **5,000** Monte Carlo paths × **4** models × **7** assets = **140,000** simulated paths per evaluation
- **B = 5,000** bootstrap replicates with rule-of-thumb block length L ≈ n^(1/3)
- **Honest disclosure** of 7 explicit limitations (Min-max ≡ Per-GBM degeneracy, in-sample inference for Phase 7, only 15 OOS Stress days, etc.)

## 📂 Repository structure

```
StochastiQ/
├── src/                              # Production source code
│   ├── config.py                     # Global constants (rf, trading days, seed=42)
│   ├── data/loaders.py               # yfinance ingestion, return computation
│   ├── models/                       # Stochastic process implementations
│   │   ├── gbm.py · merton.py · cev.py · heston.py
│   ├── simulation/monte_carlo.py     # Cholesky-correlated path generation
│   ├── optimization/
│   │   ├── markowitz.py              # Phase 2 mean-variance frontier
│   │   └── robust.py                 # Phase 4 per-model + min-max + blends
│   ├── options/                      # Phase 5 BSM overlay
│   │   ├── bsm.py · strategies.py
│   ├── validation/bootstrap.py       # Politis-Romano stationary bootstrap
│   └── regime/                       # Phase 7 HMM module
│       ├── detection.py · evaluation.py
├── notebooks/                        # 8 Jupyter notebooks (one per phase)
├── data/processed/                   # Parquet result files (gitignored)
├── reports/
│   ├── figures/                      # All 30 generated PNGs
│   ├── StochastiQ_Final_Report.tex   # Source
│   ├── StochastiQ_Final_Report.pdf   # 34-page compiled report
│   ├── references.bib                # 21 academic references
│   └── build_report.sh               # One-command compile script
├── docs/
│   └── index.html                    # GitHub Pages landing site
├── tests/                            # Unit tests
├── requirements.txt
├── LICENSE                           # MIT
└── README.md                         # This file
```

## 🚀 Reproduce all results in 4 commands

```bash
git clone https://github.com/anay-a-joshi/StochastiQ.git
cd StochastiQ
python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
jupyter lab   # then run notebooks 01 through 08 in order
```

Total runtime: **~8 minutes** on a 2024 Apple Silicon MacBook Pro, dominated by the bootstrap in notebooks 06 and 07. All randomness is seeded with `RANDOM_SEED = 42`.

To rebuild the PDF report:
```bash
cd reports/ && bash build_report.sh
```

## 📚 Key references

The work is anchored in foundational literature on portfolio theory, stochastic-process modeling, coherent risk measures, and statistical inference:

- Markowitz (1952) — mean-variance portfolio theory
- Black & Scholes (1973), Merton (1973, 1976) — option pricing & jump-diffusion
- Cox (1996), Heston (1993) — CEV & stochastic volatility
- Hamilton (1989), Ang & Bekaert (2002) — regime-switching models
- Politis & Romano (1994) — stationary block-bootstrap
- Artzner et al. (1999), Rockafellar & Uryasev (2002) — coherent risk measures, CVaR
- DeMiguel, Garlappi & Uppal (2009) — the 1/N puzzle
- Holm (1979) — multiple-testing correction

Full bibliography of 21 references in [`reports/references.bib`](reports/references.bib).

## 🌐 Project website

Visit the interactive landing page: **[anay-a-joshi.github.io/StochastiQ](https://anay-a-joshi.github.io/StochastiQ/)**

Built with vanilla HTML/CSS/JS, neon-style visual design (Robinhood/Tesla aesthetic), splash intro animation, and responsive layout for mobile + desktop.

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<sub>Designed & developed by Anay Abhijit Joshi · MGT 6081-A Spring 2026 · Georgia Institute of Technology</sub>
