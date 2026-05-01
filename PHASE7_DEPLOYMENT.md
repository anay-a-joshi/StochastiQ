# Phase 7 — Deployment Notes

## What's in this delivery

```
phase7/
├── src/regime/
│   ├── __init__.py                          (1.2 KB — public API)
│   ├── detection.py                         (15.6 KB — HMM + realized-vol regime classifiers)
│   └── evaluation.py                        (16.7 KB — NAV reconstruction, regime-conditional metrics, bootstrap inference, CVaR)
└── notebooks/
    └── 07_regime_analysis.ipynb             (57.3 KB — 40 cells: 14 markdown, 26 code)
```

## Installation steps (one terminal session)

### 1. Stage the files

```bash
cd ~/Downloads/StochastiQ

# Move source files
mv ~/Downloads/phase7/src/regime src/regime

# Move notebook
mv ~/Downloads/phase7/notebooks/07_regime_analysis.ipynb notebooks/

# (If the mv arguments don't match your download path, adjust accordingly.)
```

### 2. Update requirements

Add this line to `requirements.txt`:

```
hmmlearn>=0.3.0
```

Then install:

```bash
source venv/bin/activate
pip install hmmlearn
```

### 3. Verify imports work

Before opening the notebook, sanity-check the new module:

```bash
python -c "from src.regime import fit_hmm_regime, bootstrap_regime_sharpe_diff; print('imports OK')"
```

If this fails, the most likely cause is a parameter-name mismatch with your existing `src/validation/bootstrap.py` (see Troubleshooting below).

### 4. Close the notebook tab in JupyterLab BEFORE opening 07

Standard project rule from the continuation summary: if Jupyter has the file cached, the auto-save will overwrite the staged version. Close the tab, then reopen.

### 5. Run the notebook

The notebook is self-contained. It will:

1. Load processed log returns from `data/processed/log_returns.parquet` if present, otherwise refetch from Yahoo Finance (filenames tried, in order: `log_returns.parquet`, `returns_log.parquet`, `log_returns_daily.parquet`).
2. Pull VIX (`^VIX`) from Yahoo Finance for the same period.
3. Fit the HMM on training-period SPY only, predict labels for both training and OOS.
4. Load Phase 4 portfolio weights from `data/processed/` (filenames tried, in order: `phase4_portfolio_weights.parquet`, `portfolio_weights.parquet`, `robust_weights.parquet`, `phase4_weights.parquet`, `weights.parquet`). Schemas tolerated: long format (`portfolio`/`ticker`/`weight` columns), wide with tickers as columns, or wide with tickers as rows. Equal-Weighted is added programmatically if absent.
5. Generate 6 figures into `reports/figures/` and 12 parquet files into `data/processed/phase7_*.parquet`.

Total runtime estimate: ~30-60 seconds for the 5000-replicate bootstrap on 1300 training days.

## Troubleshooting

### Issue: ImportError on `src.validation.bootstrap`

If you get `cannot import name 'bootstrap_sharpe_ci' from 'src.validation.bootstrap'`, your Phase 6 implementation may have named the function differently. Check:

```bash
grep -n "^def " src/validation/bootstrap.py
```

If the function is named e.g. `block_bootstrap_sharpe`, edit `src/regime/evaluation.py` line 49-52 to import that name instead, e.g.:

```python
from src.validation.bootstrap import (
    block_bootstrap_sharpe as _bootstrap_sharpe_ci,
    block_bootstrap_sharpe_diff as _bootstrap_sharpe_diff,
)
```

### Issue: Bootstrap function rejects keyword arguments

The `evaluation.py` module uses `inspect.signature` to adapt between common parameter-name variations (`n_boot`/`n_bootstrap`, `block_size`/`block_length`/`expected_block_length`, `confidence`/`confidence_level`, `random_state`/`seed`, `rf`/`risk_free_rate`/`risk_free`, `trading_days`/`periods_per_year`/`annualization_factor`).

If your bootstrap.py uses a parameter name not in this list, add it to `_PARAM_ALIASES` at the top of `evaluation.py`:

```python
_PARAM_ALIASES: dict[str, list[str]] = {
    "n_boot": ["n_boot", "n_bootstrap", "n_samples", "B", "your_name_here"],
    ...
}
```

### Issue: KeyError on `point_estimate` / `ci_lower` / `ci_upper`

The notebook expects `bootstrap_sharpe_ci` and `bootstrap_sharpe_diff` to return dicts containing keys `point_estimate`, `ci_lower`, `ci_upper`, plus optionally `p_value`, `boot_mean`, `boot_std`, `sharpe_a`, `sharpe_b`. If your Phase 6 implementation uses different key names, the adapter at the bottom of `bootstrap_regime_sharpe_ci` (lines ~315-330 of `evaluation.py`) handles common aliases (`estimate`/`sharpe`/`point` for the point estimate; `lower`/`lo`/`lower_bound` for CI lower; `upper`/`hi`/`upper_bound` for CI upper). Add aliases there if needed.

### Issue: Portfolio names not matching

The notebook uses `find_portfolio()` with name hints (`"min-max"`, `"equal-weight"`, etc.) that are tolerant of underscores, capitalization, and hyphens. If your Phase 4 notebook saved weights under a name that doesn't pattern-match — e.g. `"Robust Worst Case"` for what I'm calling Min-max — edit the hints in the notebook cell that calls `find_portfolio`.

### Issue: VIX fetch fails

If yfinance returns an empty frame for `^VIX`, possible causes:
- Rate limiting → wait a minute and re-run.
- Network issue → check internet.
- Yahoo Finance changed the VIX schema — fall back to FRED's `VIXCLS` series via `pandas_datareader`.

The notebook will fail loudly rather than silently if VIX is empty; you can comment out the VIX cells (Section 3) and rerun — the HMM analysis is independent of VIX, which is only used for external validation.

## What this enables for Phase 8

Phase 7 saves 12 parquet files that Phase 8 (final aggregation) will consume:

```
data/processed/phase7_hmm_regime_labels_train.parquet
data/processed/phase7_hmm_regime_labels_oos.parquet
data/processed/phase7_realized_vol_regime_train.parquet
data/processed/phase7_regime_summary_train.parquet
data/processed/phase7_regime_summary_oos.parquet
data/processed/phase7_regime_conditional_metrics_train.parquet
data/processed/phase7_regime_conditional_metrics_oos.parquet
data/processed/phase7_bootstrap_sharpe_ci_by_regime.parquet
data/processed/phase7_paired_sharpe_diff_tests.parquet
data/processed/phase7_cvar_by_regime.parquet
data/processed/phase7_verdict_table.parquet
data/processed/phase7_hmm_model_artifacts.parquet
```

Plus 6 figures in `reports/figures/`:
```
07_regime_timeline.png         — SPY price with regime shading + posterior probabilities
07_regime_validation.png       — VIX boxplot by regime + HMM/RV confusion matrix
07_sharpe_by_regime.png        — Grouped bar chart of Sharpe ratios by regime
07_sharpe_forest_plot.png      — Forest plot of bootstrap Sharpe CIs by portfolio × regime
07_minmax_vs_ew_paired.png     — Headline visual: paired Sharpe difference with CIs
07_cvar_by_regime.png          — Tail risk comparison by portfolio by regime
```

## Suggested commit messages

```bash
# After all files are in place and notebook runs cleanly:
git add src/regime/ notebooks/07_regime_analysis.ipynb requirements.txt
git commit -m "Phase 7 complete: HMM regime detection + regime-conditional inference

New module src/regime/ with 2-state Gaussian HMM regime detection on SPY
training returns, deterministic relabeling by within-state variance, and
realized-vol classifier as robustness check. External validation against
VIX (CBOE Volatility Index, ^VIX): regime-conditional VIX summary anchors
HMM labels to canonical fear-gauge.

Regime-conditional evaluation of all 8 Phase 4 portfolios over training
period (2020-01-03 -> 2024-12-31). Headline statistical test: paired
stationary block-bootstrap on (Min-max - Equal-Weighted) Sharpe difference
within Stress regime and within Calm regime separately, reusing Phase 6's
bootstrap.py via signature-tolerant adapter.

Tail-risk evaluation via regime-conditional CVaR_95, anchored in the
Artzner et al. (1999) coherent-risk-measure framework.

OOS regime check (descriptive, post-2024 window) using HMM inference
without refit, preserving train/test discipline.

Notebook 07_regime_analysis.ipynb: 40 cells (14 markdown, 26 code),
executive summary at top, verdict table at bottom with pre-registered
decision rules. Saves 12 parquet outputs and 6 figures.

References: Hamilton (1989), Ang & Bekaert (2002), Politis & Romano (1994),
Artzner et al. (1999), DeMiguel/Garlappi/Uppal (2009)."
```

## Heads-up before Phase 8

After running Phase 7 successfully, please paste the **verdict table** output (the printed pandas DataFrame from the cell that builds `verdict_df`) into the next chat. That gives me the actual run-time results — particularly the headline `Min-max vs Equal-Weighted (Stress)` row — which I'll use to calibrate the Phase 8 final-results aggregation and the LaTeX report's narrative.
