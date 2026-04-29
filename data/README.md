# Data Directory

## Structure

- `raw/` — Untouched data pulled from external sources (yfinance, FRED). Never modify these files directly.
- `processed/` — Cleaned, transformed datasets ready for modeling.

## Data Sources

| Source | Type | Coverage |
|--------|------|----------|
| Yahoo Finance (yfinance) | Daily equity and ETF prices | 2020-01-01 onward |
| FRED | Treasury yields | 2020-01-01 onward |

## Universe

- AAPL — Large-cap technology
- MSFT — Large-cap technology
- JPM — Financials
- JNJ — Defensive healthcare
- XOM — Energy
- SPY — Broad market ETF
- GLD — Gold ETF (crisis hedge)

Raw data files are excluded from version control via `.gitignore` and regenerated from `notebooks/01_data_pull.ipynb`.
