"""
Data loading utilities for the StochastiQ project.

This module provides functions to fetch historical price data from Yahoo Finance,
clean it, compute returns, and save it in efficient formats.

All functions are designed to be reusable across notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# Default Universe
# ============================================================

DEFAULT_UNIVERSE: list[str] = [
    "AAPL",  # Large-cap technology
    "MSFT",  # Large-cap technology
    "JPM",   # Financials
    "JNJ",   # Defensive healthcare
    "XOM",   # Energy
    "SPY",   # Broad market ETF
    "GLD",   # Gold ETF (crisis hedge)
]

ASSET_DESCRIPTIONS: dict[str, str] = {
    "AAPL": "Apple Inc. (Large-cap Technology)",
    "MSFT": "Microsoft Corp. (Large-cap Technology)",
    "JPM":  "JPMorgan Chase & Co. (Financials)",
    "JNJ":  "Johnson & Johnson (Defensive Healthcare)",
    "XOM":  "Exxon Mobil Corp. (Energy)",
    "SPY":  "SPDR S&P 500 ETF (Broad Market)",
    "GLD":  "SPDR Gold Trust (Crisis Hedge)",
}


# ============================================================
# Data Acquisition
# ============================================================

def fetch_prices(
    tickers: Iterable[str] = DEFAULT_UNIVERSE,
    start: str = "2020-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    auto_adjust : bool
        If True, prices are adjusted for splits and dividends. Default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns,
        containing adjusted closing prices.
    """
    tickers = list(tickers)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    # yfinance returns a multi-index when multiple tickers are passed.
    # We extract just the 'Close' column.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker case
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # Reorder columns to match input order
    prices = prices[tickers]

    return prices


# ============================================================
# Data Quality
# ============================================================

def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality summary for a price DataFrame.

    Returns a per-ticker summary of:
    - first/last available date
    - number of trading days
    - missing values
    - number of zero or negative prices (data errors)
    """
    report = pd.DataFrame(index=prices.columns)
    report["first_date"] = prices.apply(lambda s: s.first_valid_index())
    report["last_date"] = prices.apply(lambda s: s.last_valid_index())
    report["n_observations"] = prices.count()
    report["n_missing"] = prices.isna().sum()
    report["n_zero_or_negative"] = (prices <= 0).sum()
    report["min_price"] = prices.min()
    report["max_price"] = prices.max()
    return report


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a price DataFrame.

    - Forward-fill at most 1 missing day (handles isolated holidays/glitches)
    - Drop any remaining rows with NaNs (ensures aligned series across tickers)
    """
    cleaned = prices.ffill(limit=1).dropna()
    return cleaned


# ============================================================
# Return Computation
# ============================================================

def compute_returns(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute simple and log returns from a price DataFrame.

    Returns
    -------
    dict with keys:
        'simple' : (P_t / P_{t-1}) - 1
        'log'    : ln(P_t / P_{t-1})
    """
    simple_returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    return {
        "simple": simple_returns,
        "log": log_returns,
    }


# ============================================================
# Summary Statistics
# ============================================================

def annualized_summary(
    log_returns: pd.DataFrame,
    trading_days: int = 252,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Compute annualized return statistics for each asset.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    trading_days : int
        Number of trading days per year (default 252).
    risk_free_rate : float
        Annualized risk-free rate for Sharpe ratio (default 4%).

    Returns
    -------
    pd.DataFrame
        Per-asset summary with annualized return, volatility, Sharpe,
        skewness, and excess kurtosis.
    """
    daily_mean = log_returns.mean()
    daily_std = log_returns.std()

    annual_return = daily_mean * trading_days
    annual_vol = daily_std * np.sqrt(trading_days)
    sharpe = (annual_return - risk_free_rate) / annual_vol

    summary = pd.DataFrame({
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "skewness": log_returns.skew(),
        "excess_kurtosis": log_returns.kurtosis(),
    })

    return summary


# ============================================================
# Persistence
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "parquet",
) -> Path:
    """
    Save a DataFrame to disk in the specified format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination path (without extension).
    fmt : str
        One of 'parquet' or 'csv'.

    Returns
    -------
    Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        full_path = path.with_suffix(".parquet")
        df.to_parquet(full_path, engine="pyarrow")
    elif fmt == "csv":
        full_path = path.with_suffix(".csv")
        df.to_csv(full_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return full_path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from parquet or CSV based on file extension."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
