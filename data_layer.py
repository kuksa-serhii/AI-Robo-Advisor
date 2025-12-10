import pandas as pd
import streamlit as st

from config import UNIVERSE_CSV_PATH, PRICES_CSV_PATH

import subprocess
import sys

subprocess.run([sys.executable, "download_data.py"], check=True)


@st.cache_data
def load_universe():
    """
    Load ETF universe from CSV.
    Expected column: 'ticker'.
    """
    df = pd.read_csv(UNIVERSE_CSV_PATH)
    if "ticker" not in df.columns:
        raise ValueError("universe.csv must have a 'ticker' column")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


@st.cache_data
def load_prices():
    """
    Load historical prices from CSV.
    First column: date (index), remaining columns: tickers.
    """
    df = pd.read_csv(PRICES_CSV_PATH, index_col=0, parse_dates=True)
    df.columns = [str(c).upper() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df


def compute_returns_mu_cov(prices_df: pd.DataFrame, annualization_factor: int = 252):
    """
    Compute daily returns, annualized expected returns (mu) and covariance matrix (Sigma).
    """
    returns = prices_df.pct_change().dropna(how="all").dropna(axis=1, how="all")
    mu = returns.mean() * annualization_factor
    cov = returns.cov() * annualization_factor
    return returns, mu, cov


def apply_exclusions(mu: pd.Series, cov: pd.DataFrame, exclude_tickers):
    """
    Remove excluded tickers from mu and cov.
    """
    if not exclude_tickers:
        return mu, cov

    exclude_tickers = [t.upper().strip() for t in exclude_tickers]
    keep_tickers = [t for t in mu.index if t not in exclude_tickers]

    mu_f = mu.loc[keep_tickers]
    cov_f = cov.loc[keep_tickers, keep_tickers]
    return mu_f, cov_f
