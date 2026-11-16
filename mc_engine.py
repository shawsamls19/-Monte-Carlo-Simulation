"""
mc_engine.py
============

Core maths for the portfolio risk project.

This module holds every calculation used by both the notebook and the
Streamlit dashboard, so the two always agree with each other:

    * downloading price history and turning it into daily returns
    * expected return / volatility of a weighted portfolio
    * Value at Risk (VaR) and Conditional VaR (CVaR, a.k.a. Expected Shortfall)
      using three approaches - historical, parametric (normal and Student-t)
      and Monte Carlo simulation

Conventions used throughout
---------------------------
* ``alpha`` is the tail size in percent. ``alpha=5`` means "95% confidence".
* VaR and CVaR are returned as POSITIVE numbers that represent a loss
  (e.g. 0.07 = a 7% loss), so a bigger number always means more risk.
* ``horizon`` is measured in trading days.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import norm, t


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_prices(tickers: list[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download daily closing prices from Yahoo Finance.

    Returns a DataFrame with one column per ticker. Columns that come back
    completely empty (bad or delisted tickers) are removed so they cannot
    poison the covariance matrix later on.
    """
    import yfinance as yf  # imported here so the rest of the module works offline

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    close = raw["Close"]
    # A single ticker comes back as a Series - normalise to a DataFrame
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close.dropna(axis=1, how="all")


def demo_prices(tickers: list[str], days: int = 550, seed: int = 12) -> pd.DataFrame:
    """Create realistic-looking synthetic prices for offline use.

    Used by the dashboard when Yahoo Finance cannot be reached, so the app
    still runs end-to-end. Returns are drawn from a fat-tailed (Student-t)
    distribution with a shared market factor, which gives the stocks a
    believable positive correlation.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    market = rng.standard_t(df=5, size=days) * 0.008             # common market move
    betas = rng.uniform(0.6, 1.3, size=n)                         # sensitivity to the market
    idio = rng.standard_t(df=5, size=(days, n)) * rng.uniform(0.006, 0.013, size=n)
    drift = rng.uniform(0.0004, 0.0011, size=n)                   # small positive daily drift
    rets = drift + market[:, None] * betas + idio

    dates = pd.bdate_range(end=dt.date.today(), periods=days)
    start_px = rng.uniform(5, 120, size=n)
    prices = start_px * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Turn prices into daily % returns plus their mean vector and covariance matrix."""
    returns = prices.pct_change().dropna(how="any")
    return returns, returns.mean(), returns.cov()


# ---------------------------------------------------------------------------
# 2. Portfolio statistics
# ---------------------------------------------------------------------------

def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series,
                          cov_matrix: pd.DataFrame, horizon: int) -> tuple[float, float]:
    """Expected return and volatility of the portfolio over ``horizon`` days.

    * Return scales linearly with time:          mu_p * T
    * Volatility scales with the square root:    sigma_p * sqrt(T)
    where sigma_p = sqrt(w' . Cov . w).
    """
    exp_ret = float(np.sum(mean_returns.values * weights) * horizon)
    vol = float(np.sqrt(weights @ cov_matrix.values @ weights) * np.sqrt(horizon))
    return exp_ret, vol


# ---------------------------------------------------------------------------
# 3. Historical VaR / CVaR  (no distribution assumed - uses what really happened)
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, alpha: float = 5) -> float:
    """The alpha-th percentile of the observed daily returns (a negative number = loss)."""
    return float(np.percentile(returns, alpha))


def historical_cvar(returns: pd.Series, alpha: float = 5) -> float:
    """Average of every daily return at or below the historical VaR cut-off."""
    cutoff = historical_var(returns, alpha)
    return float(returns[returns <= cutoff].mean())


# ---------------------------------------------------------------------------
# 4. Parametric VaR / CVaR  (assume a distribution shape, fit it to mean & std)
# ---------------------------------------------------------------------------

def var_parametric(port_ret: float, port_std: float, distribution: str = "normal",
                   alpha: float = 5, dof: int = 6) -> float:
    """Closed-form VaR for a normal or Student-t distribution.

    The Student-t quantile is rescaled by sqrt((nu-2)/nu) so that its
    variance matches the portfolio variance; fewer degrees of freedom
    (``dof``) means fatter tails.
    """
    a = alpha / 100
    if distribution == "normal":
        return norm.ppf(1 - a) * port_std - port_ret
    if distribution == "t-distribution":
        nu = dof
        return np.sqrt((nu - 2) / nu) * t.ppf(1 - a, nu) * port_std - port_ret
    raise ValueError("distribution must be 'normal' or 't-distribution'")


def cvar_parametric(port_ret: float, port_std: float, distribution: str = "normal",
                    alpha: float = 5, dof: int = 6) -> float:
    """Closed-form CVaR (expected loss inside the tail) for normal or Student-t."""
    a = alpha / 100
    if distribution == "normal":
        return (a ** -1) * norm.pdf(norm.ppf(a)) * port_std - port_ret
    if distribution == "t-distribution":
        nu = dof
        x_anu = t.ppf(a, nu)
        return (-1 / a) * (1 - nu) ** -1 * (nu - 2 + x_anu ** 2) * t.pdf(x_anu, nu) * port_std - port_ret
    raise ValueError("distribution must be 'normal' or 't-distribution'")


# ---------------------------------------------------------------------------
# 5. Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, weights: np.ndarray,
                       horizon: int, n_sims: int, initial_value: float,
                       seed: int | None = None) -> np.ndarray:
    """Simulate many possible future paths of the portfolio value.

    How it works
    ------------
    1. Draw independent standard-normal shocks Z for every day, stock and path.
    2. Multiply by the Cholesky factor L of the covariance matrix (Cov = L L')
       so the simulated stocks move together the way they did historically.
    3. Add the mean daily return, combine stocks with the portfolio weights,
       and compound day by day starting from ``initial_value``.

    Returns
    -------
    ndarray of shape (n_sims, horizon + 1) - column 0 is the starting value.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(weights)

    L = np.linalg.cholesky(cov_matrix.values)                      # computed once, not per path
    Z = rng.standard_normal(size=(n_sims, horizon, n_assets))
    daily_asset_rets = mean_returns.values + Z @ L.T              # correlated daily returns
    daily_port_rets = daily_asset_rets @ weights                   # (n_sims, horizon)

    growth = np.cumprod(1 + daily_port_rets, axis=1) * initial_value
    start = np.full((n_sims, 1), initial_value)
    return np.hstack([start, growth])


def mc_var(final_values: np.ndarray, alpha: float = 5) -> float:
    """Portfolio value at the alpha-th percentile of simulated outcomes."""
    return float(np.percentile(final_values, alpha))


def mc_cvar(final_values: np.ndarray, alpha: float = 5) -> float:
    """Average portfolio value across the worst alpha% of simulated outcomes."""
    cutoff = mc_var(final_values, alpha)
    return float(final_values[final_values <= cutoff].mean())


# ---------------------------------------------------------------------------
# 6. One call that runs everything (used by the dashboard)
# ---------------------------------------------------------------------------

def risk_summary(returns: pd.DataFrame, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 weights: np.ndarray, horizon: int, initial_value: float, alpha: float,
                 dof: int, paths: np.ndarray) -> pd.DataFrame:
    """Return a tidy table of VaR and CVaR (in currency) for all four methods."""
    port_daily = returns @ weights
    exp_ret, vol = portfolio_performance(weights, mean_returns, cov_matrix, horizon)

    # Historical figures are daily, scaled to the horizon with the square-root-of-time rule
    h_var = -historical_var(port_daily, alpha) * np.sqrt(horizon)
    h_cvar = -historical_cvar(port_daily, alpha) * np.sqrt(horizon)

    final = paths[:, -1]
    rows = [
        ("Historical", h_var, h_cvar),
        ("Parametric - Normal",
         var_parametric(exp_ret, vol, "normal", alpha), cvar_parametric(exp_ret, vol, "normal", alpha)),
        ("Parametric - Student-t",
         var_parametric(exp_ret, vol, "t-distribution", alpha, dof),
         cvar_parametric(exp_ret, vol, "t-distribution", alpha, dof)),
        ("Monte Carlo",
         (initial_value - mc_var(final, alpha)) / initial_value,
         (initial_value - mc_cvar(final, alpha)) / initial_value),
    ]
    df = pd.DataFrame(rows, columns=["Method", "VaR %", "CVaR %"])
    df["VaR $"] = df["VaR %"] * initial_value
    df["CVaR $"] = df["CVaR %"] * initial_value
    return df
