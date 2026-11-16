# Monte Carlo Portfolio Risk Dashboard

How much could a stock portfolio realistically lose? This project answers that with **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)**, calculated in four different ways and shown in an interactive dashboard.

![Dashboard overview](assets/dashboard_overview.png)

## Features

- **Live market data** from Yahoo Finance for any tickers you choose (ASX by default)
- **Monte Carlo simulation** of thousands of correlated future price paths (Cholesky decomposition)
- **Four risk methods compared side by side**: Historical, Parametric (Normal), Parametric (Student-t) and Monte Carlo
- **Plain-English takeaways** that update as you change settings
- **Optional AI explanation** of the results (bring your own Anthropic API key)
- **Interactive controls** for tickers, weights, investment size, horizon, number of simulations and confidence level
- **CSV export** of the results
- Falls back to clearly labelled demo data if market data can't be downloaded

## Screenshots

| Simulated paths | Outcome distribution |
|---|---|
| ![Simulated paths](assets/simulated_paths.png) | ![Outcome distribution](assets/outcome_distribution.png) |

| Method comparison | Portfolio & correlations |
|---|---|
| ![Compare methods](assets/compare_methods.png) | ![Portfolio data](assets/portfolio_data.png) |

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m streamlit run app.py
```

The dashboard opens at http://localhost:8501.

### Optional: AI explanation

The **✨ Explain with AI** button needs an Anthropic API key (https://console.anthropic.com). Use any one of these:

- paste it into the sidebar under **3 · AI summary** (kept for that session only)
- set an environment variable: `export ANTHROPIC_API_KEY=your-key-here`
- create `.streamlit/secrets.toml` containing `ANTHROPIC_API_KEY = "your-key-here"`

`secrets.toml` is listed in `.gitignore`, so your key never gets pushed. The rest of the dashboard works without a key.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: layout, charts and plain-English summary |
| `mc_engine.py` | The maths: data loading, VaR/CVaR methods and the Monte Carlo simulation |
| `Monte_Carlo.ipynb` | Step-by-step notebook walking through each method |
| `.streamlit/config.toml` | Dashboard theme |
| `requirements.txt` | Python dependencies |
| `assets/` | Screenshots used in this README |

## How it works

1. Download daily closing prices and convert them into daily returns.
2. Estimate each stock's mean return and the covariance matrix between stocks.
3. **Historical:** read the worst 5% of real past returns, scaled to the horizon with the square-root-of-time rule.
4. **Parametric:** fit a normal or fat-tailed Student-t distribution and use closed-form formulas.
5. **Monte Carlo:** factor the covariance matrix as Σ = L·Lᵀ, simulate correlated daily returns, compound them into portfolio paths, and read VaR and CVaR from the ending values.

### Key terms

- **VaR (95%)**: the loss that should not be exceeded in 95% of cases.
- **CVaR / Expected Shortfall**: the average loss in the worst 5% of cases. It shows how bad the bad days really are.

## Tech stack

Python · NumPy · pandas · SciPy · Streamlit · Plotly · yfinance

## Credits

The VaR/CVaR methodology is based on the QuantPy tutorial series. The explanations, the refactored engine and the interactive dashboard were written for this project.

## Disclaimer

For learning and exploration only. This is not financial advice, and past performance does not guarantee future results.
