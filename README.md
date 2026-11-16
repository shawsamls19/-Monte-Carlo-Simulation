# Monte Carlo Portfolio Risk Dashboard

An interactive dashboard that estimates how much a stock portfolio could realistically lose. It measures **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)** using four different methods and compares them side by side.

## Features

- Live market data from Yahoo Finance for any tickers (ASX by default)
- Monte Carlo simulation of thousands of correlated future price paths
- Four risk methods compared: Historical, Parametric (Normal), Parametric (Student-t) and Monte Carlo
- Interactive controls for tickers, weights, investment size, time horizon, number of simulations and confidence level
- Charts for simulated paths, outcome distribution, method comparison, portfolio weights, correlations and historical growth
- Plain-English takeaways that update as settings change
- Optional AI explanation of the results
- CSV export of the results

## Tools Used

- **Python**
- **NumPy** and **pandas** for data handling and calculations
- **SciPy** for the normal and Student-t distributions
- **yfinance** for historical stock prices
- **Streamlit** for the interactive dashboard
- **Plotly** for the charts
- **Jupyter Notebook** for the step-by-step analysis

## How It Works

1. Downloads daily closing prices and converts them into daily returns.
2. Calculates each stock's average return and the covariance matrix between stocks.
3. **Historical method:** takes the worst 5% of real past returns and scales them to the chosen time horizon.
4. **Parametric method:** fits a normal or fat-tailed Student-t distribution and calculates VaR and CVaR with closed-form formulas.
5. **Monte Carlo method:** uses a Cholesky decomposition of the covariance matrix to simulate correlated daily returns, compounds them into thousands of portfolio paths, and reads VaR and CVaR from the final values.
6. Shows all results on the dashboard so the methods can be compared.

**VaR (95%)** is the loss that should not be exceeded in 95% of cases. **CVaR** is the average loss in the worst 5% of cases.
