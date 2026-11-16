"""
Portfolio Risk Dashboard
========================

An interactive front-end for the Monte Carlo VaR / CVaR project.

Run it with:
    streamlit run app.py

Everything you can change lives in the left sidebar; the main page updates
automatically. All the maths is in ``mc_engine.py`` - this file only handles
layout, charts and the plain-English summary.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import mc_engine as mc

# ---------------------------------------------------------------------------
# Page set-up and colour palette
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Portfolio Risk Dashboard", page_icon="📈", layout="wide")

# Colour-blind-checked palette (fixed order - a colour always means the same thing)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
LOSS = "#c0392b"   # reserved for "loss" markers only

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      div[data-testid="stMetric"] {
          background: #ffffff; border: 1px solid #e7e6e2; border-radius: 12px;
          padding: 14px 16px 10px 16px;}
      div[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #52514e;}
      .insight {background:#f4f8fd; border-left:4px solid #2a78d6; border-radius:8px;
                padding:14px 18px; margin-bottom:8px; line-height:1.55;}
      .insight b {color:#0b0b0b;}
      .subtle {color:#52514e; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 420, y_title: str = "", x_title: str = "") -> go.Figure:
    """Apply one consistent, quiet look to every chart."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK_2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=INK),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def money(x: float) -> str:
    return f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar - every input the user can control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("1 · Portfolio")
    tickers_text = st.text_input(
        "Tickers (comma separated)", "CBA, BHP, TLS, NAB, WBC, STO",
        help="Use Yahoo Finance symbols. Add a market suffix below, or type full symbols like AAPL.")
    suffix = st.text_input("Market suffix", ".AX", help="'.AX' = Australian exchange. Leave empty for US stocks.")
    tickers = [t.strip().upper() + suffix.strip() for t in tickers_text.split(",") if t.strip()]

    lookback = st.slider("History to learn from (calendar days)", 200, 2000, 800, step=50)

    weight_mode = st.radio("Weights", ["Equal", "Random", "Custom"], horizontal=True)

    st.subheader("2 · Simulation")
    initial = st.number_input("Initial investment ($)", 1_000, 10_000_000, 10_000, step=1_000)
    horizon = st.slider("Time horizon (trading days)", 10, 252, 100, step=5)
    n_sims = st.select_slider("Number of simulations", [100, 250, 400, 1000, 2500, 5000, 10000], value=1000)
    conf = st.select_slider("Confidence level", [90, 95, 97.5, 99], value=95)
    alpha = 100 - conf

    with st.expander("Advanced"):
        dof = st.slider("Student-t degrees of freedom", 3, 30, 6,
                        help="Lower = fatter tails = more weight on extreme days.")
        seed = st.number_input("Random seed", 0, 99_999, 42,
                               help="Same seed → same simulation, so results are reproducible.")
        force_demo = st.toggle("Use demo data (offline)", value=False)

    st.subheader("3 · AI summary")
    try:
        _default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:          # no secrets.toml file - that's fine
        _default_key = ""
    _default_key = _default_key or os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input("Anthropic API key", value=_default_key, type="password",
                            help="Needed for the AI explanation. Get one at console.anthropic.com. "
                                 "It's only used for this session and never saved.")

# ---------------------------------------------------------------------------
# Data loading (cached so changing a slider does not re-download)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Downloading price history…")
def get_prices(tickers: tuple[str, ...], lookback: int, force_demo: bool) -> tuple[pd.DataFrame, bool]:
    """Return (prices, is_demo). Falls back to synthetic data if the download fails."""
    if not force_demo:
        end = dt.datetime.now()
        try:
            px = mc.load_prices(list(tickers), end - dt.timedelta(days=lookback), end)
            if not px.empty and px.shape[0] > 30:
                return px, False
        except Exception:
            pass
    return mc.demo_prices(list(tickers), days=int(lookback * 252 / 365)), True


if len(tickers) < 1:
    st.warning("Add at least one ticker in the sidebar to get started.")
    st.stop()

prices, is_demo = get_prices(tuple(tickers), lookback, force_demo)
returns, mean_returns, cov_matrix = mc.compute_returns(prices)
assets = list(returns.columns)
missing = sorted(set(tickers) - set(assets))

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

if weight_mode == "Equal":
    weights = np.full(len(assets), 1 / len(assets))
elif weight_mode == "Random":
    weights = np.random.default_rng(int(seed)).random(len(assets))
    weights /= weights.sum()
else:
    with st.sidebar:
        st.caption("Custom weights (they are rescaled to add up to 100%)")
        raw = [st.number_input(a, 0.0, 100.0, round(100 / len(assets), 1), step=1.0, key=f"w_{a}")
               for a in assets]
    raw = np.array(raw, dtype=float)
    weights = raw / raw.sum() if raw.sum() > 0 else np.full(len(assets), 1 / len(assets))

# ---------------------------------------------------------------------------
# Run the model
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Running Monte Carlo simulation…")
def run_sims(mean_r, cov, w, horizon, n_sims, initial, seed):
    return mc.simulate_portfolio(pd.Series(mean_r), pd.DataFrame(cov), np.array(w),
                                 horizon, n_sims, initial, seed)


paths = run_sims(mean_returns.values, cov_matrix.values, tuple(weights), horizon, n_sims, initial, int(seed))
final = paths[:, -1]
exp_ret, vol = mc.portfolio_performance(weights, mean_returns, cov_matrix, horizon)
summary = mc.risk_summary(returns, mean_returns, cov_matrix, weights, horizon, initial, alpha, dof, paths)

mc_row = summary.set_index("Method").loc["Monte Carlo"]
var_usd, cvar_usd = mc_row["VaR $"], mc_row["CVaR $"]
p_loss = float((final < initial).mean())
median_end = float(np.median(final))

# ---------------------------------------------------------------------------
# Header + headline numbers
# ---------------------------------------------------------------------------

st.title("📈 Portfolio Risk Dashboard")
st.markdown(
    f"<span class='subtle'>Monte Carlo simulation of a <b>{len(assets)}-stock</b> portfolio · "
    f"<b>{money(initial)}</b> invested · <b>{horizon}</b> trading days · "
    f"<b>{n_sims:,}</b> simulated futures · <b>{conf}%</b> confidence</span>",
    unsafe_allow_html=True,
)

if is_demo:
    st.info("Showing **demo data** (synthetic prices) because live prices couldn't be downloaded "
            "or demo mode is on. Results illustrate the method, not the real stocks.", icon="ℹ️")
if missing:
    st.warning(f"No data found for: {', '.join(missing)} — they were left out.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Expected end value", money(initial * (1 + exp_ret)), f"{exp_ret:+.1%}")
k2.metric("Median outcome", money(median_end), f"{median_end / initial - 1:+.1%}")
k3.metric(f"VaR ({conf}%)", money(var_usd), f"-{var_usd / initial:.1%}",
          help=f"Value at Risk: in {conf}% of simulations you lose less than this.")
k4.metric(f"CVaR ({conf}%)", money(cvar_usd), f"-{cvar_usd / initial:.1%}",
          help=f"Conditional VaR / Expected Shortfall: the average loss in the worst {alpha:g}% of simulations.")
k5.metric("Chance of a loss", f"{p_loss:.0%}", help="Share of simulations that end below what you put in.")

# ---------------------------------------------------------------------------
# Plain-English takeaways (generated from the numbers)
# ---------------------------------------------------------------------------


def takeaways() -> list[str]:
    out = [
        f"With <b>{conf}% confidence</b>, this portfolio should not lose more than "
        f"<b>{money(var_usd)}</b> ({var_usd / initial:.1%}) over the next {horizon} trading days.",
        f"If things do go badly (the worst {alpha:g}% of cases), the <b>average loss is "
        f"{money(cvar_usd)}</b>. That is the number to plan around for a bad scenario.",
        f"About <b>{p_loss:.0%}</b> of simulated futures end below your starting {money(initial)}; "
        f"the middle half of outcomes lands between <b>{money(np.percentile(final, 25))}</b> and "
        f"<b>{money(np.percentile(final, 75))}</b>.",
    ]
    s = summary.set_index("Method")["VaR $"]
    hi, lo = s.idxmax(), s.idxmin()
    if s[hi] > 1.25 * s[lo]:
        extra = (" Real returns had fatter tails than a bell curve assumes, so the model-based "
                 "figures may understate risk." if hi == "Historical" else "")
        out.append(f"The methods disagree: <b>{hi}</b> gives the largest VaR ({money(s[hi])}) and "
                   f"<b>{lo}</b> the smallest ({money(s[lo])}).{extra}")
    else:
        out.append("All four methods give similar VaR figures, which makes the estimate more trustworthy.")
    top = int(np.argmax(weights))
    if len(assets) > 1:
        corr = returns.corr().values.copy()
        np.fill_diagonal(corr, np.nan)
        i, j = np.unravel_index(np.nanargmax(corr), corr.shape)
        lead = ("Money is split equally across all stocks. " if np.allclose(weights, weights[0])
                else f"Largest holding: <b>{assets[top]}</b> at {weights[top]:.0%}. ")
        out.append(lead + "Most closely linked pair: "
                   f"<b>{assets[i]}</b> & <b>{assets[j]}</b> (correlation {corr[i, j]:.2f}), "
                   f"so they tend to fall together and add less diversification.")
    return out


with st.container():
    st.markdown("#### 💡 Key takeaways")
    st.markdown("<div class='insight'>" + "<br>".join("• " + t for t in takeaways()) + "</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AI explanation - optional, needs an Anthropic API key
# ---------------------------------------------------------------------------

# Preferred model family; the newest available model of this family is picked automatically.
# Set the AI_MODEL environment variable to force a specific model id instead.
AI_MODEL_FAMILY = "sonnet"


def build_ai_prompt() -> str:
    """Pack every result the dashboard shows into a compact brief for the model."""
    corr = returns.corr().round(2)
    holdings = ", ".join(f"{a} {w:.1%}" for a, w in zip(assets, weights))
    methods = "\n".join(f"- {r.Method}: VaR {money(r['VaR $'])} ({r['VaR %']:.2%}), "
                        f"CVaR {money(r['CVaR $'])} ({r['CVaR %']:.2%})" for _, r in summary.iterrows())
    return f"""You are explaining the output of a portfolio risk dashboard to a student who is new to finance.

Setup: {money(initial)} invested for {horizon} trading days, {n_sims:,} Monte Carlo simulations, {conf}% confidence.
Data: {"SYNTHETIC demo data, not real prices" if is_demo else f"real daily prices, {len(returns)} trading days"}.
Holdings: {holdings}

Results
- Expected end value: {money(initial * (1 + exp_ret))} ({exp_ret:+.1%}); horizon volatility {vol:.1%}
- Median simulated end value: {money(median_end)}
- 5th / 25th / 75th / 95th percentile end values: {money(np.percentile(final, 5))} / {money(np.percentile(final, 25))} / {money(np.percentile(final, 75))} / {money(np.percentile(final, 95))}
- Probability of ending below the starting value: {p_loss:.1%}
VaR and CVaR by method:
{methods}
Correlation matrix of daily returns:
{corr.to_string()}

Write a short explanation in Markdown:
1. A one-sentence headline verdict on how risky this portfolio is over the horizon.
2. 4-6 bullets in plain English: what VaR and CVaR mean here in dollars, why the methods agree or disagree, what the correlations say about diversification, and anything that stands out.
3. One bullet suggesting a setting to try next in the dashboard to learn more (e.g. change weights, horizon, confidence level).
Keep it under 220 words. Use the actual numbers. Do not give personal investment advice or tell the reader to buy or sell anything."""


def pick_model(client) -> str:
    """Use AI_MODEL if set, otherwise the newest available model of AI_MODEL_FAMILY."""
    forced = os.environ.get("AI_MODEL")
    if forced:
        return forced
    ids = [m.id for m in client.models.list(limit=100)]          # newest first
    for family in (AI_MODEL_FAMILY, "opus", "haiku"):
        match = next((i for i in ids if family in i), None)
        if match:
            return match
    return ids[0]


def ask_ai(prompt: str, key: str) -> str:
    import anthropic  # imported lazily so the app still runs without the package

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(model=pick_model(client), max_tokens=700,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


ai_col1, ai_col2 = st.columns([1, 4])
with ai_col1:
    ai_clicked = st.button("✨ Explain with AI", type="primary", width="stretch",
                           disabled=not api_key, help=None if api_key else "Add an API key in the sidebar first")
if not api_key:
    ai_col2.caption("Add your Anthropic API key in the sidebar (section 3) to get an AI-written explanation of these results.")

ai_prompt = build_ai_prompt()
if ai_clicked:
    with st.spinner("The AI is reading the results…"):
        try:
            st.session_state["ai_text"] = ask_ai(ai_prompt, api_key)
            st.session_state["ai_for"] = ai_prompt
        except Exception as err:
            st.session_state.pop("ai_text", None)
            st.error(f"Couldn't get an AI explanation: {err}")

if st.session_state.get("ai_text"):
    with st.container(border=True):
        st.markdown("##### ✨ AI explanation")
        if st.session_state.get("ai_for") != ai_prompt:
            st.caption("⚠️ Settings changed since this was written. Click **Explain with AI** again to refresh it.")
        st.markdown(st.session_state["ai_text"])
        st.caption("Written by AI. It can make mistakes, so check it against the numbers above.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_sim, tab_dist, tab_cmp, tab_port, tab_how = st.tabs(
    ["🔮 Simulated paths", "📊 Outcome distribution", "⚖️ Compare methods", "🧺 Portfolio & data", "📘 How it works"])

days = np.arange(horizon + 1)

# --- Tab 1: every simulated path, classic Monte Carlo look -----------------
# matplotlib's default colour cycle, so it looks like the notebook's plt.plot(portfolio_sims)
MPL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MAX_LINES = 500    # drawing more lines than this slows the browser without changing the picture

with tab_sim:
    show = min(n_sims, MAX_LINES)
    note = f" (showing {show:,} of {n_sims:,})" if show < n_sims else ""
    st.markdown(f"Each coloured line is one possible future for the portfolio{note}.")
    show_median = st.toggle("Highlight the median path", value=False)

    fig = go.Figure()
    # One line per simulation, drawn in order and cycling colours - exactly like plt.plot(portfolio_sims)
    for i in range(show):
        fig.add_trace(go.Scatter(
            x=days, y=paths[i], mode="lines", showlegend=False,
            line=dict(width=1.3, color=MPL_COLORS[i % len(MPL_COLORS)]),
            hovertemplate=f"Simulation #{i + 1}<br>Day %{{x}}<br>Value %{{y:$,.0f}}<extra></extra>"))
    if show_median:
        fig.add_trace(go.Scatter(x=days, y=np.median(paths, axis=0), mode="lines", name="Median path",
                                 line=dict(width=3.5, color="#000000"),
                                 hovertemplate="Median<br>Day %{x}<br>Value %{y:$,.0f}<extra></extra>"))

    fig.update_layout(
        title=dict(text="MC simulation of a stock portfolio", x=0.5, xanchor="center",
                   font=dict(size=16, color=INK)),
        height=520, margin=dict(l=10, r=20, t=50, b=10), hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family="DejaVu Sans, Inter, sans-serif", size=13, color=INK), showlegend=show_median,
    )
    frame = dict(showline=True, linecolor="#000000", linewidth=1, mirror=True, ticks="outside",
                 showgrid=False, zeroline=False)
    fig.update_xaxes(title="Days", **frame)
    fig.update_yaxes(title="Portfolio Value ($)", tickformat=",.0f", **frame)
    st.plotly_chart(fig, width="stretch")

# --- Tab 2: histogram of final values --------------------------------------
with tab_dist:
    st.markdown(f"Where the portfolio ends up after {horizon} days across all {n_sims:,} simulations. "
                f"The red region is the worst {alpha:g}% — the tail that VaR and CVaR describe.")
    var_level, cvar_level = initial - var_usd, initial - cvar_usd
    bins = np.linspace(final.min(), final.max(), 45 if n_sims >= 2500 else 30)
    counts, edges = np.histogram(final, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    tail = mids <= var_level
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids[~tail], y=counts[~tail], marker_color=BLUE, name="Simulated outcomes",
                         hovertemplate="Ending value ≈ %{x:$,.0f}<br>%{y} simulations<extra></extra>"))
    fig.add_trace(go.Bar(x=mids[tail], y=counts[tail], marker_color=LOSS, name=f"Worst {alpha:g}% (tail)",
                         hovertemplate="Ending value ≈ %{x:$,.0f}<br>%{y} simulations<extra></extra>"))
    # Reference lines, with labels stacked at the top so they never overlap the bars
    for x, label, dash, yy in [(var_level, f"VaR cut-off  {money(var_level)}", "dash", 1.0),
                               (cvar_level, f"Tail average  {money(cvar_level)}", "dot", 0.91)]:
        fig.add_vline(x=x, line=dict(color=INK, width=1.5, dash=dash))
        fig.add_annotation(x=x, y=yy, yref="paper", text=label, showarrow=False, xanchor="right",
                           xshift=-4, font=dict(color=INK, size=12), bgcolor="rgba(252,252,251,0.9)")
    fig.add_vline(x=initial, line=dict(color=INK_2, width=1))
    fig.add_annotation(x=initial, y=1.0, yref="paper", text="Start", showarrow=False, xanchor="left",
                       xshift=4, font=dict(color=INK_2, size=12))
    fig.update_xaxes(tickprefix="$")
    fig.update_layout(bargap=0.08)
    st.plotly_chart(style_fig(fig, 460, "Number of simulations", "Portfolio value at end of horizon ($)"),
                    width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best case (95th pct)", money(np.percentile(final, 95)))
    c2.metric("Median", money(median_end))
    c3.metric("Worst case (5th pct)", money(np.percentile(final, 5)))
    c4.metric("Absolute worst simulated", money(final.min()))

# --- Tab 3: compare the four methods ---------------------------------------
with tab_cmp:
    st.markdown("The same risk measured four different ways. Bigger bars = more estimated loss. "
                "When the bars disagree a lot, the assumptions behind each method matter.")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=summary["Method"], x=summary["VaR $"], orientation="h", name="VaR",
                         marker_color=BLUE, text=[money(v) for v in summary["VaR $"]], textposition="outside",
                         hovertemplate="%{y}<br>VaR: %{x:$,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(y=summary["Method"], x=summary["CVaR $"], orientation="h", name="CVaR (Expected Shortfall)",
                         marker_color=ORANGE, text=[money(v) for v in summary["CVaR $"]], textposition="outside",
                         hovertemplate="%{y}<br>CVaR: %{x:$,.0f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, summary[["VaR $", "CVaR $"]].values.max() * 1.18], tickprefix="$", tickformat=",")
    st.plotly_chart(style_fig(fig, 400, "", f"Estimated loss over {horizon} days ($)"), width="stretch")

    table = summary.copy()
    table["VaR %"] = table["VaR %"].map(lambda v: f"{v:.2%}")
    table["CVaR %"] = table["CVaR %"].map(lambda v: f"{v:.2%}")
    table["VaR $"] = table["VaR $"].map(money)
    table["CVaR $"] = table["CVaR $"].map(money)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("⬇️ Download results (CSV)", summary.to_csv(index=False), "var_cvar_results.csv", "text/csv")

# --- Tab 4: portfolio composition and historical data ----------------------
with tab_port:
    left, right = st.columns(2)
    with left:
        st.markdown("**Portfolio weights**")
        order = np.argsort(weights)
        fig = go.Figure(go.Bar(x=weights[order], y=[assets[i] for i in order], orientation="h",
                               marker_color=BLUE, text=[f"{weights[i]:.1%}" for i in order],
                               textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", range=[0, weights.max() * 1.25])
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with right:
        st.markdown("**How the stocks move together (correlation)**")
        corr = returns.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=assets, y=assets, zmin=-1, zmax=1,
            colorscale=[[0, "#c0392b"], [0.5, "#f1f0ee"], [1, "#184f95"]],
            text=np.round(corr.values, 2), texttemplate="%{text}", xgap=2, ygap=2,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    st.markdown("**Historical growth of $1** — the portfolio (thick line) against each stock")
    growth = (1 + returns).cumprod()
    fig = go.Figure()
    for k, a in enumerate(assets):
        fig.add_trace(go.Scatter(x=growth.index, y=growth[a], name=a, mode="lines",
                                 line=dict(width=1.5, color=SERIES[k % len(SERIES)]), opacity=0.75))
    fig.add_trace(go.Scatter(x=growth.index, y=(1 + returns @ weights).cumprod(), name="Portfolio",
                             mode="lines", line=dict(width=3.5, color=INK)))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(style_fig(fig, 420, "Value of $1 invested"), width="stretch")

    stats = pd.DataFrame({
        "Weight": weights,
        "Avg daily return": mean_returns.values,
        "Daily volatility": np.sqrt(np.diag(cov_matrix.values)),
        "Annualised return": mean_returns.values * 252,
        "Annualised volatility": np.sqrt(np.diag(cov_matrix.values) * 252),
    }, index=assets)
    st.dataframe(stats.style.format("{:.2%}"), width="stretch")

# --- Tab 5: methodology -----------------------------------------------------
with tab_how:
    st.markdown(f"""
### What the numbers mean
- **Value at Risk (VaR)** — a threshold. "With {conf}% confidence, we won't lose more than this."
- **Conditional VaR / Expected Shortfall (CVaR)** — the *average* loss in the worst {alpha:g}% of cases.
  It is always at least as large as VaR and says how bad the bad days really are.

### The four methods
| Method | Idea | Strength | Weakness |
|---|---|---|---|
| **Historical** | Uses the actual past daily returns, then scales to {horizon} days with √time | No shape assumed; captures real crashes | Only knows what already happened |
| **Parametric – Normal** | Fits a bell curve using mean & volatility | Fast, simple formula | Underestimates extreme moves |
| **Parametric – Student-t** | Same, but with fatter tails (ν = {dof}) | More realistic tails | Choice of ν is a judgement call |
| **Monte Carlo** | Simulates {n_sims:,} correlated futures with a Cholesky decomposition of the covariance matrix | Flexible; shows the full range of outcomes | Only as good as its inputs; slower |

### Monte Carlo step by step
1. Estimate each stock's average daily return and the covariance matrix from **{len(returns)}** trading days of history.
2. Factor the covariance matrix as **Σ = L·Lᵀ** (Cholesky) so simulated stocks move together realistically.
3. For each simulation draw random shocks **Z**, build daily returns **μ + L·Z**, weight them, and compound for {horizon} days.
4. Read VaR and CVaR from the distribution of ending values.

*Past performance is not a guarantee of future results. This tool is for learning and exploration, not financial advice.*
""")
