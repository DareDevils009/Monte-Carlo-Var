import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo VaR Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ─────────────────────────────────────────── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    color: #e0e0e0;
}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label {
    color: #b0b0d0 !important;
    font-weight: 500;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Header ─────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.12));
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
    animation: pulse-slow 8s ease-in-out infinite;
}
@keyframes pulse-slow {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    position: relative;
    z-index: 1;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    position: relative;
    z-index: 1;
}

/* ── KPI cards ──────────────────────────────────────── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99,102,241,0.15);
}
.kpi-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7c7caa;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.kpi-card .value {
    font-size: 1.65rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
}
.kpi-card .sub {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 400;
}
.kpi-green .value  { color: #34d399; }
.kpi-red .value    { color: #f87171; }
.kpi-blue .value   { color: #60a5fa; }
.kpi-purple .value { color: #a78bfa; }
.kpi-card .accent-bar {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 3px;
}
.kpi-green  .accent-bar { background: linear-gradient(90deg, #34d399, #10b981); }
.kpi-red    .accent-bar { background: linear-gradient(90deg, #f87171, #ef4444); }
.kpi-blue   .accent-bar { background: linear-gradient(90deg, #60a5fa, #3b82f6); }
.kpi-purple .accent-bar { background: linear-gradient(90deg, #a78bfa, #8b5cf6); }

/* ── Section card ───────────────────────────────────── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(12px);
}
.section-card h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Table styling ──────────────────────────────────── */
.var-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
}
.var-table thead th {
    background: rgba(99,102,241,0.15);
    color: #a78bfa;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.85rem 1.2rem;
    font-weight: 700;
    text-align: right;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.var-table thead th:first-child { text-align: left; }
.var-table tbody td {
    padding: 0.8rem 1.2rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    text-align: right;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
.var-table tbody td:first-child {
    text-align: left;
    color: #94a3b8;
    font-weight: 600;
}
.var-table tbody tr:hover {
    background: rgba(99,102,241,0.06);
}
.var-table tbody tr:last-child td { border-bottom: none; }

/* ── Button styling ─────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.25) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(99,102,241,0.4) !important;
}

/* ── Hide default streamlit elements ────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Divider ────────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
    margin: 1.5rem 0;
}

/* ── Sidebar logo area ──────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.sidebar-brand h2 {
    font-size: 1.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.sidebar-brand p {
    color: #64748b;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0.3rem 0 0 0;
    font-weight: 600;
}

/* ── Status badge ───────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    color: #34d399;
    font-weight: 600;
}
.status-badge-warn {
    background: rgba(251,191,36,0.1);
    border-color: rgba(251,191,36,0.25);
    color: #fbbf24;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_stock_data(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data[["Close"]]
        data.columns = tickers
    data = data.dropna()
    return data


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def run_monte_carlo(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    num_sims: int,
    time_horizon: int,
    investment: float,
) -> dict:
    """
    Run a Monte Carlo simulation using Cholesky decomposition
    for correlated random walks.
    """
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    num_assets = len(mean_returns)

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If not positive-definite, add small regularisation
        epsilon = 1e-8
        L = np.linalg.cholesky(cov_matrix + np.eye(num_assets) * epsilon)

    # Generate correlated random samples
    # Shape: (num_sims, time_horizon, num_assets)
    Z = np.random.standard_normal((num_sims, time_horizon, num_assets))
    correlated_Z = np.einsum("ij,ntj->nti", L, Z)

    # Simulate daily portfolio returns
    daily_portfolio_returns = np.zeros((num_sims, time_horizon))
    for t in range(time_horizon):
        asset_returns = mean_returns + correlated_Z[:, t, :]
        daily_portfolio_returns[:, t] = asset_returns @ weights

    # Cumulative returns → portfolio value paths
    cumulative_returns = np.cumsum(daily_portfolio_returns, axis=1)
    portfolio_paths = investment * np.exp(cumulative_returns)

    # Terminal portfolio values & returns
    terminal_values = portfolio_paths[:, -1]
    terminal_returns = (terminal_values - investment) / investment

    return {
        "portfolio_paths": portfolio_paths,
        "terminal_values": terminal_values,
        "terminal_returns": terminal_returns,
        "daily_portfolio_returns": daily_portfolio_returns,
    }


def calculate_var_cvar(
    terminal_returns: np.ndarray,
    investment: float,
    confidence_levels: list[float],
) -> pd.DataFrame:
    """Compute VaR and CVaR at multiple confidence levels."""
    results = []
    for cl in confidence_levels:
        alpha = 1 - cl
        var_pct = np.percentile(terminal_returns, alpha * 100)
        var_dollar = var_pct * investment
        # CVaR (Expected Shortfall)
        tail = terminal_returns[terminal_returns <= var_pct]
        cvar_pct = tail.mean() if len(tail) > 0 else var_pct
        cvar_dollar = cvar_pct * investment
        results.append(
            {
                "Confidence": f"{cl:.0%}",
                "VaR (%)": f"{var_pct:.2%}",
                "VaR ($)": f"${abs(var_dollar):,.2f}",
                "CVaR (%)": f"{cvar_pct:.2%}",
                "CVaR ($)": f"${abs(cvar_dollar):,.2f}",
            }
        )
    return pd.DataFrame(results)


def kpi_card(label: str, value: str, sub: str, color: str) -> str:
    return f"""
    <div class="kpi-card kpi-{color}">
        <div class="accent-bar"></div>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
)

COLOR_PALETTE = [
    "#818cf8", "#a78bfa", "#c084fc", "#e879f9",
    "#f472b6", "#fb7185", "#f87171", "#fbbf24",
    "#34d399", "#22d3ee",
]

VAR_COLORS = {
    "90%": "#fbbf24",
    "95%": "#f87171",
    "99%": "#ef4444",
}


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><h2>📊 VaR Engine</h2>'
        '<p>Monte Carlo Simulation</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 🏦 Portfolio Setup")
    popular_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD",
        "DIS", "NFLX", "PYPL", "ADBE", "CRM",
    ]
    tickers = st.multiselect(
        "Select Stocks",
        options=popular_tickers,
        default=["AAPL", "MSFT", "GOOGL"],
        help="Choose 1 or more stocks for your portfolio",
    )

    investment = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=100_000_000,
        value=100_000,
        step=5000,
        format="%d",
    )

    # Weights
    st.markdown("##### Portfolio Weights")
    if tickers:
        equal_w = round(1.0 / len(tickers), 4)
        weights_raw = {}
        for t in tickers:
            weights_raw[t] = st.slider(
                f"{t}",
                min_value=0.0,
                max_value=1.0,
                value=equal_w,
                step=0.01,
                key=f"w_{t}",
            )
        total_w = sum(weights_raw.values())
        # Normalise
        if total_w > 0:
            weights = np.array([weights_raw[t] / total_w for t in tickers])
        else:
            weights = np.array([equal_w] * len(tickers))

        if abs(total_w - 1.0) > 0.01:
            st.caption(f"⚠️ Weights sum to {total_w:.2f} — auto-normalised to 1.0")
        else:
            st.caption(f"✅ Weights sum to {total_w:.2f}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Simulation Parameters")

    num_sims = st.select_slider(
        "Number of Simulations",
        options=[1000, 2500, 5000, 10000, 25000, 50000, 100000],
        value=10000,
    )

    time_horizon = st.slider(
        "Time Horizon (Trading Days)",
        min_value=1,
        max_value=252,
        value=21,
        help="21 ≈ 1 month, 63 ≈ 1 quarter, 252 ≈ 1 year",
    )

    lookback = st.selectbox(
        "Historical Lookback",
        options=["6mo", "1y", "2y", "5y"],
        index=1,
    )

    confidence_levels = st.multiselect(
        "Confidence Levels",
        options=[0.90, 0.95, 0.975, 0.99],
        default=[0.90, 0.95, 0.99],
        format_func=lambda x: f"{x:.1%}",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    run_sim = st.button("🚀  Run Simulation", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>Monte Carlo Value-at-Risk Engine</h1>"
    "<p>Simulate portfolio risk with correlated random walks · Live market data · Interactive analytics</p>"
    "</div>",
    unsafe_allow_html=True,
)

if not tickers:
    st.info("👈 Select at least one stock ticker from the sidebar to get started.")
    st.stop()

if not confidence_levels:
    st.warning("Please select at least one confidence level.")
    st.stop()

# ── Run simulation ──────────────────────────────────────────────────
if run_sim:
    with st.spinner("Fetching market data..."):
        try:
            prices = fetch_stock_data(tickers, period=lookback)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

        if prices.empty or len(prices) < 30:
            st.error("Not enough historical data. Try different tickers or a longer lookback.")
            st.stop()

        log_returns = calculate_log_returns(prices)

    with st.spinner(f"Running {num_sims:,} simulations over {time_horizon} days..."):
        results = run_monte_carlo(log_returns, weights, num_sims, time_horizon, investment)

    # Store results in session state
    st.session_state["results"] = results
    st.session_state["prices"] = prices
    st.session_state["log_returns"] = log_returns
    st.session_state["params"] = {
        "tickers": tickers,
        "weights": weights,
        "investment": investment,
        "num_sims": num_sims,
        "time_horizon": time_horizon,
        "confidence_levels": confidence_levels,
    }

# ── Display results ─────────────────────────────────────────────────
if "results" not in st.session_state:
    # Show placeholder
    st.markdown(
        '<div class="section-card">'
        "<h3>🎯 Ready to Simulate</h3>"
        "<p style='color:#64748b'>Configure your portfolio in the sidebar and hit "
        "<strong>Run Simulation</strong> to generate risk analytics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Show a quick preview of selected tickers
    with st.spinner("Loading price preview..."):
        try:
            preview = fetch_stock_data(tickers, period="3mo")
            norm_preview = preview / preview.iloc[0] * 100

            fig = go.Figure()
            for i, col in enumerate(norm_preview.columns):
                fig.add_trace(
                    go.Scatter(
                        x=norm_preview.index,
                        y=norm_preview[col],
                        name=col,
                        line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                    )
                )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Price Performance (Normalised, 3M)", font=dict(size=14, color="#c4b5fd")),
                yaxis_title="Indexed (100)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.stop()

# Unpack
res = st.session_state["results"]
params = st.session_state["params"]
terminal_returns = res["terminal_returns"]
terminal_values = res["terminal_values"]
portfolio_paths = res["portfolio_paths"]
inv = params["investment"]
conf_levels = params["confidence_levels"]

# Compute VaR table
var_table = calculate_var_cvar(terminal_returns, inv, conf_levels)

# Key metrics for KPI cards
mean_terminal = terminal_values.mean()
var_95_pct = np.percentile(terminal_returns, 5)
var_95_dollar = var_95_pct * inv
tail_95 = terminal_returns[terminal_returns <= var_95_pct]
cvar_95_pct = tail_95.mean() if len(tail_95) > 0 else var_95_pct
worst_case = terminal_values.min()
best_case = terminal_values.max()

# ── KPI Cards ───────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        kpi_card(
            "Expected Portfolio Value",
            f"${mean_terminal:,.0f}",
            f"Mean of {params['num_sims']:,} simulations",
            "green",
        ),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        kpi_card(
            "Value at Risk (95%)",
            f"${abs(var_95_dollar):,.0f}",
            f"{var_95_pct:.2%} potential loss",
            "red",
        ),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card(
            "Conditional VaR (95%)",
            f"${abs(cvar_95_pct * inv):,.0f}",
            f"{cvar_95_pct:.2%} expected shortfall",
            "blue",
        ),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        kpi_card(
            "Worst-Case Scenario",
            f"${worst_case:,.0f}",
            f"Best: ${best_case:,.0f}",
            "purple",
        ),
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Simulated Portfolio Paths ───────────────────────────────────────
st.markdown(
    '<div class="section-card"><h3>📈 Simulated Portfolio Paths</h3></div>',
    unsafe_allow_html=True,
)

# Sample paths to keep the chart responsive
max_display = min(500, portfolio_paths.shape[0])
sample_idx = np.random.choice(portfolio_paths.shape[0], max_display, replace=False)
days = np.arange(1, portfolio_paths.shape[1] + 1)

fig_paths = go.Figure()

# Individual paths (faint)
for idx in sample_idx:
    fig_paths.add_trace(
        go.Scatter(
            x=days,
            y=portfolio_paths[idx],
            mode="lines",
            line=dict(color="rgba(129,140,248,0.06)", width=0.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

# Percentile bands
p5 = np.percentile(portfolio_paths, 5, axis=0)
p25 = np.percentile(portfolio_paths, 25, axis=0)
p50 = np.median(portfolio_paths, axis=0)
p75 = np.percentile(portfolio_paths, 75, axis=0)
p95 = np.percentile(portfolio_paths, 95, axis=0)
mean_path = portfolio_paths.mean(axis=0)

# 5-95 band
fig_paths.add_trace(
    go.Scatter(x=days, y=p95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
)
fig_paths.add_trace(
    go.Scatter(
        x=days, y=p5, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99,102,241,0.08)",
        name="5th–95th Percentile", hoverinfo="skip",
    )
)
# 25-75 band
fig_paths.add_trace(
    go.Scatter(x=days, y=p75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
)
fig_paths.add_trace(
    go.Scatter(
        x=days, y=p25, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99,102,241,0.15)",
        name="25th–75th Percentile", hoverinfo="skip",
    )
)

# Median & Mean
fig_paths.add_trace(
    go.Scatter(
        x=days, y=p50, mode="lines",
        line=dict(color="#a78bfa", width=2.5),
        name="Median Path",
    )
)
fig_paths.add_trace(
    go.Scatter(
        x=days, y=mean_path, mode="lines",
        line=dict(color="#34d399", width=2, dash="dot"),
        name="Mean Path",
    )
)

# Investment line
fig_paths.add_hline(
    y=inv, line_dash="dash", line_color="rgba(251,191,36,0.5)",
    annotation_text=f"Initial: ${inv:,.0f}",
    annotation_position="top right",
    annotation_font_color="#fbbf24",
)

fig_paths.update_layout(
    **PLOTLY_LAYOUT,
    height=480,
    yaxis_title="Portfolio Value ($)",
    xaxis_title="Trading Days",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font=dict(size=11),
    ),
)
st.plotly_chart(fig_paths, use_container_width=True)

# ── Return Distribution ────────────────────────────────────────────
col_hist, col_table = st.columns([3, 2])

with col_hist:
    st.markdown(
        '<div class="section-card"><h3>📊 Terminal Return Distribution</h3></div>',
        unsafe_allow_html=True,
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=terminal_returns * 100,
            nbinsx=80,
            marker=dict(
                color="rgba(129,140,248,0.6)",
                line=dict(color="rgba(129,140,248,0.8)", width=0.5),
            ),
            name="Return Distribution",
            hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        )
    )

    # VaR lines
    for cl in sorted(conf_levels):
        alpha = 1 - cl
        var_val = np.percentile(terminal_returns, alpha * 100) * 100
        cl_label = f"{cl:.0%}"
        color = VAR_COLORS.get(cl_label, "#fbbf24")
        fig_hist.add_vline(
            x=var_val, line_dash="dash", line_color=color, line_width=2,
            annotation_text=f"VaR {cl_label}: {var_val:.1f}%",
            annotation_position="top left",
            annotation_font=dict(size=10, color=color),
        )

    # Mean line
    mean_ret = terminal_returns.mean() * 100
    fig_hist.add_vline(
        x=mean_ret, line_dash="dot", line_color="#34d399", line_width=2,
        annotation_text=f"Mean: {mean_ret:.1f}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#34d399"),
    )

    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_table:
    st.markdown(
        '<div class="section-card"><h3>📋 VaR & CVaR Summary</h3></div>',
        unsafe_allow_html=True,
    )

    # Build HTML table
    table_html = '<table class="var-table"><thead><tr>'
    for col in var_table.columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"
    for _, row in var_table.iterrows():
        table_html += "<tr>"
        for val in row:
            table_html += f"<td>{val}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # Additional stats
    st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-card"><h3>📐 Distribution Statistics</h3></div>',
        unsafe_allow_html=True,
    )

    stats_data = {
        "Metric": [
            "Mean Return", "Median Return", "Std Deviation",
            "Skewness", "Kurtosis", "Min Return", "Max Return",
        ],
        "Value": [
            f"{terminal_returns.mean():.4%}",
            f"{np.median(terminal_returns):.4%}",
            f"{terminal_returns.std():.4%}",
            f"{pd.Series(terminal_returns).skew():.4f}",
            f"{pd.Series(terminal_returns).kurtosis():.4f}",
            f"{terminal_returns.min():.4%}",
            f"{terminal_returns.max():.4%}",
        ],
    }
    stats_df = pd.DataFrame(stats_data)
    stats_html = '<table class="var-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
    for _, row in stats_df.iterrows():
        stats_html += f'<tr><td>{row["Metric"]}</td><td>{row["Value"]}</td></tr>'
    stats_html += "</tbody></table>"
    st.markdown(stats_html, unsafe_allow_html=True)

# ── Portfolio Weights Breakdown ─────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_pie, col_corr = st.columns(2)

with col_pie:
    st.markdown(
        '<div class="section-card"><h3>🧩 Portfolio Allocation</h3></div>',
        unsafe_allow_html=True,
    )
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=params["tickers"],
                values=params["weights"],
                hole=0.55,
                marker=dict(colors=COLOR_PALETTE[: len(params["tickers"])]),
                textinfo="label+percent",
                textfont=dict(size=12, color="white"),
                hovertemplate="%{label}: %{percent}<extra></extra>",
            )
        ]
    )
    fig_pie.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        showlegend=False,
        annotations=[
            dict(
                text=f"${inv / 1000:.0f}K",
                x=0.5, y=0.5, font_size=18, font_color="#a78bfa",
                showarrow=False, font=dict(family="Inter", weight=700),
            )
        ],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_corr:
    st.markdown(
        '<div class="section-card"><h3>🔗 Return Correlation Matrix</h3></div>',
        unsafe_allow_html=True,
    )
    log_rets = st.session_state["log_returns"]
    corr = log_rets.corr()

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0, "#312e81"],
                [0.5, "#1e1b4b"],
                [1, "#818cf8"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>ρ = %{z:.3f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(color="#94a3b8"),
                title=dict(text="ρ", font=dict(color="#a78bfa")),
            ),
        )
    )
    fig_corr.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(side="bottom", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.75rem;padding:1rem 0">'
    "Monte Carlo VaR Engine · Built with Streamlit & Plotly · "
    f"Simulation ran at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</p>",
    unsafe_allow_html=True,
)
