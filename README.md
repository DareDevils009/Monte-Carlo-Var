# 📊 Monte Carlo VaR Engine

This is a sophisticated Streamlit-based application designed to simulate portfolio risk using Monte Carlo methods. It empowers investors and financial analysts to estimate **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** for customizable stock portfolios, providing crucial insights into potential downside risk over a specific time horizon.

## 🚀 Features

*   **Portfolio Customization:**
    *   Select from a curated list of popular tickers (e.g., AAPL, MSFT, TSLA, JPM).
    *   Dynamically adjust portfolio weights using interactive sliders.
    *   Automatic weight normalization to ensure 100% allocation.
*   **Advanced Simulation Engine:**
    *   Utilizes **Cholesky decomposition** to generate correlated random walks based on historical covariance.
    *   Run up to **100,000 simulations** to project potential future portfolio values.
*   **Comprehensive Risk Metrics:**
    *   Calculate **VaR** (Value at Risk) at multiple confidence levels (90%, 95%, 99%).
    *   Calculate **CVaR** (Expected Shortfall) to understand tail risk severity.
*   **Interactive Visualizations (Plotly):**
    *   **Simulated Portfolio Paths:** Visualize thousands of potential future price paths with confidence intervals (5th-95th and 25th-75th percentiles).
    *   **Terminal Return Distribution:** Analyze the probability distribution of final returns with overlaid VaR thresholds.
*   **Real-Time Data Integration:**
    *   Fetches adjusted close prices directly from Yahoo Finance via `yfinance`.
    *   Configurable historical lookback periods (6mo, 1y, 2y, 5y).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/monte-carlo-var-engine.git
    cd monte-carlo-var-engine
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+ installed. Then runs:
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Dependencies

The application relies on the following key libraries:

*   `streamlit`: For the interactive web interface.
*   `yfinance`: For fetching historical market data.
*   `numpy` & `pandas`: For high-performance numerical computations and data manipulation.
*   `plotly`: For creating interactive and responsive financial charts.
*   `scipy`: For statistical functions.

## 🖥️ Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open in Browser:**
    The application will automatically open in your default web browser (usually at `http://localhost:8501`).

3.  **Configure & Analyze:**
    *   Use the **Sidebar** to select stocks, set investment amount, adjust weights, and define simulation parameters.
    *   Click **🚀 Run Simulation** to generate the analysis.
    *   Explore the KPI cards, portfolio path charts, and return distribution histograms.

## 📈 Methodology

The engine uses Monte Carlo simulation with Geometric Brownian Motion (GBM) adapted for multiple correlated assets:

1.  **Data Ingestion:** Downloads historical adjusted close prices.
2.  **Returns Analysis:** Calculates daily log returns and the covariance matrix.
3.  **Cholesky Decomposition:** Decomposes the covariance matrix to generate correlated random shocks.
4.  **Simulation:** Projects daily returns for each asset over the specified time horizon (default: 21 trading days).
5.  **Aggregation:** Combines individual asset simulations into a portfolio view based on user-defined weights.
6.  **Risk Calculation:** Derives VaR and CVaR from the distribution of terminal portfolio values.

---

*Note: This tool is for educational and informational purposes only and does not constitute financial advice.*
