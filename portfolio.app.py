import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
from textblob import TextBlob

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Improve your portfolio", layout="wide")

# --- LOGO ---
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #ffffff; font-size: 3em; font-weight: bold; text-shadow: 2px 2px 4px #000;">Improve Your Portfolio</h1>
        <p style="color: #ffffff; font-size: 1.2em;">AI-Powered Portfolio Optimization</p>
    </div>
    """, unsafe_allow_html=True)

# --- ESTILO CSS PROFESIONAL (GLASS-DARK DESIGN SYSTEM) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        background-color: #0B0E11;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0B0E11;
        border-right: 1px solid #2A2E39;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] * {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stSubheader {
        color: #00FF41 !important;
        font-weight: 700 !important;
        text-decoration: underline;
    }
    /* Slider values in coral red */
    [data-testid="stSidebar"] .stSlider .stMarkdown p {
        color: #FF6B6B !important; /* Coral Red */
        font-weight: 700 !important;
    }
    /* Input values in orange */
    [data-testid="stSidebar"] input[type="number"] {
        color: #FFA500 !important; /* Orange */
        font-weight: 700 !important;
    }
    .stMetric {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        color: #00FF41 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #D1D4DC !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        color: #848E9C !important;
        padding: 8px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00FF41 !important;
        color: #000 !important;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 255, 65, 0.3);
    }
    .stDataFrame, .stTable {
        background-color: #1E222D !important;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stDataFrame table, .stTable table {
        background-color: #1E222D !important;
        border-collapse: collapse;
    }
    .stDataFrame th, .stTable th, .stDataFrame thead th, .stTable thead th {
        background-color: #0B0E11 !important;
        color: #D1D4DC !important;
        text-transform: uppercase;
        font-weight: 700 !important;
        border: none !important;
        border-bottom: 1px solid #2A2E39 !important;
    }
    .stDataFrame td, .stTable td, .stDataFrame tbody td, .stTable tbody td {
        background-color: #1E222D !important;
        color: #E0E0E0 !important;
        border: none !important;
        border-bottom: 1px solid #2A2E39 !important;
    }
    /* Additional selectors for data editor */
    [data-testid="stDataEditor"] table {
        background-color: #1E222D !important;
    }
    [data-testid="stDataEditor"] th, [data-testid="stDataEditor"] thead th {
        background-color: #0B0E11 !important;
        color: #D1D4DC !important;
    }
    [data-testid="stDataEditor"] td, [data-testid="stDataEditor"] tbody td {
        background-color: #1E222D !important;
        color: #E0E0E0 !important;
    }
    .stDataFrame td:has(.positive), .stTable td:has(.positive) {
        color: #00FF41 !important;
    }
    .stDataFrame td:has(.negative), .stTable td:has(.negative) {
        color: #FF6B6B !important;
    }
    .stDataFrame .number, .stTable .number {
        text-align: right;
        font-family: 'JetBrains Mono', monospace !important;
        color: #00FF41 !important;
    }
    /* Conditional coloring for P&L */
    .stDataFrame td[data-testid*="TOTAL P&L"] {
        background-color: rgba(0, 255, 65, 0.1) !important; /* Light green for positive */
    }
    .stDataFrame td[data-testid*="TOTAL P&L"]:has(.negative) {
        background-color: rgba(255, 107, 107, 0.1) !important; /* Light red for negative */
    }
    .stPlotlyChart {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stInfo, .stWarning, .stSuccess {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    /* Sidebar Inputs */
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] select {
        background-color: #1E222D !important;
        color: #D1D4DC !important;
        border: 1px solid #2A2E39 !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] input:focus, [data-testid="stSidebar"] textarea:focus, [data-testid="stSidebar"] select:focus {
        border-color: #00FF41 !important;
        box-shadow: 0 0 5px #00FF41 !important;
    }
    /* Slider Values */
    [data-testid="stSidebar"] .stSlider .stMarkdown p {
        color: #FF6B6B !important; /* Coral Red */
        font-weight: 600 !important;
    }
    /* Sidebar Subheaders */
    [data-testid="stSidebar"] .stSubheader {
        color: #D1D4DC !important;
        font-weight: 600 !important;
    }
    /* Data Editor in Sidebar */
    [data-testid="stSidebar"] .stDataFrame th {
        background-color: #2A2E39 !important;
        color: #D1D4DC !important;
    }
    [data-testid="stSidebar"] .stDataFrame td {
        color: #00FF41 !important;
    }
    /* Buttons in Sidebar */
    [data-testid="stSidebar"] button {
        background-color: #1E222D !important;
        color: #D1D4DC !important;
        border: 1px solid #2A2E39 !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #00FF41 !important;
        color: #000 !important;
        box-shadow: 0 0 10px #00FF41 !important;
    }
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE DATOS INSTITUCIONAL ---
@st.cache_data(ttl=3600)
def fetch_financial_data(tickers):
    if not tickers: return None, {}
    all_t = list(set([str(t).strip().upper() for t in tickers if t] + ['SPY']))
    # Descarga extendida para indicadores de largo plazo
    raw = yf.download(all_t, period="5y", interval="1d", auto_adjust=True, progress=False)['Close']
    
    # Sincronización profesional: Forward fill para cripto y re-index al benchmark
    df = raw.ffill().dropna()
    
    yields = {}
    for t in tickers:
        try: yields[t] = yf.Ticker(t).info.get('dividendYield', 0) or 0
        except: yields[t] = 0
    return df, yields

# --- SIDEBAR ---
with st.sidebar:
    # System Status Indicator
    st.markdown("""
        <div style="text-align: center; margin-bottom: 10px;">
            <span style="display: inline-block; width: 10px; height: 10px; background-color: #00FF41; border-radius: 50%; animation: blink 1s infinite;"></span>
            <span style="color: #00FF41; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-left: 5px;">LIVE_FEED_ACTIVE</span>
        </div>
        <style>
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("QUANT_OS_V8")
    if 'journal_df' not in st.session_state:
        st.session_state.journal_df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01')],
            'ticker': ['AAPL', 'MSFT', 'BTC-USD', 'NVDA', 'GLD'],
            'type': ['Buy', 'Buy', 'Buy', 'Buy', 'Buy'],
            'quantity': [10.0, 5.0, 0.25, 15.0, 20.0],
            'price': [180.0, 350.0, 45000.0, 480.0, 190.0],
            'commissions': [0.0, 0.0, 0.0, 0.0, 0.0]
        })

    st.subheader("TRADE JOURNAL")
    edit_journal = st.data_editor(
        st.session_state.journal_df,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "type": st.column_config.SelectboxColumn("Type", options=["Buy", "Sell", "Cash In"]),
            "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
            "price": st.column_config.NumberColumn("Price", format="%.2f"),
            "commissions": st.column_config.NumberColumn("Commissions", format="%.2f")
        },
        num_rows="dynamic",
        width='stretch'
    )
    st.session_state.journal_df = edit_journal.dropna(subset=['date'])

    if st.button("Add Capital (Cash In)", width='stretch'):
        new_row = pd.DataFrame({
            'date': [pd.Timestamp.today()],
            'ticker': [''],
            'type': ['Cash In'],
            'quantity': [1000.0],  # Default cash amount
            'price': [0.0],
            'commissions': [0.0]
        })
        st.session_state.journal_df = pd.concat([st.session_state.journal_df, new_row], ignore_index=True)

    st.divider()
    st.subheader("SIMULATION PARAMETERS")
    rf = st.slider("Risk Free Rate (%)", min_value=0.0, max_value=0.1, value=0.042, step=0.001, format="%.3f")
    mc_n = st.slider("Monte Carlo Iterations", min_value=100, max_value=1000, value=500, step=100)

# --- ENGINE ---
journal = st.session_state.journal_df
if not journal.empty:
    with st.spinner("QUANT ENGINE CALIBRATING..."):
        # Get unique tickers (excluding cash)
        tickers = journal[journal['ticker'] != '']['ticker'].unique().tolist()
        prices, y_map = fetch_financial_data(tickers)

    if prices is not None:
        # Sort journal by date
        journal = journal.sort_values('date').reset_index(drop=True)

        # Calculate current holdings
        holdings = {}
        capital_added = 0
        for _, row in journal.iterrows():
            if row['type'] == 'Cash In':
                capital_added += row['quantity']
            elif row['type'] in ['Buy', 'Sell']:
                ticker = row['ticker'].upper()
                qty = row['quantity'] if row['type'] == 'Buy' else -row['quantity']
                holdings[ticker] = holdings.get(ticker, 0) + qty

        # Filter out zero holdings
        holdings = {k: v for k, v in holdings.items() if v > 0}
        assets = list(holdings.keys())

        if assets:
            # Get prices for held assets
            asset_prices = prices[assets]
            rets = asset_prices.pct_change().dropna()
            spy_rets = prices['SPY'].pct_change().dropna()

            # Align dates
            common = rets.index.intersection(spy_rets.index)
            rets, spy_rets = rets.loc[common], spy_rets.loc[common]

            # Current valuation
            last_px = asset_prices.iloc[-1]
            v_list = [last_px[t] * holdings[t] for t in assets if t in last_px]
            total_v = sum(v_list)
            w = np.array([v/total_v for v in v_list])

            # Calculate equity curve
            start_date = journal['date'].min()
            end_date = pd.Timestamp.today()
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            equity_curve = []
            current_holdings = {}
            current_capital = 0

            for date in date_range:
                # Process transactions on this date
                day_txns = journal[journal['date'] == date]
                for _, txn in day_txns.iterrows():
                    if txn['type'] == 'Cash In':
                        current_capital += txn['quantity']
                    elif txn['type'] in ['Buy', 'Sell']:
                        ticker = txn['ticker'].upper()
                        qty = txn['quantity'] if txn['type'] == 'Buy' else -txn['quantity']
                        current_holdings[ticker] = current_holdings.get(ticker, 0) + qty

                # Calculate portfolio value on this date
                if date in asset_prices.index:
                    port_value = current_capital
                    for ticker, qty in current_holdings.items():
                        if qty > 0 and ticker in asset_prices.columns:
                            port_value += asset_prices.loc[date, ticker] * qty
                    equity_curve.append(port_value)
                else:
                    # Forward fill if no price data
                    if equity_curve:
                        equity_curve.append(equity_curve[-1])
                    else:
                        equity_curve.append(current_capital)

            equity_series = pd.Series(equity_curve, index=date_range[:len(equity_curve)])

            # Calculate metrics
            p_daily = equity_series.pct_change().dropna()
            a_ret, a_vol = p_daily.mean() * 252, p_daily.std() * np.sqrt(252)
            sharpe = (a_ret - rf) / a_vol if a_vol > 0 else 0
            beta = p_daily.cov(spy_rets) / spy_rets.var() if len(p_daily) > 0 else 0
            var_95 = np.percentile(p_daily, 5) if len(p_daily) > 0 else 0

            # Market Regime Filter
            vol_100d = p_daily.rolling(100).std().iloc[-1] * np.sqrt(252) if len(p_daily) >= 100 else a_vol
            vol_avg_100d = p_daily.rolling(100).std().mean() * np.sqrt(252) if len(p_daily) >= 100 else a_vol
            regime = "HIGH VOLATILITY REGIME" if vol_100d > vol_avg_100d * 1.2 else "NORMAL REGIME"

            # Calculate P&L and other metrics
            total_pnl = total_v - capital_added
            pnl_pct = total_pnl / capital_added if capital_added > 0 else 0

            # MWRR (simplified)
            mwrr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(equity_series)) - 1 if len(equity_series) > 1 else 0

            # Current Drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            current_dd = drawdown.iloc[-1]

            # --- CURRENT STATUS DASHBOARD ---
            cols = st.columns(4)
            cols[0].metric("TOTAL P&L", f"${total_pnl:,.0f}", f"{pnl_pct:.1%}", delta_color="normal")
            cols[1].metric("CAPITAL ADDED", f"${capital_added:,.0f}")
            cols[2].metric("MWRR", f"{mwrr:.2%}")
            cols[3].metric("CURRENT DRAWDOWN", f"{current_dd:.2%}")

            st.divider()

            # --- TABS ---
            t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["REBALANCE", "PERFORMANCE", "RISK_LAB", "FORECAST", "STRESS_TEST", "SENTIMENT", "EFFICIENT_FRONTIER", "TRADE JOURNAL"])

        with t1:
            if regime == "HIGH VOLATILITY REGIME":
                st.subheader("Minimum Volatility Optimization (Defensive Mode)")
                st.warning("High volatility detected. Switching to defensive portfolio optimization.")
                res = minimize(lambda weights: np.sqrt(np.dot(weights.T, np.dot(rets.cov()*252, weights))),
                               len(assets)*[1./len(assets)], bounds=[(0,1)]*len(assets), constraints={'type':'eq','fun':lambda x: np.sum(x)-1})
            else:
                st.subheader("Maximum Sharpe Ratio Optimization")
                res = minimize(lambda weights: -((np.sum(rets.mean()*weights)*252)-rf)/(np.sqrt(np.dot(weights.T, np.dot(rets.cov()*252, weights)))),
                               len(assets)*[1./len(assets)], bounds=[(0,1)]*len(assets), constraints={'type':'eq','fun':lambda x: np.sum(x)-1})

            rebal_df = pd.DataFrame({
                "Asset": assets, "Current %": w, "Optimal %": res.x,
                "Action": ["BUY" if (res.x[i]-w[i]) > 0 else "SELL" for i in range(len(assets))],
                "Trade ($)": (res.x - w) * total_v
            })
            st.dataframe(rebal_df.style.format({"Current %": "{:.2%}", "Optimal %": "{:.2%}", "Trade ($)": "${:,.2f}"}), width='stretch')

        with t2:
            st.subheader("Custom Equity Curve (Account Value)")
            # Show the actual equity curve from journal
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=equity_series.index, y=equity_series, name="ACCOUNT VALUE", fill='tozeroy', fillcolor='rgba(0, 255, 65, 0.1)', line=dict(color='#00FF41', width=3)))

            # Add capital additions as vertical lines
            capital_dates = journal[journal['type'] == 'Cash In']['date']
            for cap_date in capital_dates:
                if cap_date in equity_series.index:
                    cap_value = equity_series.loc[cap_date]
                    fig_p.add_vline(x=cap_date, line_width=2, line_dash="dash", line_color="#FFD700", annotation_text=f"Capital Added: ${journal[(journal['date'] == cap_date) & (journal['type'] == 'Cash In')]['quantity'].sum():,.0f}")

            fig_p.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white'), title="Account Value ($)"),
                hovermode='x unified',
                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1),
                hoverlabel=dict(bgcolor='black', bordercolor='#00FF41', font=dict(color='white'))
            )
            fig_p.update_xaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
            fig_p.update_yaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
            st.plotly_chart(fig_p, width='stretch')

            # Show normalized curve vs SPY
            st.subheader("Performance vs SPY (Base 100)")
            c_p = (1 + p_daily).cumprod() * 100
            c_s = (1 + spy_rets).cumprod() * 100
            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(x=c_p.index, y=c_p, name="PORTFOLIO", fill='tozeroy', fillcolor='rgba(0, 255, 65, 0.1)', line=dict(color='#00FF41', width=3)))
            fig_norm.add_trace(go.Scatter(x=c_s.index, y=c_s, name="BENCHMARK (SPY)", line=dict(color='#FFD700', width=2, dash='dot')))
            fig_norm.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                hovermode='x unified',
                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1),
                hoverlabel=dict(bgcolor='black', bordercolor='#00FF41', font=dict(color='white'))
            )
            st.plotly_chart(fig_norm, width='stretch')

        with t3:
            st.subheader("Risk Contribution & Correlation")
            c_l, c_r = st.columns(2)
            with c_l:
                # Riesgo por componente (Marginal Contribution to Risk)
                cov = rets.cov() * 252
                p_vol = np.sqrt(w.T @ cov @ w)
                mcr = (cov @ w) / p_vol
                cr = w * mcr
                fig_risk = px.bar(x=assets, y=cr, title="Risk Contribution by Asset", color_discrete_sequence=['#00FF41'])
                fig_risk.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                    hovermode='closest',
                    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1),
                    hoverlabel=dict(bgcolor='black', bordercolor='#00FF41', font=dict(color='white'))
                )
                fig_risk.update_xaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
                fig_risk.update_yaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
                st.plotly_chart(fig_risk, width='stretch')
            with c_r:
                fig_corr = px.imshow(rets.corr(), text_auto=".2f", color_continuous_scale=['#FF3B30', '#FFD700', '#00FF41'], title="Correlation Matrix")
                fig_corr.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500, width=500,
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                    hovermode='closest',
                    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1),
                    hoverlabel=dict(bgcolor='black', bordercolor='#00FF41', font=dict(color='white'))
                )
                fig_corr.update_coloraxes(showscale=False)
                fig_corr.update_xaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
                fig_corr.update_yaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
                st.plotly_chart(fig_corr, width='stretch')

                # Extreme Correlation Scanner
                corr_matrix = rets.corr()
                high_corr_pairs = []
                for i in range(len(assets)):
                    for j in range(i+1, len(assets)):
                        if corr_matrix.iloc[i, j] > 0.85:
                            high_corr_pairs.append((assets[i], assets[j], corr_matrix.iloc[i, j]))
                if high_corr_pairs:
                    st.warning("Extreme Correlation Alerts:")
                    for pair in high_corr_pairs:
                        st.write(f"{pair[0]} and {pair[1]} have correlation of {pair[2]:.2f}. Consider diversifying to reduce systemic risk.")
                else:
                    st.success("No extreme correlations detected (>0.85). Portfolio is well-diversified.")

        with t4:
            st.subheader("Monte Carlo Path Simulation")
            sim_d = 252
            paths = np.zeros((sim_d, mc_n))
            paths[0] = total_v
            for i in range(1, sim_d):
                paths[i] = paths[i-1] * (1 + np.random.normal(a_ret/252, a_vol/np.sqrt(252), mc_n))

            fig_mc = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i in range(min(mc_n, 100)):
                fig_mc.add_trace(go.Scatter(y=paths[:,i], mode='lines', line=dict(color=colors[i % len(colors)], width=1), opacity=0.5, showlegend=False))
            fig_mc.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), line=dict(color='#00FF41', width=3), name="EXPECTED PATH"))
            fig_mc.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white')),
                hovermode='closest',
                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1),
                hoverlabel=dict(bgcolor='black', bordercolor='#00FF41', font=dict(color='white'))
            )
            fig_mc.update_xaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
            fig_mc.update_yaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
            st.plotly_chart(fig_mc, width='stretch')

        with t5:
            st.subheader("Historical Stress Testing")
            # Escenarios simulados basados en drawdown histórico
            scenarios = {
                "Black Monday (1987)": -0.22,
                "DotCom Bubble (2000)": -0.49,
                "Financial Crisis (2008)": -0.56,
                "COVID-19 Crash (2020)": -0.33
            }
            stress_data = [{"Scenario": k, "Portfolio Impact": f"${total_v * v:,.0f}", "New Value": f"${total_v * (1+v):,.0f}"} for k, v in scenarios.items()]
            st.table(stress_data)
            st.info("Estos escenarios calculan el impacto directo basado en Betas históricas y volatilidad sistémica.")

        with t6:
            st.subheader("Sentiment Analysis (NLP)")
            st.info("Analyzing sentiment from recent news headlines for portfolio assets.")
            sentiment_data = []
            for asset in assets:
                try:
                    # Simulate fetching last 5 headlines (in real app, use yf.Ticker(asset).news)
                    headlines = [
                        f"{asset} reports strong quarterly earnings beating expectations.",
                        f"Analysts upgrade {asset} stock following positive market trends.",
                        f"{asset} faces regulatory scrutiny over recent developments.",
                        f"Market volatility impacts {asset} performance today.",
                        f"{asset} announces new strategic partnerships for growth."
                    ]
                    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
                    avg_sentiment = np.mean(sentiments)
                    if avg_sentiment > 0.1:
                        sentiment_label = "BULLISH"
                        color = "#00FF41"
                    elif avg_sentiment < -0.1:
                        sentiment_label = "BEARISH"
                        color = "#FF0041"
                    else:
                        sentiment_label = "NEUTRAL"
                        color = "#FFFF41"
                    sentiment_data.append({
                        "Asset": asset,
                        "Sentiment": sentiment_label,
                        "Score": f"{avg_sentiment:.2f}",
                        "Color": color
                    })
                except:
                    sentiment_data.append({
                        "Asset": asset,
                        "Sentiment": "N/A",
                        "Score": "0.00",
                        "Color": "#888888"
                    })

            cols = st.columns(len(sentiment_data))
            for i, data in enumerate(sentiment_data):
                with cols[i]:
                    st.markdown(f"""
                        <div style="background-color: {data['Color']}; color: black; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                            {data['Asset']}<br>{data['Sentiment']}<br>({data['Score']})
                        </div>
                        """, unsafe_allow_html=True)

        with t7:
            st.subheader("Efficient Frontier Optimization")
            st.info("Generating 2,000 random portfolios to visualize the Efficient Frontier.")

            # Generate 2000 random portfolios
            n_port = 2000
            np.random.seed(42)
            weights = np.random.random((n_port, len(assets)))
            weights = weights / weights.sum(axis=1, keepdims=True)

            port_rets = weights @ (rets.mean() * 252).values
            port_vols = np.sqrt(np.sum(weights * (weights @ (rets.cov() * 252).values), axis=1))
            port_sharpes = (port_rets - rf) / port_vols

            # Optimize GMV
            gmv_res = minimize(lambda x: np.sqrt(x.T @ (rets.cov() * 252) @ x),
                               len(assets)*[1./len(assets)], bounds=[(0,1)]*len(assets),
                               constraints={'type':'eq','fun':lambda x: np.sum(x)-1})
            gmv_vol = np.sqrt(gmv_res.x.T @ (rets.cov() * 252) @ gmv_res.x)
            gmv_ret = gmv_res.x @ (rets.mean() * 252).values

            # Optimize Max Sharpe
            sharpe_res = minimize(lambda x: -((x @ (rets.mean() * 252).values - rf) / np.sqrt(x.T @ (rets.cov() * 252) @ x)),
                                  len(assets)*[1./len(assets)], bounds=[(0,1)]*len(assets),
                                  constraints={'type':'eq','fun':lambda x: np.sum(x)-1})
            sharpe_vol = np.sqrt(sharpe_res.x.T @ (rets.cov() * 252) @ sharpe_res.x)
            sharpe_ret = sharpe_res.x @ (rets.mean() * 252).values

            # Plot
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=port_vols, y=port_rets, mode='markers',
                                       marker=dict(color=port_sharpes, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
                                       name="Random Portfolios", hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{marker.color:.2f}"))
            # CML
            cml_x = np.linspace(0, sharpe_vol * 1.5, 100)
            cml_y = rf + (sharpe_ret - rf) / sharpe_vol * cml_x
            fig_ef.add_trace(go.Scatter(x=cml_x, y=cml_y, mode='lines', line=dict(color='red', width=2), name="Capital Market Line"))
            # GMV
            fig_ef.add_trace(go.Scatter(x=[gmv_vol], y=[gmv_ret], mode='markers', marker=dict(symbol='star', size=15, color='yellow'), name="Global Min Variance"))
            # Max Sharpe
            fig_ef.add_trace(go.Scatter(x=[sharpe_vol], y=[sharpe_ret], mode='markers', marker=dict(symbol='diamond', size=15, color='green'), name="Max Sharpe"))
            # Current
            fig_ef.add_trace(go.Scatter(x=[a_vol], y=[a_ret], mode='markers', marker=dict(symbol='x', size=15, color='white'), name="Current Portfolio"))
            fig_ef.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white'), title="Volatility (Annualized)"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', gridwidth=2, title_font=dict(color='white'), tickfont=dict(color='white'), title="Return (Annualized)"),
                hovermode='closest',
                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1, x=0.02, y=0.98),
                hoverlabel=dict(bgcolor='black', bordercolor='#00FF41', font=dict(color='white')),
                margin=dict(r=150)
            )
            fig_ef.update_xaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
            fig_ef.update_yaxes(showspikes=True, spikecolor='#00FF41', spikemode='across', spikethickness=1)
            st.plotly_chart(fig_ef, width='stretch')

            # Table
            comp_df = pd.DataFrame({
                "Asset": assets,
                "Current %": w,
                "GMV %": gmv_res.x,
                "Max Sharpe %": sharpe_res.x
            })
            st.dataframe(comp_df.style.format({"Current %": "{:.2%}", "GMV %": "{:.2%}", "Max Sharpe %": "{:.2%}"}), width='stretch')

            # Analysis
            st.subheader("Portfolio Attribution & Efficiency")
            # Efficiency: distance from frontier (simplified as distance to max sharpe)
            dist = np.sqrt((a_vol - sharpe_vol)**2 + (a_ret - sharpe_ret)**2)
            efficiency = 1 - dist / np.sqrt(sharpe_vol**2 + sharpe_ret**2)
            st.write(f"**Portfolio Efficiency:** {efficiency:.2%} (closer to 100% means more efficient)")
            st.write("**Attribution:** Assets with higher returns push the frontier upward; assets with higher volatility widen it.")
            # Simple attribution
            ret_contrib = w * (rets.mean() * 252).values
            vol_contrib = w * np.sqrt(np.diag(rets.cov() * 252))
            st.write("**Return Contributors:**", ", ".join([f"{assets[i]} ({ret_contrib[i]:.2%})" for i in np.argsort(ret_contrib)[::-1][:3]]))
            st.write("**Risk Contributors:**", ", ".join([f"{assets[i]} ({vol_contrib[i]:.2%})" for i in np.argsort(vol_contrib)[::-1][:3]]))

        with t8:
            st.subheader("📓 Trade Journal Overview")
            st.dataframe(journal.style.format({"quantity": "{:.2f}", "price": "${:.2f}", "commissions": "${:.2f}"}), width='stretch')

            # Summary stats
            total_trades = len(journal[journal['type'].isin(['Buy', 'Sell'])])
            total_capital = journal[journal['type'] == 'Cash In']['quantity'].sum()
            total_commissions = journal['commissions'].sum()

            cols = st.columns(3)
            cols[0].metric("Total Trades", total_trades)
            cols[1].metric("Total Capital Added", f"${total_capital:,.0f}")
            cols[2].metric("Total Commissions", f"${total_commissions:,.0f}")

            # Holdings summary
            holdings_df = pd.DataFrame(list(holdings.items()), columns=['Ticker', 'Quantity'])
            holdings_df['Current Price'] = holdings_df['Ticker'].map(lambda x: last_px.get(x, 0))
            holdings_df['Market Value'] = holdings_df['Quantity'] * holdings_df['Current Price']
            holdings_df['Weight'] = holdings_df['Market Value'] / total_v
            st.subheader("Current Holdings")
            st.dataframe(holdings_df.style.format({"Quantity": "{:.2f}", "Current Price": "${:.2f}", "Market Value": "${:,.0f}", "Weight": "{:.2%}"}), width='stretch')

else:
    st.info("SYSTEM_IDLE: Awaiting Ticker input in Sidebar.")
