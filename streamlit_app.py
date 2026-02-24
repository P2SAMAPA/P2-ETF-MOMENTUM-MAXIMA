import streamlit as st
import pandas as pd
import numpy as np
import gitlab
import os
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIG & PROFESSIONAL THEME ---
st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .signal-banner {
        background-color: #00d1b2;
        color: #0e1117;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: bold;
    }
    .signal-text { font-size: 3.5rem; letter-spacing: -1px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    try:
        token = os.getenv('GITLAB_API_TOKEN')
        project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
        gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        project = gl.projects.get(project_path)
        file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
        return pd.read_parquet(BytesIO(file_info.decode()))
    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        return None

df = load_data()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Model Parameters")
    training_months = st.select_slider("Model Training Period (Months)", options=[3, 6, 9, 12, 15, 18], value=9)
    training_days = training_months * 21
    
    st.divider()
    t_costs_bps = st.slider("Transaction Cost (bps)", 10, 50, 10, 5)
    t_cost_pct = t_costs_bps / 10000

# --- 3. MAIN DASHBOARD ---
if df is not None:
    # c) Restore missing status banner
    st.info(f"📁 Dataset updated till: **{df.index.max().date()}**")
    st.title("📈 P2-ETF Forecaster")
    
    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    prices = df.xs('Close', axis=1, level=1)[universe]
    volumes = df.xs('Volume', axis=1, level=1)[universe]
    cash_rate = df[('CASH', 'Rate')].iloc[-1] / 100

    # b) Holding Period Optimization & a) Cost Flow
    best_hold_period = 1
    max_net_return = -np.inf
    
    # Simulate 1d, 3d, 5d to find the highest return net of costs
    for hp in [1, 3, 5]:
        hp_prices = prices.iloc[::hp]
        hp_returns = hp_prices.pct_change().dropna()
        # Simplified: Net return = Gross - (Costs * switches)
        est_net = hp_returns.mean().mean() - (t_cost_pct / hp) 
        if est_net > max_net_return:
            max_net_return = est_net
            best_hold_period = hp

    # d) Scoring with Z-Score + Momentum + Volume
    training_prices = prices.tail(training_days + 1)
    mom_returns = (training_prices.iloc[-1] / training_prices.iloc[0]) - 1
    
    # Z-Score of Returns
    z_score = (mom_returns - mom_returns.mean()) / mom_returns.std()
    # Volume Factor
    vol_ratio = volumes.tail(training_days + 1).iloc[-1] / volumes.tail(training_days + 1).mean()
    
    # Final Composite Score
    composite_score = z_score + mom_returns + vol_ratio
    
    # Filter: Absolute Momentum vs Cash
    top_ticker = composite_score.idxmax()
    if mom_returns[top_ticker] < (cash_rate / 12 * training_months):
        final_signal = "CASH"
    else:
        final_signal = top_ticker

    # B. SIGNAL BANNER
    st.markdown(f"""
        <div class="signal-banner">
            <div style="text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Next Trading Day Signal</div>
            <div class="signal-text">{datetime.now().date()} ➔ {final_signal}</div>
            <div style="font-size: 1rem; opacity: 0.8;">Optimum Hold: {best_hold_period}D | Fee: {t_costs_bps} bps</div>
        </div>
        """, unsafe_allow_html=True)

    # C. RANKINGS
    st.subheader(f"📊 Composite Scoring Matrix ({training_months}M Window)")
    rank_df = pd.DataFrame({
        "ETF": universe,
        "Return": mom_returns,
        "Z-Score": z_score,
        "Vol Fuel": vol_ratio,
        "Final Score": composite_score
    }).sort_values("Final Score", ascending=False)
    st.dataframe(rank_df.style.format({"Return": "{:.2%}", "Z-Score": "{:.2f}", "Vol Fuel": "{:.2f}x"}), use_container_width=True)

    # D. AUDIT TRAIL with Color Coding
    st.subheader("📋 Audit Trail (Last 15 Trading Days)")
    audit_trail = pd.DataFrame(index=prices.tail(15).index)
    audit_ticker = top_ticker if final_signal != "CASH" else universe[0]
    audit_trail['Price'] = prices[audit_ticker].tail(15)
    audit_trail['Net_Return'] = prices[audit_ticker].pct_change().tail(15)
    
    # Styling function: Positive Green, Negative Red
    def color_returns(val):
        color = '#00d1b2' if val > 0 else '#ff4b4b'
        return f'color: {color}'

    st.table(audit_trail.style.applymap(color_returns, subset=['Net_Return']).format({"Price": "{:.2f}", "Net_Return": "{:.2%}"}))

else:
    st.error("Vault empty.")
