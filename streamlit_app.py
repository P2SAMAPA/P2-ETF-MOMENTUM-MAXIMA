import streamlit as st
import pandas as pd
import numpy as np
import gitlab
import os
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIG & THEME ---
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
    .methodology-box {
        background-color: #161b22;
        padding: 20px;
        border-left: 5px solid #00d1b2;
        border-radius: 8px;
        margin-top: 30px;
    }
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

# --- 2. SIDEBAR (STRICT CONTROLS) ---
with st.sidebar:
    st.title("⚙️ Model Parameters")
    # Training period strictly mapped to 21 trading days per month
    training_months = st.select_slider("Model Training Period (Months)", options=[3, 6, 9, 12, 15, 18], value=9)
    training_days = int(training_months * 21)
    
    st.divider()
    t_costs_bps = st.slider("Transaction Cost (bps)", 10, 50, 10, 5)
    t_cost_pct = t_costs_bps / 10000

# --- 3. MAIN DASHBOARD ---
if df is not None:
    # Restored Dataset Status
    st.info(f"📁 Dataset updated till: **{df.index.max().date()}**")
    st.title("🚀 P2-ETF Momentum Maxima")
    
    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    prices = df.xs('Close', axis=1, level=1)[universe]
    volumes = df.xs('Volume', axis=1, level=1)[universe]
    cash_rate = df[('CASH', 'Rate')].iloc[-1] / 100

    # A. HOLDING PERIOD OPTIMIZATION (1d, 3d, 5d)
    best_hp = 1
    max_perf = -np.inf
    for hp in [1, 3, 5]:
        hp_ret = prices.pct_change(hp).tail(training_days).mean().mean() - (t_cost_pct / hp)
        if hp_ret > max_perf:
            max_perf = hp_ret
            best_hp = hp

    # B. COMPOSITE SCORING (Z-SCORE + MOMENTUM + VOLUME)
    # Slicing data strictly by training_days to ensure slider changes output
    lookback_prices = prices.tail(training_days + 1)
    returns_series = (lookback_prices.iloc[-1] / lookback_prices.iloc[0]) - 1
    
    # Calculate Z-Scores for Price Momentum
    z_scores = (returns_series - returns_series.mean()) / returns_series.std()
    # Volume Fuel Ratio
    vol_fuel = volumes.tail(training_days).iloc[-1] / volumes.tail(training_days).mean()
    
    # Final Rank Score
    final_scores = z_scores + returns_series + vol_fuel
    top_ticker = final_scores.idxmax()

    # Filter: Absolute Momentum vs Risk-Free Rate
    if returns_series[top_ticker] < (cash_rate / 252 * training_days):
        signal = "CASH"
        banner_color = "#ff4b4b"
    else:
        signal = top_ticker
        banner_color = "#00d1b2"

    # C. SIGNAL BANNER
    st.markdown(f"""
        <div class="signal-banner" style="background-color: {banner_color};">
            <div style="text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Next Trading Day Signal</div>
            <div class="signal-text">{datetime.now().date()} ➔ {signal}</div>
            <div style="font-size: 1rem; opacity: 0.8;">Window: {training_months}M | Optimize: {best_hp}D Hold</div>
        </div>
        """, unsafe_allow_html=True)

    # D. PERFORMANCE RANKINGS (Multi-Factor)
    st.subheader(f"📊 {training_months}M Multi-Factor Ranking Matrix")
    rank_df = pd.DataFrame({
        "ETF": universe,
        "Return": returns_series,
        "Z-Score": z_scores,
        "Vol Fuel": vol_fuel,
        "Final Score": final_scores
    }).sort_values("Final Score", ascending=False)
    
    st.dataframe(rank_df.style.format({
        "Return": "{:.2%}", "Z-Score": "{:.2f}", "Vol Fuel": "{:.2f}x", "Final Score": "{:.4f}"
    }), use_container_width=True)

    # E. AUDIT TRAIL (Last 15 Trading Days)
    st.subheader(f"📋 Audit Trail: {top_ticker} (Last 15 Trading Days)")
    audit_data = pd.DataFrame(index=prices.tail(15).index)
    audit_data['Price'] = prices[top_ticker].tail(15)
    audit_data['Net_Return'] = prices[top_ticker].pct_change().tail(15)

    def color_val(val):
        color = '#00d1b2' if val > 0 else '#ff4b4b'
        return f'color: {color}'

    st.table(audit_data.style.applymap(color_val, subset=['Net_Return']).format({
        "Price": "{:.2f}", "Net_Return": "{:.2%}"
    }))

    # F. METHODOLOGY BOX
    st.markdown(f"""
        <div class="methodology-box">
            <h4>📖 Model Methodology</h4>
            <p>This system utilizes a <b>Cross-Sectional Momentum</b> framework filtered by <b>Absolute Momentum</b>.</p>
            <ul>
                <li><b>Training:</b> Data is sliced based on the {training_months}-month slider ({training_days} trading days).</li>
                <li><b>Scoring:</b> A composite Z-Score is generated using Price Momentum and Relative Volume Fuel.</li>
                <li><b>Optimization:</b> The model evaluates 1d, 3d, and 5d holding periods, selecting the frequency with the highest return net of the {t_costs_bps} bps transaction cost.</li>
                <li><b>Safety:</b> If the top ETF underperforms the 3-Month T-Bill (CASH), the system defaults to a defensive posture.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("Vault empty. Please run ingestor.py first.")
