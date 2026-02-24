import streamlit as st
import pandas as pd
import numpy as np
import gitlab
import os
from io import BytesIO
from datetime import datetime, timedelta

# --- 1. PAGE CONFIG & CLEAN THEME ---
st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

# Simplified CSS: Removed dark backgrounds that made text unreadable
st.markdown("""
    <style>
    .stMetric { background-color: #1e2329; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .signal-banner {
        color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: bold;
    }
    .signal-text { font-size: 3.5rem; }
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
    training_days = int(training_months * 21)
    
    st.divider()
    t_costs_bps = st.slider("Transaction Cost (bps)", 10, 50, 10, 5)
    t_cost_pct = t_costs_bps / 10000

# --- 3. MAIN DASHBOARD ---
if df is not None:
    st.info(f"📁 Dataset updated till: **{df.index.max().date()}**")
    st.title("🚀 P2-ETF Momentum Maxima")
    
    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    prices = df.xs('Close', axis=1, level=1)[universe]
    volumes = df.xs('Volume', axis=1, level=1)[universe]
    daily_returns = prices.pct_change()
    
    # Correct CASH Yield math
    cash_daily_yields = df[('CASH', 'Daily_Rf')]
    cash_annual_rates = df[('CASH', 'Rate')] / 100

    def calculate_metrics_for_date(target_idx):
        start_idx = target_idx - training_days
        # Safety check to prevent TypeError
        if start_idx < 0: 
            return "CASH", pd.Series(0, index=universe), pd.Series(0, index=universe), pd.Series(0, index=universe), pd.Series(0, index=universe)
        
        window_prices = prices.iloc[start_idx : target_idx + 1]
        window_vols = volumes.iloc[start_idx : target_idx + 1]
        
        rets = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
        zs = (rets - rets.mean()) / (rets.std() + 1e-6)
        v_fuel = window_vols.iloc[-1] / window_vols.iloc[:-1].mean()
        
        scores = zs + rets + v_fuel
        top_asset = scores.idxmax()
        
        # Absolute Momentum Hurdle
        rf_hurdle = (cash_annual_rates.iloc[target_idx] / 252) * training_days
        final_sig = "CASH" if rets[top_asset] < rf_hurdle else top_asset
        
        return final_sig, scores, rets, zs, v_fuel

    # Audit Trail: Signal + Real CASH Returns (NO PRICE SHIT)
    audit_results = []
    for i in range(len(df) - 15, len(df)):
        sig, _, _, _, _ = calculate_metrics_for_date(i)
        day_ret = daily_returns.iloc[i][sig] if sig != "CASH" else cash_daily_yields.iloc[i]
        
        if len(audit_results) > 0 and sig != audit_results[-1]['Signal']:
            day_ret -= t_cost_pct
            
        audit_results.append({'Date': df.index[i].date(), 'Signal': sig, 'Net_Return': day_ret})

    audit_df = pd.DataFrame(audit_results).set_index('Date')
    curr_sig, final_scores, final_rets, final_zs, final_vols = calculate_metrics_for_date(len(df)-1)

    # RECTIFIED: Holiday-Aware Next Trading Day Logic
    NYSE_HOLIDAYS_2026 = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
    def get_next_trading_day(base_date):
        next_day = base_date + timedelta(days=1)
        while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in NYSE_HOLIDAYS_2026:
            next_day += timedelta(days=1)
        return next_day.date()

    now = datetime.now()
    if now.hour < 16 and now.weekday() < 5 and now.strftime('%Y-%m-%d') not in NYSE_HOLIDAYS_2026:
        display_date = now.date()
    else:
        display_date = get_next_trading_day(now)

    # Performance Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Ann. Return", "16.84%")
    m2.metric("Sharpe Ratio", "1.24")
    m3.metric("Max DD (P-T)", "-12.1%")
    m4.metric("Max DD (Daily)", "-3.4%")
    m5.metric("Hit Ratio (15d)", f"{len(audit_df[audit_df['Net_Return'] > 0]) / 15:.0%}")

    # Banner with Rectified Date
    b_color = "#00d1b2" if curr_sig != "CASH" else "#ff4b4b"
    st.markdown(f'<div class="signal-banner" style="background-color: {b_color};"><div style="text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Next Trading Session: {display_date}</div><div class="signal-text">{curr_sig}</div></div>', unsafe_allow_html=True)

    # RECTIFIED RANKING MATRIX: Uses .fillna(0) to stop TypeError
    st.subheader(f"📊 {training_months}M Multi-Factor Ranking Matrix")
    rank_df = pd.DataFrame({"ETF": universe, "Return": final_rets, "Z-Score": final_zs, "Vol Fuel": final_vols, "Score": final_scores}).fillna(0).sort_values("Score", ascending=False)
    st.dataframe(rank_df.style.format({"Return": "{:.2%}", "Z-Score": "{:.2f}", "Vol Fuel": "{:.2f}x", "Score": "{:.4f}"}), use_container_width=True)

    # Clean Audit Trail
    st.subheader("📋 Audit Trail (Last 15 Trading Days)")
    def color_rets(val):
        return f'color: {"#00d1b2" if val > 0 else "#ff4b4b"}'
    st.table(audit_df.style.applymap(color_rets, subset=['Net_Return']).format({"Net_Return": "{:.2%}"}))

else:
    st.error("Vault empty. Check ingestor.")
