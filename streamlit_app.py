import streamlit as st
import pandas as pd
import gitlab
import os
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIG & PROFESSIONAL THEME ---
st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

# Modern Slate/Teal Theme - Removed all "horrendous" green gradients
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

# --- 2. SIDEBAR (ONLY TWO SPECIFIC CONTROLS) ---
with st.sidebar:
    st.title("⚙️ Model Parameters")
    
    # Training period slider: 3m to 18m only
    training_months = st.select_slider(
        "Model Training Period (Months)",
        options=[3, 6, 9, 12, 15, 18],
        value=9
    )
    training_days = training_months * 21 # Converting months to trading days
    
    st.divider()
    
    # Transaction cost slider: 10-50bps in 5bps steps
    t_costs_bps = st.slider("Transaction Cost (bps)", 10, 50, 10, 5)
    
    st.divider()
    if df is not None:
        st.caption(f"Data Source: 2008 → {df.index.max().date()}")

# --- 3. MAIN DASHBOARD ---
if df is not None:
    st.title("📈 P2-ETF Forecaster")
    
    # Core Universe
    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    prices = df.xs('Close', axis=1, level=1)[universe]
    volumes = df.xs('Volume', axis=1, level=1)[universe]
    
    # A. TRAINING LOGIC (Strict Lookback)
    # Model only "sees" the training_days window to generate the signal
    training_prices = prices.tail(training_days + 1)
    training_volumes = volumes.tail(training_days + 1)
    
    # 1. Price Momentum (Total Return over training window)
    returns = (training_prices.iloc[-1] / training_prices.iloc[0]) - 1
    
    # 2. Volume Fuel Filter (Current Vol vs Window Average)
    vol_filter = training_volumes.iloc[-1] / training_volumes.mean()
    
    # 3. Composite Score
    va_score = returns * vol_filter
    top_ticker = va_score.idxmax()

    # B. SIGNAL BANNER
    st.markdown(f"""
        <div class="signal-banner">
            <div style="text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Next Trading Day Signal</div>
            <div class="signal-text">{datetime.now().date()} ➔ {top_ticker}</div>
            <div style="font-size: 1rem; opacity: 0.8;">Training Period: {training_months} Months • Fee: {t_costs_bps} bps</div>
        </div>
        """, unsafe_allow_html=True)

    # C. PERFORMANCE RANKINGS
    st.subheader(f"📊 {training_months}M Momentum Training Matrix")
    rank_df = pd.DataFrame({
        "ETF": universe,
        "Period Return": returns,
        "Volume Fuel": vol_filter,
        "VA Score": va_score
    }).sort_values("VA Score", ascending=False)
    
    # Professional styling for table
    st.dataframe(rank_df.style.format({
        "Period Return": "{:.2%}", "Volume Fuel": "{:.2f}x", "VA Score": "{:.4f}"
    }), use_container_width=True)

    # D. AUDIT TRAIL
    st.subheader("📋 Audit Trail (Last 15 Trading Days)")
    st.table(prices[top_ticker].tail(15).to_frame(name="Signal Price"))

else:
    st.error("Vault empty. Check GitLab.")
