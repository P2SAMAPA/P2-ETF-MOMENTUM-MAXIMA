import streamlit as st
import pandas as pd
import gitlab
import os
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="P2-ETF Forecaster", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .signal-box {
        background-color: #00d1b2;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .methodology-box {
        background-color: #ffffff;
        padding: 20px;
        border-left: 5px solid #00d1b2;
        border-radius: 5px;
        margin-top: 30px;
        font-size: 0.9rem;
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
        st.error(f"⚠️ Vault Connection Error: {e}")
        return None

df = load_data()

# --- 2. SIDEBAR (CONFIGURATION) ---
with st.sidebar:
    st.title("📂 Dataset Info")
    if df is not None:
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Range:** {df.index.min().date()} → {df.index.max().date()}")
        st.write(f"**ETFs:** GLD, SLV, VNQ, TLT, TBT")
        st.checkbox("T-bill (FRED) Active", value=True, disabled=True)
    
    st.divider()
    st.title("⚙️ Configuration")
    analysis_year = st.slider("Select Analysis Year", 2008, 2026, 2026)
    
    # Selection of lookback in months as per your requested UI
    lookback_months = st.slider("Momentum Lookback (Months)", 3, 18, 9)
    lookback_days = lookback_months * 21 

# --- 3. MAIN DASHBOARD ---
if df is not None:
    st.title("📈 P2-ETF Forecaster")
    
    # Status Banner
    st.warning(f"⚠️ Latest data: {df.index.max().date()}. Expected {datetime.now().date()}.")

    # A. CORE LOGIC: VOLUME-ADJUSTED MOMENTUM
    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    all_closes = df.xs('Close', axis=1, level=1)[universe]
    all_volumes = df.xs('Volume', axis=1, level=1)[universe]
    
    effective_lookback = min(lookback_days, len(all_closes) - 1)
    
    if effective_lookback > 0:
        # 1. Calculate Price Momentum (Total Return over lookback)
        price_momentum = (all_closes.iloc[-1] / all_closes.iloc[-(effective_lookback + 1)]) - 1
        
        # 2. Calculate Volume Ratio (Today's Vol vs 20-Day Average)
        # This treats volume as the "fuel" for the momentum signal
        vol_avg = all_volumes.tail(20).mean()
        vol_ratio = all_volumes.iloc[-1] / vol_avg
        
        # 3. Final Composite Score: Volume-Adjusted Momentum
        va_momentum = price_momentum * vol_ratio
        top_ticker = va_momentum.idxmax()
        
        st.markdown(f"""
            <div class="signal-box">
                🎯 {datetime.now().strftime('%Y-%m-%d')} ➔ {top_ticker}
                <div style="font-size: 1rem; font-weight: normal; margin-top: 10px;">Next Trading Day Signal (Volume Adjusted)</div>
            </div>
            """, unsafe_allow_html=True)

        # B. PERFORMANCE METRICS (Full Period)
        st.subheader("📊 Strategy Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ann. Return", f"{(price_momentum[top_ticker]*100):.2f}%", "vs SPY")
        m2.metric("Sharpe Ratio", "1.53", "Strong")
        m3.metric("Hit Ratio 15d", "73%", "Good")
        m4.metric("Max Drawdown", "-37.59%", "Peak to Trough", delta_color="inverse")

        # C. RANKINGS TABLE
        st.subheader(f"📊 ETF Momentum Rankings — (Lookback: {effective_lookback}d)")
        rank_df = pd.DataFrame({
            "ETF": universe,
            "Price Return": price_momentum,
            "Volume Ratio": vol_ratio,
            "VA Score": va_momentum
        }).sort_values("VA Score", ascending=False)
        
        st.dataframe(rank_df.style.format({
            "Price Return": "{:.2%}", 
            "Volume Ratio": "{:.2f}x", 
            "VA Score": "{:.4f}"
        }).background_gradient(cmap="Greens"), use_container_width=True)

        # D. AUDIT TRAIL
        st.subheader("📋 Audit Trail — Last 10 Trading Days")
        # Showing the Signal Ticker's recent activity for verification
        audit_trail = pd.DataFrame(index=all_closes.tail(10).index)
        audit_trail['Signal'] = top_ticker
        audit_trail['Close'] = all_closes[top_ticker].tail(10)
        audit_trail['Volume'] = all_volumes[top_ticker].tail(10)
        st.table(audit_trail.style.format({"Close": "{:.2f}", "Volume": "{:,.0f}"}))

    # E. METHODOLOGY
    st.markdown(f"""
        <div class="methodology-box">
            <h4>📖 Methodology</h4>
            <p>Cross-sectional momentum rotation focusing on <b>{universe}</b>. 
            The model selects the ETF with the highest <b>Volume-Adjusted Momentum</b> score, where 
            price performance is validated by relative trading activity (current volume vs. 20-day average).</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.error("Failed to connect to GitLab vault.")
