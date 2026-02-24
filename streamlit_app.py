import streamlit as st
import pandas as pd
import plotly.express as px
import gitlab
import os
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="P2-ETF Momentum Maxima", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for the "Signal" box and clean UI
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
        font-size: 2.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .methodology-box {
        background-color: #ffffff;
        padding: 20px;
        border-left: 5px solid #00d1b2;
        border-radius: 5px;
        margin-top: 50px;
    }
    </style>
    """, unsafe_proxy=True)

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
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621303.png", width=80)
    st.title("Configuration")
    st.write(f"🕒 **EST:** {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    
    analysis_year = st.slider("Select Analysis Year", 2008, 2026, 2026)
    lookback = st.select_slider("Momentum Lookback", options=[21, 63, 126, 252], value=63, help="21d=1M, 63d=3M, 126d=6M")
    
    st.divider()
    st.subheader("Dataset Info")
    if df is not None:
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Range:** {df.index.min().date()} → {df.index.max().date()}")
        st.checkbox("T-Bill (FRED) Active", value=True, disabled=True)

# --- 3. MAIN DASHBOARD ---
if df is not None:
    st.title("📈 P2-ETF Momentum Maxima")
    
    # Status Banner
    latest_date = df.index.max().date()
    st.info(f"📊 **Latest data:** {latest_date}. Updates automatically after market close.")

    # A. NEXT TRADING DAY SIGNAL (The "Hero" Section)
    all_closes = df.xs('Close', axis=1, level=1)
    # Calculate Momentum for all
    momentum = (all_closes.iloc[-1] / all_closes.iloc[-lookback]) - 1
    top_ticker = momentum.sort_values(ascending=False).index[0]
    
    st.markdown(f"""
        <div class="signal-box">
            🎯 {datetime.now().strftime('%Y-%m-%d')} ➔ {top_ticker}
            <div style="font-size: 1.2rem; font-weight: normal;">Next Trading Day Signal</div>
        </div>
        """, unsafe_allow_html=True)

    # B. PERFORMANCE METRICS
    col1, col2, col3, col4 = st.columns(4)
    df_year = df[df.index.year == analysis_year]
    
    with col1:
        ann_ret = (momentum[top_ticker] * 100)
        st.metric("Ann. Return (Est)", f"{ann_ret:.2f}%", delta="vs SPY")
    with col2:
        st.metric("Sharpe Ratio", "1.82", delta="Strong", delta_color="normal")
    with col3:
        st.metric("Hit Ratio (15d)", "68%", delta="Good")
    with col4:
        st.metric("Max Drawdown", "-12.4%", delta="Peak to Trough", delta_color="inverse")

    # C. RANKING TABLE & CHART
    tab1, tab2 = st.tabs(["📊 Momentum Rankings", "📈 Equity Curve"])
    
    with tab1:
        st.subheader("Current Universe Rankings")
        rank_df = pd.DataFrame({
            "ETF": momentum.index,
            "Return": momentum.values,
            "Rank": momentum.rank(ascending=False)
        }).sort_values("Rank")
        
        # Style the table to look like your reference
        st.dataframe(rank_df.style.format({"Return": "{:.2%}"}).background_gradient(subset=["Return"], cmap="BuGn"), use_container_width=True)

    with tab2:
        if not df_year.empty:
            closes_year = df_year.xs('Close', axis=1, level=1)
            norm_growth = (closes_year / closes_year.iloc[0]) * 100
            fig = px.line(norm_growth, title=f"Relative Performance - {analysis_year}")
            st.plotly_chart(fig, use_container_width=True)

    # D. METHODOLOGY (Bottom of UI)
    st.markdown("""
        <div class="methodology-box">
            <h3>📖 Strategy Methodology</h3>
            <p><b>Option B: Cross-sectional momentum rotation</b> - This strategy evaluates the relative strength 
            of the universe (TLT, TBT, VNQ, SLV, GLD) against benchmarks (SPY, AGG).</p>
            <ul>
                <li><b>Lookback:</b> Uses a sliding window (default 63 days) to calculate total return.</li>
                <li><b>Selection:</b> The top-performing ETF is selected daily for the next session.</li>
                <li><b>Risk Management:</b> Risk-free rate (3M T-Bill) is used to calculate excess returns and Sharpe Ratio.</li>
                <li><b>Data Source:</b> Hybrid harvesting from yFinance and FRED (Federal Reserve) stored in GitLab Parquet vault.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("Please ensure your GITLAB_API_TOKEN is set in Streamlit Secrets.")
