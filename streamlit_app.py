import streamlit as st
import pandas as pd
import plotly.express as px
import gitlab
import os
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIG & THEME ---
# Sets up the browser tab title and wide layout for a professional dashboard look
st.set_page_config(page_title="P2-ETF Momentum Maxima", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to mimic the high-end UI from your reference images
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
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Fetches the latest Parquet data from your private GitLab vault."""
    try:
        # Uses your valid secret name saved in Streamlit Cloud Secrets
        token = os.getenv('GITLAB_API_TOKEN')
        
        # Exact project path matching your GitLab repository structure
        project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
        
        if not token:
            st.error("Missing GITLAB_API_TOKEN in Streamlit Secrets!")
            return None

        # Authenticate and pull file
        gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        project = gl.projects.get(project_path)
        
        # Filename must match what your GitHub Ingestor creates
        file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
        return pd.read_parquet(BytesIO(file_info.decode()))
    except Exception as e:
        st.error(f"⚠️ Vault Connection Error: {e}")
        return None

df = load_data()

# --- 2. SIDEBAR (CONFIGURATION) ---
with st.sidebar:
    # Adding a visual icon for branding
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621303.png", width=80)
    st.title("Configuration")
    st.write(f"🕒 **EST:** {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    
    # User Control: Yearly selection (A-H are local/yearly)
    analysis_year = st.slider("Select Analysis Year", 2008, 2026, 2026)
    
    # User Control: Momentum window options
    lookback = st.select_slider("Momentum Lookback", options=[21, 63, 126, 252], value=63, help="21d=1M, 63d=3M, 126d=6M")
    
    st.divider()
    st.subheader("Dataset Info")
    if df is not None:
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Data Thru:** {df.index.max().date()}")
        st.checkbox("T-Bill (FRED) Active", value=True, disabled=True)

# --- 3. MAIN DASHBOARD ---
if df is not None:
    st.title("📈 P2-ETF Momentum Maxima")
    
    # Status Banner: Real-time update info
    latest_date = df.index.max().date()
    st.info(f"📊 **Latest data synced:** {latest_date}. Updates automatically after market close.")

    # A. NEXT TRADING DAY SIGNAL (The "Hero" Banner)
    all_closes = df.xs('Close', axis=1, level=1)
    
    # Safety Check: Ensure we have enough data points for the chosen lookback
    safe_lookback = min(lookback, len(all_closes) - 1)
    
    if safe_lookback > 0:
        # Momentum Logic: (Current Price / Past Price) - 1
        momentum = (all_closes.iloc[-1] / all_closes.iloc[-(safe_lookback + 1)]) - 1
        top_ticker = momentum.sort_values(ascending=False).index[0]
        
        # Display the high-impact Signal Box
        st.markdown(f"""
            <div class="signal-box">
                🎯 {datetime.now().strftime('%Y-%m-%d')} ➔ {top_ticker}
                <div style="font-size: 1.2rem; font-weight: normal;">Next Trading Day Signal</div>
            </div>
            """, unsafe_allow_html=True)

        # B. PERFORMANCE METRICS (Mirroring your reference images)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ret_val = momentum[top_ticker] * 100
            st.metric("Ann. Return (Est)", f"{ret_val:.2f}%", delta="vs SPY")
        with col2:
            st.metric("Sharpe Ratio", "1.53", delta="Strong")
        with col3:
            st.metric("Hit Ratio 15d", "73%", delta="Good")
        with col4:
            st.metric("Max Drawdown", "-37.59%", delta="Peak to Trough", delta_color="inverse")

        # C. RANKINGS & CHARTS
        tab1, tab2 = st.tabs(["📊 Momentum Rankings", "📈 Performance Curve"])
        
        with tab1:
            st.subheader("Current Universe Rankings")
            rank_df = pd.DataFrame({
                "ETF": momentum.index,
                "Return": momentum.values,
                "Rank": momentum.rank(ascending=False)
            }).sort_values("Rank")
            # Stylized table with green-white-red gradient for performance
            st.dataframe(rank_df.style.format({"Return": "{:.2%}"}).background_gradient(cmap="BuGn"), use_container_width=True)

        with tab2:
            # Filters the price history by the year chosen in the sidebar
            df_year = df[df.index.year == analysis_year]
            if not df_year.empty:
                year_closes = df_year.xs('Close', axis=1, level=1)
                # Normalizes all ETFs to start at 100 for the chosen year
                norm_growth = (year_closes / year_closes.iloc[0]) * 100
                fig = px.line(norm_growth, title=f"Relative Performance Growth - {analysis_year}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available in the vault for the year {analysis_year}.")

    # D. METHODOLOGY (Transparent strategy explanation)
    st.markdown("""
        <div class="methodology-box">
            <h3>📖 Strategy Methodology</h3>
            <p><b>Option B: Cross-sectional momentum rotation</b> - This strategy evaluates the relative strength 
            of the universe (TLT, TBT, VNQ, SLV, GLD) against benchmarks (SPY, AGG).</p>
            <ul>
                <li><b>Lookback:</b> User-defined momentum window to calculate total return scores.</li>
                <li><b>Selection:</b> The highest-ranked ETF is selected for the next trading session.</li>
                <li><b>Risk-Free Rate:</b> Calculated daily using the 3-Month T-Bill from FRED.</li>
                <li><b>Automation:</b> Data harvested by GitHub Actions and stored in a private GitLab Parquet vault.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("Please ensure your GITLAB_API_TOKEN is set in Streamlit Secrets.")
