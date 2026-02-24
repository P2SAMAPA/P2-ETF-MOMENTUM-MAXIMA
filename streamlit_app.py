import streamlit as st
import pandas as pd
import plotly.express as px
import gitlab
import os
from io import BytesIO

# --- 1. SETTINGS & DATA LOADING ---
st.set_page_config(page_title="ETF Momentum Maxima", layout="wide")

@st.cache_data(ttl=3600) # Cache for 1 hour to stay fast
def load_data_from_gitlab():
    try:
        token = os.getenv('GITLAB_API_TOKEN')
        project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
        
        gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        project = gl.projects.get(project_path)
        
        # Download the file from GitLab
        file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
        content = file_info.decode()
        
        df = pd.read_parquet(BytesIO(content))
        return df
    except Exception as e:
        st.error(f"Failed to load data from GitLab: {e}")
        return None

df = load_data_from_gitlab()

# --- 2. SIDEBAR (INPUTS) ---
with st.sidebar:
    st.title("🛠 Strategy Settings")
    st.write("Adjust parameters for backtesting.")
    
    # Year Selection (Note: Instruction says I, J, K are fixed, A-H are local)
    analysis_year = st.slider("Select Analysis Year", 2008, 2026, 2024)
    
    # Momentum Window
    lookback = st.number_input("Lookback Period (Days)", min_value=10, max_value=252, value=60)
    
    # Benchmark Selection
    benchmarks = st.multiselect("Compare against:", ['SPY', 'AGG'], default=['SPY'])

# --- 3. MAIN INTERFACE (OUTPUTS) ---
st.title("🚀 ETF Momentum Maxima Dashboard")

if df is not None:
    # Filter by chosen year
    df_year = df[df.index.year == analysis_year].copy()
    
    # --- OUTPUT A: Key Metrics (The "Scoreboard") ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk-Free Rate (Avg)", f"{df_year['CASH']['Rate'].mean():.2f}%")
    with col2:
        st.metric("Total Data Points", len(df_year))
    with col3:
        st.metric("Last Updated", df.index.max().strftime('%Y-%m-%d'))

    # --- OUTPUT B: Visualizations ---
    tab1, tab2 = st.tabs(["📈 Performance", "📊 Momentum Ranking"])

    with tab1:
        st.subheader("Normalized Price Action (Base 100)")
        # Calculate normalized growth starting from 100
        norm_df = (df_year.xs('Close', axis=1, level=1) / df_year.xs('Close', axis=1, level=1).iloc[0]) * 100
        fig = px.line(norm_df, labels={"value": "Price (Indexed to 100)", "index": "Date"})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Current Momentum Rankings")
        # Simple Momentum calculation: (Price / Price n-days ago) - 1
        current_prices = df.xs('Close', axis=1, level=1).iloc[-1]
        past_prices = df.xs('Close', axis=1, level=1).iloc[-lookback]
        momentum_scores = ((current_prices / past_prices) - 1).sort_values(ascending=False)
        
        st.bar_chart(momentum_scores)
        
    # --- OUTPUT C: Raw Data View ---
    with st.expander("🔍 View Raw Parquet Data"):
        st.dataframe(df_year.tail(20))
