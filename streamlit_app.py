import streamlit as st
import pandas as pd
import plotly.express as px
import gitlab
import os
from io import BytesIO

# --- 1. SETTINGS & DATA LOADING ---
st.set_page_config(page_title="ETF Momentum Maxima", layout="wide")

@st.cache_data(ttl=3600)
def load_data_from_gitlab():
    try:
        token = os.getenv('GITLAB_API_TOKEN')
        # Updated to match your exact GitLab Project Path capitalization
        project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
        
        if not token:
            st.error("Missing GITLAB_API_TOKEN in Streamlit Secrets!")
            return None

        gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        project = gl.projects.get(project_path)
        
        file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
        content = file_info.decode()
        
        df = pd.read_parquet(BytesIO(content))
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

df = load_data_from_gitlab()

# --- 2. SIDEBAR (INPUTS) ---
with st.sidebar:
    st.title("🛠 Strategy Settings")
    # Assets A-H are yearly, I-J-K are cloud-fixed (2008-2026)
    analysis_year = st.slider("Select Analysis Year", 2008, 2026, 2026)
    lookback = st.number_input("Lookback Period (Days)", min_value=1, max_value=252, value=60)
    benchmarks = st.multiselect("Compare against:", ['SPY', 'AGG'], default=['SPY'])

# --- 3. MAIN INTERFACE (OUTPUTS) ---
st.title("🚀 ETF Momentum Maxima Dashboard")

if df is not None:
    # Filter by chosen year
    df_year = df[df.index.year == analysis_year].copy()
    
    if df_year.empty:
        st.warning(f"No data found for the year {analysis_year}. Since the vault was just created, try selecting 2026.")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                avg_rf = df_year['CASH']['Rate'].mean()
                st.metric("Avg Risk-Free Rate", f"{avg_rf:.2f}%")
            except:
                st.metric("Risk-Free Rate", "N/A")
        with col2:
            st.metric("Available Days", len(df_year))
        with col3:
            st.metric("Latest Sync", df.index.max().strftime('%Y-%m-%d'))

        # Tabs
        tab1, tab2 = st.tabs(["📈 Price Performance", "📊 Momentum Ranking"])

        with tab1:
            st.subheader(f"Normalized Growth: {analysis_year} (Base 100)")
            close_prices = df_year.xs('Close', axis=1, level=1)
            
            # GUARD: Only normalize if we have at least 1 row
            if len(close_prices) > 0:
                norm_df = (close_prices / close_prices.iloc[0]) * 100
                fig = px.line(norm_df)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Momentum (Last {lookback} Days)")
            all_closes = df.xs('Close', axis=1, level=1)
            
            # GUARD: Ensure lookback doesn't exceed available history
            actual_lookback = min(lookback, len(all_closes) - 1)
            if actual_lookback > 0:
                current = all_closes.iloc[-1]
                past = all_closes.iloc[-(actual_lookback + 1)]
                mom_scores = ((current / past) - 1).sort_values(ascending=False)
                st.bar_chart(mom_scores)
            else:
                st.info("Not enough historical days yet to calculate momentum.")
                
    with st.expander("🔍 Debug Raw Data Structure"):
        st.write("Last 5 rows of entire dataset:")
        st.dataframe(df.tail(5))
