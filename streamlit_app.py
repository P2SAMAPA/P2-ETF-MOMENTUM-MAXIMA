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
        # Using the valid secret name you confirmed
        token = os.getenv('GITLAB_API_TOKEN')
        
        # Ensure this matches your GitLab URL exactly
        # https://gitlab.com/p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA
        project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
        
        if not token:
            st.error("Missing GITLAB_API_TOKEN in Streamlit Secrets!")
            return None

        gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        project = gl.projects.get(project_path)
        
        # This filename must match what ingestor.py creates
        file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
        content = file_info.decode()
        
        df = pd.read_parquet(BytesIO(content))
        return df
    except gitlab.exceptions.GitlabGetError as e:
        st.error(f"GitLab Error: {e.response_code} - Check project path or filename.")
        return None
    except Exception as e:
        st.error(f"General Error: {e}")
        return None

df = load_data_from_gitlab()

# --- 2. SIDEBAR (INPUTS) ---
with st.sidebar:
    st.title("🛠 Strategy Settings")
    
    # Selection for Year (A-H are local/yearly, I-J-K are cloud/fixed)
    analysis_year = st.slider("Select Analysis Year", 2008, 2026, 2024)
    
    lookback = st.number_input("Lookback Period (Days)", min_value=10, max_value=252, value=60)
    
    benchmarks = st.multiselect("Compare against:", ['SPY', 'AGG'], default=['SPY'])

# --- 3. MAIN INTERFACE (OUTPUTS) ---
st.title("🚀 ETF Momentum Maxima Dashboard")

if df is not None:
    # Filter by chosen year for assets A-H
    df_year = df[df.index.year == analysis_year].copy()
    
    # Metric Display
    col1, col2, col3 = st.columns(3)
    with col1:
        # Check if CASH/Rate exists in the multi-index
        try:
            avg_rf = df_year['CASH']['Rate'].mean()
            st.metric("Avg Risk-Free Rate (FRED)", f"{avg_rf:.2f}%")
        except:
            st.metric("Risk-Free Rate", "N/A")
    with col2:
        st.metric("Days of Data", len(df_year))
    with col3:
        st.metric("Latest Data Point", df.index.max().strftime('%Y-%m-%d'))

    # Visualizations
    tab1, tab2 = st.tabs(["📈 Price Performance", "📊 Momentum Ranking"])

    with tab1:
        # Normalized Price Action (Base 100)
        # Selecting 'Close' level from MultiIndex
        close_prices = df_year.xs('Close', axis=1, level=1)
        norm_df = (close_prices / close_prices.iloc[0]) * 100
        fig = px.line(norm_df, title=f"ETF Relative Growth in {analysis_year}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Momentum Calculation: (Current / Past) - 1
        all_closes = df.xs('Close', axis=1, level=1)
        current = all_closes.iloc[-1]
        past = all_closes.iloc[-min(lookback, len(all_closes))]
        mom_scores = ((current / past) - 1).sort_values(ascending=False)
        
        st.bar_chart(mom_scores)
        
    with st.expander("🔍 View Raw Data Structure"):
        st.write(df_year.tail(10))
