import streamlit as st
import pandas as pd
import numpy as np
import gitlab
import os
import base64
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

# --- 1. PAGE CONFIG & HIGH-CONTRAST THEME ---
st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

# FIX 1: Remove unsafe_proxy parameter - it's not needed
st.markdown("""
    <style>
    .stMetric { background-color: #1e2329; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    /* Force high-contrast visibility for metric labels and values */
    [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { color: #00d1b2 !important; }
    
    .signal-banner {
        color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: bold;
    }
    .signal-text { font-size: 2.5rem; }  /* Reduced size for better fit */
    </style>
    """, unsafe_allow_html=True)  # This parameter is correct

@st.cache_data(ttl=3600)
def load_data():
    try:
        token = os.getenv('GITLAB_API_TOKEN')
        if not token:
            st.error("⚠️ GITLAB_API_TOKEN environment variable not set")
            return None
            
        project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
        gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
        project = gl.projects.get(project_path)
        
        # Fetch file metadata
        file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
        
        # Decode Base64 content to binary
        file_content = base64.b64decode(file_info.content)
        
        return pd.read_parquet(BytesIO(file_content))
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
    # Standardize data alignment
    df = df.sort_index().ffill()
    
    st.info(f"📁 Dataset updated till: **{df.index.max().date()}**")
    st.title("🚀 P2-ETF Momentum Maxima")
    
    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    benchmarks = ['SPY', 'AGG']
    prices = df.xs('Close', axis=1, level=1)[universe + benchmarks]
    volumes = df.xs('Volume', axis=1, level=1)[universe + benchmarks]
    daily_returns = prices.pct_change()
    
    cash_daily_yields = df[('CASH', 'Daily_Rf')]
    cash_annual_rates = df[('CASH', 'Rate')] / 100

    def calculate_metrics_for_date(target_idx):
        actual_days = min(training_days, target_idx)
        if actual_days < 5: 
            return "CASH", pd.Series(0, index=universe), pd.Series(0, index=universe), pd.Series(0, index=universe), pd.Series(0, index=universe)
            
        start_idx = target_idx - actual_days
        window_prices = prices.iloc[start_idx : target_idx + 1][universe]  # Restrict to universe for scoring
        window_vols = volumes.iloc[start_idx : target_idx + 1][universe]
        window_daily_rf = cash_daily_yields.iloc[start_idx : target_idx + 1]
        
        rets = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
        zs = (rets - rets.mean()) / (rets.std() + 1e-6)
        v_fuel = window_vols.iloc[-1] / window_vols.iloc[:-1].mean()
        
        scores = zs + rets + v_fuel
        top_asset = scores.idxmax()
        
        # Absolute Momentum Hurdle vs Compounded Risk-Free Return
        rf_hurdle = np.prod(1 + window_daily_rf) - 1
        final_sig = "CASH" if rets[top_asset] < rf_hurdle else top_asset
        
        return final_sig, scores, rets, zs, v_fuel

    # Full Backtest for Dynamic Metrics
    @st.cache_data(show_spinner=False)
    def run_backtest(training_days, t_cost_pct):
        signals = []
        strat_returns = []
        prev_sig = None
        for i in range(training_days, len(df)):
            sig, _, _, _, _ = calculate_metrics_for_date(i)
            day_ret = daily_returns.iloc[i][sig] if sig != "CASH" else cash_daily_yields.iloc[i]
            
            if prev_sig is not None and sig != prev_sig:
                day_ret -= t_cost_pct
            strat_returns.append(day_ret)
            signals.append({'Date': df.index[i], 'Signal': sig, 'Net_Return': day_ret})
            prev_sig = sig
        
        if not strat_returns:
            return pd.DataFrame(), 0, 0, 0, 0
        
        strat_df = pd.DataFrame(signals).set_index('Date')
        cum_ret = np.cumprod(1 + np.array(strat_returns)) - 1
        ann_ret = (np.prod(1 + np.array(strat_returns)) ** (252 / len(strat_returns))) - 1
        rf_mean = cash_daily_yields.mean() * 252  # Annualized Rf
        sharpe = (ann_ret - rf_mean) / (np.std(strat_returns) * np.sqrt(252)) if np.std(strat_returns) > 0 else 0
        max_dd = np.min(cum_ret - np.maximum.accumulate(cum_ret)) if len(cum_ret) > 0 else 0
        daily_dd = np.min(strat_returns) if len(strat_returns) > 0 else 0
        
        return strat_df, ann_ret, sharpe, max_dd, daily_dd

    strat_df, ann_ret, sharpe, max_dd, daily_dd = run_backtest(training_days, t_cost_pct)

    # Benchmark Metrics (Dynamic)
    def compute_benchmark_metrics(ticker):
        bm_returns = daily_returns[ticker].dropna()
        bm_ann_ret = (np.prod(1 + bm_returns) ** (252 / len(bm_returns))) - 1 if len(bm_returns) > 0 else 0
        rf_mean = cash_daily_yields.mean() * 252
        bm_sharpe = (bm_ann_ret - rf_mean) / (np.std(bm_returns) * np.sqrt(252)) if np.std(bm_returns) > 0 else 0
        return bm_ann_ret, bm_sharpe

    spy_ann_ret, spy_sharpe = compute_benchmark_metrics('SPY')
    agg_ann_ret, agg_sharpe = compute_benchmark_metrics('AGG')

    # Audit Trail (Last 15 Sessions)
    audit_df = strat_df.tail(15)
    hit_ratio = len(audit_df[audit_df['Net_Return'] > 0]) / len(audit_df) if len(audit_df) > 0 else 0

    curr_sig, final_scores, final_rets, final_zs, final_vols = calculate_metrics_for_date(len(df)-1)

    # Holiday-aware date projection using market calendar
    def get_next_trading_day(base_date):
        try:
            nyse = mcal.get_calendar('NYSE')
            # Get schedule for next 10 days to find the next open day
            schedule = nyse.schedule(start_date=base_date + timedelta(days=1), end_date=base_date + timedelta(days=10))
            if not schedule.empty:
                return schedule.index[0].date()
        except Exception as e:
            st.warning(f"Market calendar error: {e}")
        
        # Fallback if no schedule (unlikely)
        next_day = base_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.date()

    display_date = get_next_trading_day(df.index.max().date())

    # Dashboard Elements
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strat Ann. Return", f"{ann_ret:.2%}")
    col1.metric("SPY Ann. Return", f"{spy_ann_ret:.2%}")
    col1.metric("AGG Ann. Return", f"{agg_ann_ret:.2%}")
    col2.metric("Strat Sharpe", f"{sharpe:.2f}")
    col2.metric("SPY Sharpe", f"{spy_sharpe:.2f}")
    col2.metric("AGG Sharpe", f"{agg_sharpe:.2f}")
    col3.metric("Max DD (P-T)", f"{max_dd:.1%}")
    col4.metric("Max DD (Daily)", f"{daily_dd:.1%}")
    col5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")

    b_color = "#00d1b2" if curr_sig != "CASH" else "#ff4b4b"
    st.markdown(f'<div class="signal-banner" style="background-color: {b_color};"><div style="text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px;">Next Trading Session: {display_date}</div><div class="signal-text">{curr_sig}</div></div>', unsafe_allow_html=True)

    st.subheader(f"📊 {training_months}M Multi-Factor Ranking Matrix")
    
    # FIX 2: Handle NaN/None values before formatting
    rank_df = pd.DataFrame({
        "ETF": universe, 
        "Return": final_rets, 
        "Z-Score": final_zs, 
        "Vol Fuel": final_vols, 
        "Score": final_scores
    }).sort_values("Score", ascending=False)
    
    # Replace NaN values with 0 to avoid formatting errors
    rank_df = rank_df.fillna(0)
    
    # Display dataframe with proper formatting
    st.dataframe(
        rank_df.style.format({
            "Return": "{:.2%}", 
            "Z-Score": "{:.2f}", 
            "Vol Fuel": "{:.2f}x", 
            "Score": "{:.4f}"
        }),
        use_container_width=True  # Replace with width='stretch' if you get deprecation warning
    )

    st.subheader("📋 Audit Trail (Last 15 Trading Days)")
    
    # FIX 3: Handle NaN values in audit trail
    if not audit_df.empty:
        audit_df = audit_df.fillna(0)
        
        def color_rets(val):
            if pd.isna(val):
                return ''
            return f'color: {"#00d1b2" if val > 0 else "#ff4b4b"}'
        
        # FIX 4: Replace deprecated applymap with map
        styled_df = audit_df.style.map(color_rets, subset=['Net_Return']).format({"Net_Return": "{:.2%}"})
        st.table(styled_df)

    # Equity Curve Chart
    if not strat_df.empty:
        st.subheader("📈 Equity Curve")
        strat_cum_ret = (1 + strat_df['Net_Return']).cumprod() - 1
        spy_cum_ret = (1 + daily_returns['SPY'].loc[strat_df.index]).cumprod() - 1
        agg_cum_ret = (1 + daily_returns['AGG'].loc[strat_df.index]).cumprod() - 1
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(strat_cum_ret, label='Strategy')
        ax.plot(spy_cum_ret, label='SPY')
        ax.plot(agg_cum_ret, label='AGG')
        ax.legend()
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Return')
        st.pyplot(fig)

else:
    st.error("⚠️ Unable to load data. Please check your GitLab connection and token.")

# Display any deprecation warnings in a less intrusive way
st.caption("Note: Some display features use deprecated methods that will be updated in future versions.")
