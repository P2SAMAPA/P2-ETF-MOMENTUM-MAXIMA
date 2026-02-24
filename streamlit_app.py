import streamlit as st
import pandas as pd
import numpy as np
import gitlab
import os
import base64
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import traceback

# --- 1. PAGE CONFIG MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

# ------------------------------------------------------------
# ERROR WRAPPER – everything inside this try block
# ------------------------------------------------------------
try:
    # ------------------------------------------------------------
    # CUSTOM STYLING
    # ------------------------------------------------------------
    st.markdown("""
        <style>
        .stMetric { background-color: #1e2329; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
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
        .signal-text { font-size: 2.5rem; }
        </style>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # DATA LOADING FROM GITLAB (with enhanced error checking)
    # ------------------------------------------------------------
    @st.cache_data(ttl=3600)
    def load_data():
        try:
            token = os.getenv('GITLAB_API_TOKEN')
            if token is None:
                st.error("❌ GITLAB_API_TOKEN environment variable not set. Please add it in HF Space Secrets.")
                return None

            project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
            gl = gitlab.Gitlab('https://gitlab.com', private_token=token)

            # Test token and project access
            try:
                project = gl.projects.get(project_path)
            except gitlab.exceptions.GitlabAuthenticationError:
                st.error("❌ GitLab authentication failed. Your token may be invalid or expired.")
                return None
            except gitlab.exceptions.GitlabGetError as e:
                st.error(f"❌ GitLab project not found. Check project path: {project_path}\nError: {e}")
                return None

            # Fetch file metadata
            try:
                file_info = project.files.get(file_path='etf_momentum_data.parquet', ref='main')
            except gitlab.exceptions.GitlabGetError as e:
                st.error(f"❌ File not found in GitLab repository. Expected path: 'etf_momentum_data.parquet'\nError: {e}")
                return None

            # Decode Base64 content
            file_content = base64.b64decode(file_info.content)

            # Quick check: Parquet files start with b'PK'
            if not file_content.startswith(b'PK'):
                st.error("❌ Downloaded file does not appear to be a valid Parquet file (missing PK header).")
                st.write(f"First 100 bytes: {file_content[:100]}")
                return None

            # Read into DataFrame
            df = pd.read_parquet(BytesIO(file_content))
            st.success(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            st.error(f"⚠️ Unexpected error during data loading: {e}")
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
            window_prices = prices.iloc[start_idx : target_idx + 1][universe]
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
        @st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: id})
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

        # Holiday-aware date projection (robust for 2026, but generalized)
        holidays_2026 = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
        def get_next_trading_day(base_date):
            next_day = base_date + timedelta(days=1)
            while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in holidays_2026:
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
        rank_df = pd.DataFrame({"ETF": universe, "Return": final_rets, "Z-Score": final_zs, "Vol Fuel": final_vols, "Score": final_scores}).sort_values("Score", ascending=False)
        # Fixed deprecated parameter: use_container_width → width='stretch'
        st.dataframe(rank_df.style.format({"Return": "{:.2%}", "Z-Score": "{:.2f}", "Vol Fuel": "{:.2f}x", "Score": "{:.4f}"}), width='stretch')

        st.subheader("📋 Audit Trail (Last 15 Trading Days)")
        def color_rets(val):
            return f'color: {"#00d1b2" if val > 0 else "#ff4b4b"}'
        # Fixed deprecated applymap → map
        st.table(audit_df.style.map(color_rets, subset=['Net_Return']).format({"Net_Return": "{:.2%}"}))

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
        st.error("❌ Vault empty. Check ingestor and GitLab connection.")

except Exception as e:
    st.error(f"❌ An unexpected error occurred in the app:\n{traceback.format_exc()}")
