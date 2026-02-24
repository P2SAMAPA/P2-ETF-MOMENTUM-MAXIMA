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
import requests
import urllib.parse

st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

try:
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
    # DATA LOADING (cached)
    # ------------------------------------------------------------
    @st.cache_data(ttl=3600)
    def load_data():
        try:
            token = os.getenv('GITLAB_API_TOKEN')
            if token is None:
                st.error("❌ GITLAB_API_TOKEN environment variable not set.")
                return None

            proj_enc = urllib.parse.quote('p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA', safe='')
            file_enc = urllib.parse.quote('etf_momentum_data.parquet', safe='')
            url = f"https://gitlab.com/api/v4/projects/{proj_enc}/repository/files/{file_enc}/raw?ref=main"

            headers = {"PRIVATE-TOKEN": token}
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code != 200:
                st.error(f"❌ GitLab API error: {resp.status_code}")
                return None

            content = resp.content
            if content[:4] != b'PAR1':
                st.error("❌ File is not a valid Parquet file.")
                return None

            return pd.read_parquet(BytesIO(content))
        except Exception as e:
            st.error(f"⚠️ Connection Error: {e}")
            return None

    df = load_data()
    if df is None:
        st.stop()

    df = df.sort_index().ffill()
    st.info(f"📁 Dataset updated till: **{df.index.max().date()}**")
    st.title("🚀 P2-ETF Momentum Maxima")

    universe = ['GLD', 'SLV', 'VNQ', 'TLT', 'TBT']
    benchmarks = ['SPY', 'AGG']
    prices = df.xs('Close', axis=1, level=1)[universe + benchmarks]
    volumes = df.xs('Volume', axis=1, level=1)[universe + benchmarks]
    daily_returns = prices.pct_change()
    cash_daily_yields = df[('CASH', 'Daily_Rf')]

    # ------------------------------------------------------------
    # SIDEBAR CONTROLS (with keys to preserve state)
    # ------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Model Parameters")
        training_months = st.select_slider(
            "Training Period (Months)",
            options=[3, 6, 9, 12, 15, 18],
            value=9,
            key="training_months"
        )
        training_days = int(training_months * 21)

        st.divider()
        t_costs_bps = st.slider(
            "Transaction Cost (bps)",
            min_value=10, max_value=50, value=10, step=5,
            key="tcost"
        )
        t_cost_pct = t_costs_bps / 10000

        st.divider()
        st.subheader("🛑 Trailing Stop")
        stop_loss_pct = st.slider(
            "Stop loss (2-day cumulative return)",
            min_value=-25, max_value=-10, value=-12, step=1,
            format="%d%%",
            key="stop_loss",
            help="If 2‑day total return ≤ this value, switch to CASH."
        ) / 100.0

        z_exit_threshold = st.slider(
            "Z‑score exit threshold",
            min_value=0.8, max_value=1.8, value=0.9, step=0.1,
            key="z_exit",
            help="Exit CASH when max Z‑score > this value."
        )

        st.divider()
        st.subheader("📊 Additional Filters")
        use_vol_filter = st.checkbox(
            "Volatility filter",
            value=True,
            key="vol_filter",
            help="Excludes assets with 20‑day annualized volatility above the selected threshold."
        )
        # Volatility threshold slider – integer percent (20–50)
        vol_threshold_pct = st.slider(
            "Max annualized volatility (%)",
            min_value=20, max_value=50, value=40, step=5,
            disabled=not use_vol_filter,
            key="vol_threshold_pct",
            help="Assets with 20‑day annualized volatility above this percentage are excluded when filter is active."
        )
        vol_threshold = vol_threshold_pct / 100.0  # convert to decimal

        use_ma_filter = st.checkbox(
            "Moving average filter (price > 200d MA)",
            value=True,
            key="ma_filter",
            help="Only assets trading above their 200‑day simple moving average are considered."
        )

    # ------------------------------------------------------------
    # HELPER FUNCTIONS (cached)
    # ------------------------------------------------------------
    @st.cache_data
    def compute_rolling_vol(returns, window=20):
        return returns.rolling(window).std() * np.sqrt(252)

    @st.cache_data
    def compute_sma(prices, window=200):
        return prices.rolling(window).mean()

    rolling_vol = compute_rolling_vol(daily_returns[universe])
    sma_200 = compute_sma(prices[universe])

    # ------------------------------------------------------------
    # CORE SIGNAL FUNCTION
    # ------------------------------------------------------------
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

        # Apply filters
        valid_assets = universe.copy()
        if use_vol_filter:
            current_vol = rolling_vol.iloc[target_idx]
            valid_assets = [a for a in valid_assets if current_vol[a] <= vol_threshold]

        if use_ma_filter:
            current_price = prices.iloc[target_idx][universe]
            current_sma = sma_200.iloc[target_idx]
            valid_assets = [a for a in valid_assets if current_price[a] > current_sma[a]]

        if not valid_assets:
            return "CASH", scores, rets, zs, v_fuel

        valid_scores = scores[valid_assets]
        top_asset = valid_scores.idxmax()
        rf_hurdle = np.prod(1 + window_daily_rf) - 1
        final_sig = "CASH" if rets[top_asset] < rf_hurdle else top_asset

        return final_sig, scores, rets, zs, v_fuel

    # ------------------------------------------------------------
    # BACKTEST WITH STOP LOSS (cached, includes vol_threshold)
    # ------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def run_backtest_with_stop(training_days, t_cost_pct, stop_loss_pct, z_exit_threshold,
                               use_vol_filter, use_ma_filter, vol_threshold):
        signals = []
        strat_returns = []
        stop_active = False
        prev_sig = None
        returns_history = []

        for i in range(training_days, len(df)):
            model_sig, scores, rets, zs, _ = calculate_metrics_for_date(i)
            max_z = zs.max()

            # Stop logic
            if stop_active:
                if max_z > z_exit_threshold:
                    stop_active = False
                    final_sig = model_sig
                else:
                    final_sig = "CASH"
            else:
                if len(returns_history) >= 2:
                    cum_two = (1 + returns_history[-2]) * (1 + returns_history[-1]) - 1
                    if cum_two <= stop_loss_pct:
                        stop_active = True
                        final_sig = "CASH"
                    else:
                        final_sig = model_sig
                else:
                    final_sig = model_sig

            day_ret = daily_returns.iloc[i][final_sig] if final_sig != "CASH" else cash_daily_yields.iloc[i]
            if prev_sig is not None and final_sig != prev_sig:
                day_ret -= t_cost_pct

            strat_returns.append(day_ret)
            returns_history.append(day_ret)
            signals.append({
                'Date': df.index[i],
                'Signal': final_sig,
                'Net_Return': day_ret,
                'ModelSignal': model_sig,
                'StopActive': stop_active,
                'MaxZ': max_z
            })
            prev_sig = final_sig

        if not strat_returns:
            return pd.DataFrame(), 0, 0, 0, 0

        strat_df = pd.DataFrame(signals).set_index('Date')
        returns_arr = np.array(strat_returns)
        cum_ret = np.cumprod(1 + returns_arr) - 1

        ann_ret = (np.prod(1 + returns_arr) ** (252 / len(returns_arr))) - 1
        rf_mean = cash_daily_yields.mean() * 252
        sharpe = (ann_ret - rf_mean) / (np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0

        wealth = 1 + cum_ret
        peak = np.maximum.accumulate(wealth)
        drawdown = (wealth - peak) / peak
        max_dd = np.min(drawdown)
        daily_dd = np.min(returns_arr)

        return strat_df, ann_ret, sharpe, max_dd, daily_dd

    # Run backtest (with spinner to indicate recomputation)
    with st.spinner("Running backtest..."):
        strat_df, ann_ret, sharpe, max_dd, daily_dd = run_backtest_with_stop(
            training_days, t_cost_pct, stop_loss_pct, z_exit_threshold,
            use_vol_filter, use_ma_filter, vol_threshold
        )

    # Benchmark metrics (cached)
    @st.cache_data
    def benchmark_metrics(ticker):
        bm_ret = daily_returns[ticker].dropna()
        bm_ann = (np.prod(1 + bm_ret) ** (252 / len(bm_ret))) - 1 if len(bm_ret) > 0 else 0
        rf = cash_daily_yields.mean() * 252
        bm_sharpe = (bm_ann - rf) / (np.std(bm_ret) * np.sqrt(252)) if np.std(bm_ret) > 0 else 0
        return bm_ann, bm_sharpe

    spy_ann, spy_sharpe = benchmark_metrics('SPY')
    agg_ann, agg_sharpe = benchmark_metrics('AGG')

    # Audit trail
    audit_df = strat_df.tail(15)
    hit_ratio = len(audit_df[audit_df['Net_Return'] > 0]) / len(audit_df) if len(audit_df) > 0 else 0

    # Current signal and ranking matrix
    curr_sig, final_scores, final_rets, final_zs, final_vols = calculate_metrics_for_date(len(df)-1)

    # Next trading day projection
    holidays_2026 = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
                     "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
    def next_trading_day(base):
        d = base + timedelta(days=1)
        while d.weekday() >= 5 or d.strftime('%Y-%m-%d') in holidays_2026:
            d += timedelta(days=1)
        return d
    display_date = next_trading_day(df.index.max().date())

    # Dashboard
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strat Ann. Return", f"{ann_ret:.2%}")
    col1.metric("SPY Ann. Return", f"{spy_ann:.2%}")
    col1.metric("AGG Ann. Return", f"{agg_ann:.2%}")
    col2.metric("Strat Sharpe", f"{sharpe:.2f}")
    col2.metric("SPY Sharpe", f"{spy_sharpe:.2f}")
    col2.metric("AGG Sharpe", f"{agg_sharpe:.2f}")
    col3.metric("Max DD (P-T)", f"{max_dd:.1%}")
    col4.metric("Max DD (Daily)", f"{daily_dd:.1%}")
    col5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")

    bg = "#00d1b2" if curr_sig != "CASH" else "#ff4b4b"
    st.markdown(
        f'<div class="signal-banner" style="background-color:{bg};">'
        f'<div style="text-transform:uppercase;">Next Session: {display_date}</div>'
        f'<div class="signal-text">{curr_sig}</div></div>',
        unsafe_allow_html=True
    )

    st.subheader(f"📊 {training_months}M Multi-Factor Ranking Matrix")
    rank_df = pd.DataFrame({
        "ETF": universe,
        "Return": final_rets,
        "Z-Score": final_zs,
        "Vol Fuel": final_vols,
        "Score": final_scores
    }).sort_values("Score", ascending=False)
    st.dataframe(
        rank_df.style.format({"Return": "{:.2%}", "Z-Score": "{:.2f}", "Vol Fuel": "{:.2f}x", "Score": "{:.4f}"}),
        width='stretch',
        key="rank_matrix"
    )

    st.subheader("📋 Audit Trail (Last 15 Trading Days)")
    def color_rets(v):
        return f'color: {"#00d1b2" if v > 0 else "#ff4b4b"}'
    st.dataframe(
        audit_df[['Signal', 'Net_Return']].style.map(color_rets, subset=['Net_Return']).format({"Net_Return": "{:.2%}"}),
        use_container_width=True,
        key="audit_trail"
    )

    # Equity curve (cached)
    @st.cache_data(ttl=600)
    def plot_curve():
        fig, ax = plt.subplots(figsize=(10, 5))
        strat_cum = (1 + strat_df['Net_Return']).cumprod() - 1
        spy_cum = (1 + daily_returns['SPY'].loc[strat_df.index]).cumprod() - 1
        agg_cum = (1 + daily_returns['AGG'].loc[strat_df.index]).cumprod() - 1
        ax.plot(strat_cum, label='Strategy')
        ax.plot(spy_cum, label='SPY')
        ax.plot(agg_cum, label='AGG')
        ax.legend()
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3)
        return fig

    if not strat_df.empty:
        st.subheader("📈 Equity Curve")
        st.pyplot(plot_curve())

except Exception as e:
    st.error(f"❌ An unexpected error occurred:\n{traceback.format_exc()}")
