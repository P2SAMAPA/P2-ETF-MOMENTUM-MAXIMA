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

# ============================================================
# MODULE-LEVEL CACHED FUNCTIONS
# Defined here (not inside try/fragment) so their identity is
# stable across reruns — this is what makes @st.cache_data work
# correctly and prevents flickering.
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    try:
        token = os.getenv('GITLAB_API_TOKEN') or st.secrets.get("GITLAB_API_TOKEN")
        if not token:
            return None, "❌ GITLAB_API_TOKEN not found in environment variables or Streamlit secrets."

        proj_enc = urllib.parse.quote('p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA', safe='')
        file_enc = urllib.parse.quote('etf_momentum_data.parquet', safe='')
        url = f"https://gitlab.com/api/v4/projects/{proj_enc}/repository/files/{file_enc}/raw?ref=main"

        headers = {"PRIVATE-TOKEN": token}
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code != 200:
            return None, f"❌ GitLab API error: {resp.status_code}"

        content = resp.content

        # GitLab may return binary files base64-encoded even via the /raw endpoint.
        # Detect and decode if necessary before checking the Parquet magic bytes.
        if content[:4] != b'PAR1':
            try:
                content = base64.b64decode(content)
            except Exception:
                pass

        if content[:4] != b'PAR1':
            return None, "❌ File is not a valid Parquet file."

        return pd.read_parquet(BytesIO(content)), None
    except Exception as e:
        return None, f"⚠️ Connection Error: {e}"


@st.cache_data
def compute_rolling_vol(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252)


@st.cache_data
def compute_sma(prices, window=200):
    return prices.rolling(window).mean()


@st.cache_data
def benchmark_metrics(daily_returns_spy, daily_returns_agg, cash_daily_yields):
    results = {}
    rf = cash_daily_yields.mean() * 252
    for label, bm_ret in [('SPY', daily_returns_spy), ('AGG', daily_returns_agg)]:
        bm_ret = bm_ret.dropna()
        bm_ann = (np.prod(1 + bm_ret) ** (252 / len(bm_ret))) - 1 if len(bm_ret) > 0 else 0
        bm_sharpe = (bm_ann - rf) / (np.std(bm_ret) * np.sqrt(252)) if np.std(bm_ret) > 0 else 0
        results[label] = (bm_ann, bm_sharpe)
    return results


@st.cache_data(show_spinner=False)
def get_equity_curve_fig(strat_series, spy_series, agg_series):
    fig, ax = plt.subplots(figsize=(10, 5))
    strat_cum = (1 + strat_series).cumprod() - 1
    spy_cum = (1 + spy_series).cumprod() - 1
    agg_cum = (1 + agg_series).cumprod() - 1
    ax.plot(strat_cum, label='Strategy')
    ax.plot(spy_cum, label='SPY')
    ax.plot(agg_cum, label='AGG')
    ax.legend()
    ax.set_title('Cumulative Returns')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    return fig


@st.cache_data(show_spinner=False)
def run_backtest_with_stop(prices_universe, volumes_universe, cash_daily_yields,
                           daily_returns_universe, daily_returns_spy, daily_returns_agg,
                           rolling_vol, sma_200,
                           training_days, t_cost_pct, stop_loss_pct, z_exit_threshold,
                           use_vol_filter, use_ma_filter, vol_threshold):

    universe = list(prices_universe.columns)
    n = len(prices_universe)

    # Fast window = 1/3 of training period, minimum 5 days
    fast_days = max(5, training_days // 3)

    # ------------------------------------------------------------------
    # VECTORISED SIGNAL COMPUTATION
    # ------------------------------------------------------------------

    price_arr = prices_universe.values.astype(float)       # (n, 5)
    vol_arr   = volumes_universe.values.astype(float)      # (n, 5)

    # --- Slow window (training period) start rows ---
    lookback   = np.minimum(np.full(n, training_days, dtype=int), np.arange(n))
    start_rows = np.maximum(np.arange(n) - lookback, 0)

    # --- Fast window (1/3 training period) start rows ---
    fast_lookback   = np.minimum(np.full(n, fast_days, dtype=int), np.arange(n))
    fast_start_rows = np.maximum(np.arange(n) - fast_lookback, 0)

    # Slow returns: price[t] / price[t - training_days] - 1
    rets_mat = price_arr / price_arr[start_rows] - 1       # (n, 5)

    # Fast returns: price[t] / price[t - fast_days] - 1
    fast_rets_mat = price_arr / price_arr[fast_start_rows] - 1  # (n, 5)

    # Momentum acceleration: fast return minus slow return
    # Positive = momentum speeding up, negative = fading
    accel_mat = fast_rets_mat - rets_mat                   # (n, 5)

    # Z-scores across assets at each row
    rets_mean = rets_mat.mean(axis=1, keepdims=True)
    rets_std  = rets_mat.std(axis=1, keepdims=True) + 1e-6
    zs_mat    = (rets_mat - rets_mean) / rets_std          # (n, 5)

    # Volume fuel: last vol / mean of prior window vols
    vol_df        = pd.DataFrame(vol_arr, index=prices_universe.index, columns=universe)
    vol_roll_mean = vol_df.shift(1).rolling(training_days, min_periods=1).mean()
    vfuel_mat     = vol_arr / (vol_roll_mean.values + 1e-9)  # (n, 5)

    # Ranks across assets (1=worst, 5=best) at each row
    def row_rank(mat):
        temp  = mat.argsort(axis=1)
        ranks = np.empty_like(temp)
        rows  = np.arange(mat.shape[0])[:, None]
        ranks[rows, temp] = np.arange(mat.shape[1])
        return ranks + 1  # 1-indexed

    ret_rank_mat   = row_rank(rets_mat)    # (n, 5)  max 5
    z_rank_mat     = row_rank(zs_mat)      # (n, 5)  max 5
    accel_rank_mat = row_rank(accel_mat)   # (n, 5)  max 5

    # Conditional volume rank: rank 1-5 only if vfuel > 1.0, else 0
    vol_rank_mat = row_rank(vfuel_mat).astype(float)       # (n, 5)
    vol_rank_mat[vfuel_mat <= 1.0] = 0.0                   # no confirmation = 0

    # Total rank sum: max = 5+5+5+5 = 20
    rank_sum_mat = (ret_rank_mat + z_rank_mat +
                    accel_rank_mat + vol_rank_mat)          # (n, 5)  max=20

    # Max z-score per row (used for stop-loss exit)
    max_z_arr = zs_mat.max(axis=1)                         # (n,)

    # Rolling RF hurdle: cumulative cash yield over training window
    # Use log-sum approximation for speed: sum(log(1+rf)) ≈ log(prod(1+rf))
    rf_arr = cash_daily_yields.values.astype(float)
    log_rf = np.log1p(rf_arr)
    rf_cumsum = np.cumsum(log_rf)
    rf_start  = rf_cumsum[start_rows]
    # rf_start for row 0 should be 0
    rf_start[0] = 0.0
    rf_hurdle_arr = np.expm1(rf_cumsum - rf_start)        # (n,)

    # Filters: boolean mask (n, 5)  True = asset is tradeable
    tradeable = np.ones((n, len(universe)), dtype=bool)

    if use_vol_filter:
        vol_mask = rolling_vol.values <= vol_threshold     # (n, 5)
        tradeable &= vol_mask

    if use_ma_filter:
        ma_mask = prices_universe.values > sma_200.values  # (n, 5)
        tradeable &= ma_mask

    # ------------------------------------------------------------------
    # MODEL SIGNAL: best rank_sum among tradeable assets, tie→highest ret
    # ------------------------------------------------------------------
    # Mask out non-tradeable assets by setting their rank_sum very low
    masked_rank = rank_sum_mat.astype(float).copy()
    masked_rank[~tradeable] = -999.0

    # Primary sort: rank_sum desc; secondary: rets desc (encode as fractional)
    rets_norm   = (rets_mat - rets_mat.min(axis=1, keepdims=True)) / \
                  ((rets_mat.max(axis=1, keepdims=True) - rets_mat.min(axis=1, keepdims=True)) + 1e-9) * 0.5
    score_mat   = masked_rank + rets_norm                  # (n, 5)

    best_asset_idx = score_mat.argmax(axis=1)              # (n,) index into universe
    best_rets      = rets_mat[np.arange(n), best_asset_idx]
    any_tradeable  = tradeable.any(axis=1)                 # (n,) bool

    # Model signal: asset name or "CASH"
    model_signals = np.where(
        (~any_tradeable) | (best_rets < rf_hurdle_arr),
        "CASH",
        np.array(universe)[best_asset_idx]
    )

    # ------------------------------------------------------------------
    # STOP-LOSS LOGIC (sequential — must remain a loop, but it's O(n) int ops)
    # ------------------------------------------------------------------
    final_signals  = model_signals.copy()
    stop_active    = False
    strat_rets_arr = np.zeros(n)

    # Pre-build return lookup: daily_returns for each asset + cash
    dr_arr   = daily_returns_universe.values.astype(float)  # (n, 5)
    cash_arr = cash_daily_yields.values.astype(float)        # (n,)
    asset_to_col = {a: i for i, a in enumerate(universe)}

    for i in range(training_days, n):
        msig = model_signals[i]
        mz   = max_z_arr[i]

        if stop_active:
            stop_active = mz <= z_exit_threshold
            sig = "CASH" if stop_active else msig
        else:
            if i >= training_days + 2:
                cum2 = (1 + strat_rets_arr[i-2]) * (1 + strat_rets_arr[i-1]) - 1
                if cum2 <= stop_loss_pct:
                    stop_active = True
                    sig = "CASH"
                else:
                    sig = msig
            else:
                sig = msig

        final_signals[i] = sig
        day_ret = dr_arr[i, asset_to_col[sig]] if sig != "CASH" else cash_arr[i]
        # Transaction cost on switch
        if i > training_days and final_signals[i] != final_signals[i-1]:
            day_ret -= t_cost_pct
        strat_rets_arr[i] = day_ret

    # ------------------------------------------------------------------
    # ASSEMBLE OUTPUT
    # ------------------------------------------------------------------
    idx_slice    = prices_universe.index[training_days:]
    final_slice  = final_signals[training_days:]
    model_slice  = model_signals[training_days:]
    rets_slice   = strat_rets_arr[training_days:]
    maxz_slice   = max_z_arr[training_days:]

    strat_df = pd.DataFrame({
        'Signal':     final_slice,
        'Net_Return': rets_slice,
        'ModelSignal': model_slice,
        'MaxZ':        maxz_slice,
    }, index=idx_slice)

    if len(rets_slice) == 0:
        return pd.DataFrame(), 0, 0, 0, 0

    returns_arr = rets_slice
    ann_ret  = (np.prod(1 + returns_arr) ** (252 / len(returns_arr))) - 1
    rf_mean  = cash_daily_yields.mean() * 252
    sharpe   = (ann_ret - rf_mean) / (np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0

    wealth   = np.cumprod(1 + returns_arr)
    peak     = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_dd   = np.min(drawdown)
    daily_dd = np.min(returns_arr)

    return strat_df, ann_ret, sharpe, max_dd, daily_dd


# ============================================================
# APP
# ============================================================

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
    # DATA LOADING
    # ------------------------------------------------------------
    df, load_error = load_data()
    if load_error:
        st.error(load_error)
        st.stop()
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

    rolling_vol = compute_rolling_vol(daily_returns[universe])
    sma_200 = compute_sma(prices[universe])

    # ------------------------------------------------------------
    # SIDEBAR CONTROLS
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
            value=False,
            key="vol_filter",
            help="Excludes assets with 20‑day annualized volatility above the selected threshold."
        )
        vol_threshold_pct = st.slider(
            "Max annualized volatility (%)",
            min_value=20, max_value=50, value=40, step=5,
            disabled=not use_vol_filter,
            key="vol_threshold_pct",
            help="Assets with 20‑day annualized volatility above this percentage are excluded."
        )
        vol_threshold = vol_threshold_pct / 100.0

        use_ma_filter = st.checkbox(
            "Moving average filter (price > 200d MA)",
            value=False,
            key="ma_filter",
            help="Only assets trading above their 200‑day simple moving average are considered."
        )

    # ------------------------------------------------------------
    # FRAGMENT: only reruns when slider values change
    # ------------------------------------------------------------
    @st.fragment
    def update_dashboard():
        try:
            if 'render_count' not in st.session_state:
                st.session_state.render_count = 0
            st.session_state.render_count += 1

            strat_df, ann_ret, sharpe, max_dd, daily_dd = run_backtest_with_stop(
                prices[universe], volumes[universe], cash_daily_yields,
                daily_returns[universe], daily_returns['SPY'], daily_returns['AGG'],
                rolling_vol, sma_200,
                training_days, t_cost_pct, stop_loss_pct, z_exit_threshold,
                use_vol_filter, use_ma_filter, vol_threshold
            )

            bm = benchmark_metrics(daily_returns['SPY'], daily_returns['AGG'], cash_daily_yields)
            spy_ann, spy_sharpe = bm['SPY']
            agg_ann, agg_sharpe = bm['AGG']

            # Audit trail
            audit_df = strat_df.tail(15)
            hit_ratio = len(audit_df[audit_df['Net_Return'] > 0]) / len(audit_df) if len(audit_df) > 0 else 0

            # Current signal (lightweight recompute for the last row only)
            last_idx = len(prices) - 1
            actual_days = min(training_days, last_idx)
            fast_days   = max(5, actual_days // 3)
            start_idx      = last_idx - actual_days
            fast_start_idx = last_idx - min(fast_days, last_idx)

            window_prices      = prices[universe].iloc[start_idx: last_idx + 1]
            fast_window_prices = prices[universe].iloc[fast_start_idx: last_idx + 1]
            window_vols        = volumes[universe].iloc[start_idx: last_idx + 1]
            window_daily_rf    = cash_daily_yields.iloc[start_idx: last_idx + 1]

            rets      = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
            fast_rets = (fast_window_prices.iloc[-1] / fast_window_prices.iloc[0]) - 1
            accel     = fast_rets - rets
            zs        = (rets - rets.mean()) / (rets.std() + 1e-6)
            v_fuel    = window_vols.iloc[-1] / window_vols.iloc[:-1].mean()

            ret_rank   = rets.rank(method='min', ascending=True).fillna(0).astype(int)
            z_rank     = zs.rank(method='min', ascending=True).fillna(0).astype(int)
            accel_rank = accel.rank(method='min', ascending=True).fillna(0).astype(int)

            # Conditional volume rank: 1-5 only if vfuel > 1.0, else 0
            vol_rank_raw = v_fuel.rank(method='min', ascending=True).fillna(0).astype(int)
            vol_rank     = vol_rank_raw.where(v_fuel > 1.0, 0)

            rank_sum = ret_rank + z_rank + accel_rank + vol_rank
            final_rets, final_zs, final_vols, final_accel = rets, zs, v_fuel, accel
            final_ret_rank, final_z_rank, final_accel_rank, final_vol_rank = ret_rank, z_rank, accel_rank, vol_rank

            valid_assets = universe.copy()
            if use_vol_filter:
                current_vol = rolling_vol.iloc[last_idx]
                valid_assets = [a for a in valid_assets if current_vol[a] <= vol_threshold]
            if use_ma_filter:
                current_price = prices[universe].iloc[last_idx]
                current_sma = sma_200.iloc[last_idx]
                valid_assets = [a for a in valid_assets if current_price[a] > current_sma[a]]

            if not valid_assets:
                curr_sig = "CASH"
            else:
                valid_df = pd.DataFrame({
                    'rank_sum': rank_sum[valid_assets],
                    'return': rets[valid_assets]
                }).sort_values(['rank_sum', 'return'], ascending=[False, False])
                top_asset = valid_df.index[0]
                rf_hurdle = np.prod(1 + window_daily_rf) - 1
                curr_sig = "CASH" if rets[top_asset] < rf_hurdle else top_asset

            # Next trading day projection
            holidays_2026 = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
                             "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
            def next_trading_day(base):
                d = base + timedelta(days=1)
                while d.weekday() >= 5 or d.strftime('%Y-%m-%d') in holidays_2026:
                    d += timedelta(days=1)
                return d
            display_date = next_trading_day(df.index.max().date())

            # ----------------------------------------------------------
            # Render all output into a single container so the browser
            # paints everything in one go rather than streaming elements
            # ----------------------------------------------------------
            container = st.container()
            with container:
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

                # ── Hold Period Analysis (hidden when CASH) ───────────────
                if curr_sig != "CASH" and not strat_df.empty:
                    st.subheader(f"⏱ Optimal Hold Period Analysis — {curr_sig}")

                    # Find all historical dates where model picked curr_sig
                    asset_dates = strat_df[strat_df['Signal'] == curr_sig].index
                    asset_prices = prices[curr_sig]

                    rows = []
                    for hold_days, label in [(1, "1 Day"), (3, "3 Days"), (5, "5 Days")]:
                        gross_rets = []
                        for dt in asset_dates:
                            loc = prices.index.get_loc(dt)
                            end_loc = loc + hold_days
                            if end_loc < len(prices):
                                g = asset_prices.iloc[end_loc] / asset_prices.iloc[loc] - 1
                                gross_rets.append(g)

                        if len(gross_rets) == 0:
                            continue

                        gross_arr    = np.array(gross_rets)
                        net_arr      = gross_arr - 2 * t_cost_pct   # entry + exit cost
                        avg_net      = net_arr.mean()
                        per_day_net  = avg_net / hold_days           # normalised: removes cost-drag bias
                        win_rate     = (net_arr > 0).mean()
                        n_obs        = len(gross_rets)

                        rows.append({
                            "Hold":            label,
                            "Avg Net Ret":     avg_net,
                            "Per-Day Net Ret": per_day_net,
                            "Win Rate":        win_rate,
                            "# Signals":       n_obs
                        })

                    if rows:
                        hold_df  = pd.DataFrame(rows)
                        best_idx = hold_df["Per-Day Net Ret"].idxmax()
                        hold_df["Best"] = hold_df.index.map(
                            lambda i: "⭐ Best" if i == best_idx else ""
                        )
                        st.dataframe(
                            hold_df.style.format({
                                "Avg Net Ret":     "{:.3%}",
                                "Per-Day Net Ret": "{:.3%}",
                                "Win Rate":        "{:.0%}",
                                "# Signals":       "{:.0f}"
                            }),
                            use_container_width=True,
                            key="hold_period_table"
                        )
                        best_row = hold_df.loc[best_idx]
                        st.caption(
                            f"Based on {int(best_row['# Signals'])} historical {curr_sig} signals "
                            f"under current parameter settings. Net of 2× transaction cost "
                            f"({t_costs_bps}bps in + {t_costs_bps}bps out)."
                        )
                        st.warning(
                            "⚠️ **Disclaimer:** Hold period returns shown are historical averages "
                            "based on past signals under the current parameter settings. "
                            "Annualised figures compound short-period averages and will appear "
                            "elevated — they do not represent achievable annual returns. "
                            "Past performance is not indicative of future results. "
                            "This analysis is informational only and does not constitute "
                            "investment advice. Always follow the live model signal on the day.",
                            icon=None
                        )

                fast_days_display = max(5, training_days // 3)
                st.subheader(f"📊 {training_months}M Multi-Factor Ranking Matrix  ·  Accel window: {fast_days_display}d")
                rank_df = pd.DataFrame({
                    "ETF":        universe,
                    "Return":     final_rets,
                    "Ret Rank":   final_ret_rank,
                    "Z-Score":    final_zs,
                    "Z Rank":     final_z_rank,
                    "Accel":      final_accel,
                    "Accel Rank": final_accel_rank,
                    "Vol Fuel":   final_vols,
                    "Vol Rank":   final_vol_rank,
                    "Rank Sum":   rank_sum
                }).sort_values("Rank Sum", ascending=False)
                st.dataframe(
                    rank_df.style.format({
                        "Return":     "{:.2%}",
                        "Ret Rank":   "{:.0f}",
                        "Z-Score":    "{:.2f}",
                        "Z Rank":     "{:.0f}",
                        "Accel":      "{:.2%}",
                        "Accel Rank": "{:.0f}",
                        "Vol Fuel":   "{:.2f}x",
                        "Vol Rank":   "{:.0f}",
                        "Rank Sum":   "{:.0f}"
                    }),
                    use_container_width=True,
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

                if not strat_df.empty:
                    st.subheader("📈 Equity Curve")
                    strat_series = strat_df['Net_Return'].copy()
                    spy_series = daily_returns['SPY'].loc[strat_df.index].copy()
                    agg_series = daily_returns['AGG'].loc[strat_df.index].copy()
                    fig = get_equity_curve_fig(strat_series, spy_series, agg_series)
                    st.pyplot(fig, clear_figure=False)

                # ── Methodology ──────────────────────────────────────────
                st.divider()
                st.subheader("📖 Methodology")
                fast_days_disp = max(5, training_days // 3)
                st.markdown(f"""
**Universe:** GLD · SLV · VNQ · TLT · TBT (Gold, Silver, Real Estate, Long Bonds, Inverse Long Bonds)

**Objective:** Maximum absolute return via systematic momentum rotation. One asset (or CASH) is held at a time.

**Ranking — 4 factors, max score 20:**

| Factor | How it's computed | Max Rank |
|---|---|---|
| **Return Rank** | Total price return over the {training_months}-month training window | 5 |
| **Z-Score Rank** | Cross-sectional z-score of returns — how far above the universe average | 5 |
| **Momentum Acceleration** | Fast return ({fast_days_disp}d, = ⅓ of training window) minus slow return — rewards accelerating momentum | 5 |
| **Volume Confirmation** | Ranks 1–5 only if volume fuel > 1.0× (current vol > prior window avg). Assets below 1.0× score 0 | 5 |

The asset with the highest combined rank is selected, provided its return exceeds the risk-free hurdle (rolling T-bill yield over the training window). Otherwise CASH is held.

**Optional Risk Brakes (sidebar):**
- **Volatility filter** — excludes assets with 20-day annualised volatility above the selected threshold
- **200-day MA filter** — excludes assets trading below their 200-day simple moving average
- **Trailing stop** — switches to CASH if the 2-day cumulative return breaches the stop level; re-enters when the max cross-sectional Z-score recovers above the exit threshold
- **Transaction cost** — applied on every position switch
                """)

        except Exception as frag_err:
            st.error(f"❌ Fragment error:\n{traceback.format_exc()}")

    # ------------------------------------------------------------
    # CALL THE FRAGMENT
    # ------------------------------------------------------------
    update_dashboard()

except Exception as e:
    st.error(f"❌ An unexpected error occurred:\n{traceback.format_exc()}")
