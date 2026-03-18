import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import traceback
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="P2-ETF Forecaster", layout="wide")

# ============================================================
# SDHO INTEGRATION CONSTANTS
# From Dean (2026) "Scale Invariant Dynamics in Market Price Momentum"
# ============================================================

SDHO_OMEGA = 1.15
SDHO_R2 = 0.57
MIN_CASH_BARS_AFTER_STOP = 1

QUADRANT_RALLY_FADING_PENALTY = -3.0
QUADRANT_CRASH_RECOVERY_BONUS = 1.0

PHI_LOOKBACK_MULTIPLIER = {
    'XLE':  0.60,
    'GLD':  0.75,
    'SLV':  0.75,
    'VNQ':  0.90,
    'SPY':  1.00,
    'QQQ':  1.00,
    'XLV':  1.00,
    'XLF':  1.00,
    'XLI':  1.00,
    'HYG':  1.10,
    'VCIT': 1.25,
    'LQD':  1.30,
    'TLT':  1.40,
    'AGG':  1.30,
}


def get_asset_training_days(ticker: str, base_days: int) -> int:
    """Return SDHO Φ-calibrated look-back for a given ETF."""
    mult = PHI_LOOKBACK_MULTIPLIER.get(ticker, 1.0)
    return max(10, int(base_days * mult))


# ============================================================
# MODULE-LEVEL CACHED FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    """
    Load dataset from Hugging Face Dataset.
    
    Returns:
        df: MultiIndex DataFrame with (Ticker, Close/Volume) columns
        error: Error message string if failed, None if successful
    """
    try:
        # Download from Hugging Face (public dataset, no token needed for read)
        file_path = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-momentum-maxima",
            filename="etf_momentum_data.parquet",
            repo_type="dataset"
        )
        
        df = pd.read_parquet(file_path)
        
        # Ensure CASH column exists (backward compatibility check)
        if ('CASH', 'Daily_Rf') not in df.columns:
            # Fallback: create dummy if missing
            df[('CASH', 'Daily_Rf')] = 0.05 / 252
            
        # Ensure we have the latest FRED data for today (in case HF dataset is slightly stale)
        try:
            import pandas_datareader.data as web
            from datetime import datetime, timedelta
            
            # Try to fetch last 30 days of FRED data to fill any recent gaps
            recent_start = datetime.now() - timedelta(days=30)
            rf_recent = web.DataReader('DTB3', 'fred', start=recent_start)
            rf_recent = rf_recent / 100 / 252
            rf_recent.columns = ['Daily_Rf']
            
            # Update only the recent dates that exist in our df
            for date, row in rf_recent.iterrows():
                if date in df.index:
                    df.loc[date, ('CASH', 'Daily_Rf')] = row['Daily_Rf']
                    
        except Exception:
            # If FRED fetch fails, continue with stored data
            pass
            
        return df, None
        
    except Exception as e:
        return None, f"❌ Hugging Face Dataset error: {e}\n\nPlease verify the dataset exists at: https://huggingface.co/datasets/P2SAMAPA/p2-etf-momentum-maxima"


@st.cache_data
def compute_rolling_vol(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252)


@st.cache_data
def compute_sma(prices, window=200):
    return prices.rolling(window).mean()


@st.cache_data
def benchmark_metrics(daily_returns_spy, daily_returns_agg, cash_daily_yields):
    results = {}
    rf = float(cash_daily_yields.mean()) * 252
    for label, bm_ret in [('SPY', daily_returns_spy), ('AGG', daily_returns_agg)]:
        bm_ret = bm_ret.dropna()
        bm_ann = (np.prod(1 + bm_ret.values) ** (252 / len(bm_ret))) - 1 if len(bm_ret) > 0 else 0
        std = float(np.std(bm_ret.values))
        bm_sharpe = (bm_ann - rf) / (std * np.sqrt(252)) if std > 0 else 0
        results[label] = (bm_ann, bm_sharpe)
    return results


@st.cache_data(show_spinner=False)
def get_equity_curve_fig(strat_series, spy_series, agg_series):
    fig, ax = plt.subplots(figsize=(10, 5))
    strat_cum = (1 + strat_series).cumprod() - 1
    spy_cum = (1 + spy_series).cumprod() - 1
    agg_cum = (1 + agg_series).cumprod() - 1
    ax.plot(strat_cum, label='Strategy (SDHO-enhanced)', color='#00d1b2')
    ax.plot(spy_cum,   label='SPY', color='#4a9eff', linestyle='--')
    ax.plot(agg_cum,   label='AGG', color='#ff9f43', linestyle='--')
    ax.legend()
    ax.set_title('Cumulative Returns — SDHO-Enhanced Momentum Strategy')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def run_backtest_with_stop(prices_universe, volumes_universe, cash_daily_yields,
                           daily_returns_universe, daily_returns_spy, daily_returns_agg,
                           rolling_vol, sma_200,
                           training_days, t_cost_pct, stop_loss_pct, z_exit_threshold,
                           use_vol_filter, use_ma_filter, vol_threshold,
                           use_phi_calibration, use_quadrant_filter,
                           quadrant_penalty, quadrant_bonus,
                           use_omega_cooldown):
    """
    Core backtest engine with optional SDHO overlays.
    """

    universe = list(prices_universe.columns)
    n = len(prices_universe)

    price_arr = prices_universe.values.astype(float)
    vol_arr   = volumes_universe.values.astype(float)

    # ── SDHO Enhancement 1: Φ-calibrated per-asset look-backs ────────────
    if use_phi_calibration:
        asset_windows = np.array([
            get_asset_training_days(t, training_days) for t in universe
        ])
        lookback_mat = np.minimum(
            np.tile(asset_windows, (n, 1)),
            np.arange(n)[:, None]
        )
        start_rows_mat = np.maximum(
            np.arange(n)[:, None] - lookback_mat, 0
        )
        rets_mat = np.zeros((n, len(universe)))
        for j in range(len(universe)):
            sr = start_rows_mat[:, j]
            rets_mat[:, j] = price_arr[:, j] / price_arr[sr, j] - 1

        fast_windows = np.maximum(5, asset_windows // 3)
        fast_lookback_mat = np.minimum(
            np.tile(fast_windows, (n, 1)),
            np.arange(n)[:, None]
        )
        fast_start_rows_mat = np.maximum(
            np.arange(n)[:, None] - fast_lookback_mat, 0
        )
        fast_rets_mat = np.zeros((n, len(universe)))
        for j in range(len(universe)):
            sr = fast_start_rows_mat[:, j]
            fast_rets_mat[:, j] = price_arr[:, j] / price_arr[sr, j] - 1

        median_window = int(np.median(asset_windows))
        lookback_scalar = np.minimum(np.full(n, median_window, dtype=int), np.arange(n))
        start_rows = np.maximum(np.arange(n) - lookback_scalar, 0)
    else:
        fast_days = max(5, training_days // 3)
        lookback = np.minimum(np.full(n, training_days, dtype=int), np.arange(n))
        start_rows = np.maximum(np.arange(n) - lookback, 0)
        fast_lookback = np.minimum(np.full(n, fast_days, dtype=int), np.arange(n))
        fast_start_rows = np.maximum(np.arange(n) - fast_lookback, 0)
        rets_mat = price_arr / price_arr[start_rows] - 1
        fast_rets_mat = price_arr / price_arr[fast_start_rows] - 1

    # ── Momentum acceleration (phase space y-coordinate) ─────────────────
    accel_mat = fast_rets_mat - rets_mat

    # ── Z-scores across assets at each row ────────────────────────────────
    rets_mean = rets_mat.mean(axis=1, keepdims=True)
    rets_std  = rets_mat.std(axis=1, keepdims=True) + 1e-6
    zs_mat    = (rets_mat - rets_mean) / rets_std

    # ── Volume fuel ────────────────────────────────────────────────────────
    vol_df        = pd.DataFrame(vol_arr, index=prices_universe.index, columns=universe)
    vol_roll_mean = vol_df.shift(1).rolling(training_days, min_periods=1).mean()
    vfuel_mat     = vol_arr / (vol_roll_mean.values + 1e-9)

    # ── Ranking utility ────────────────────────────────────────────────────
    def row_rank(mat):
        temp  = mat.argsort(axis=1)
        ranks = np.empty_like(temp)
        rows  = np.arange(mat.shape[0])[:, None]
        ranks[rows, temp] = np.arange(mat.shape[1])
        return ranks + 1

    ret_rank_mat   = row_rank(rets_mat)
    z_rank_mat     = row_rank(zs_mat)
    accel_rank_mat = row_rank(accel_mat)

    vol_rank_mat = row_rank(vfuel_mat).astype(float)
    vol_rank_mat[vfuel_mat <= 1.0] = 0.0

    rank_sum_mat = (ret_rank_mat + z_rank_mat +
                    accel_rank_mat + vol_rank_mat).astype(float)

    # ── SDHO Enhancement 2: Phase quadrant filter ─────────────────────────
    if use_quadrant_filter:
        rally_fading    = (rets_mat > 0) & (accel_mat < 0)
        crash_recovery  = (rets_mat < 0) & (accel_mat > 0)

        quadrant_adj = np.where(rally_fading,   quadrant_penalty, 0.0)
        quadrant_adj += np.where(crash_recovery, quadrant_bonus,   0.0)

        rank_sum_mat += quadrant_adj

    max_z_arr = zs_mat.max(axis=1)

    # ── RF hurdle ──────────────────────────────────────────────────────────
    rf_arr    = cash_daily_yields.values.astype(float)
    log_rf    = np.log1p(rf_arr)
    rf_cumsum = np.cumsum(log_rf)
    rf_start  = rf_cumsum[start_rows]
    rf_start[0] = 0.0
    rf_hurdle_arr = np.expm1(rf_cumsum - rf_start)

    # ── Filters ────────────────────────────────────────────────────────────
    tradeable = np.ones((n, len(universe)), dtype=bool)
    if use_vol_filter:
        tradeable &= rolling_vol.values <= vol_threshold
    if use_ma_filter:
        tradeable &= prices_universe.values > sma_200.values

    # ── Best asset selection ───────────────────────────────────────────────
    masked_rank = rank_sum_mat.copy()
    masked_rank[~tradeable] = -999.0

    rets_norm = (rets_mat - rets_mat.min(axis=1, keepdims=True)) / \
                ((rets_mat.max(axis=1, keepdims=True) -
                  rets_mat.min(axis=1, keepdims=True)) + 1e-9) * 0.5
    score_mat  = masked_rank + rets_norm

    best_asset_idx = score_mat.argmax(axis=1)
    best_rets      = rets_mat[np.arange(n), best_asset_idx]
    any_tradeable  = tradeable.any(axis=1)

    model_signals = np.where(
        (~any_tradeable) | (best_rets < rf_hurdle_arr),
        "CASH",
        np.array(universe)[best_asset_idx]
    )

    # ── Stop-loss + SDHO Enhancement 3: Ω-based re-entry cooldown ─────────
    final_signals  = model_signals.copy()
    stop_active    = False
    strat_rets_arr = np.zeros(n)
    cash_bars_held = 0

    dr_arr       = daily_returns_universe.values.astype(float)
    cash_arr     = cash_daily_yields.values.astype(float)
    asset_to_col = {a: i for i, a in enumerate(universe)}

    min_cash_bars = MIN_CASH_BARS_AFTER_STOP if use_omega_cooldown else 0

    for i in range(training_days, n):
        msig = model_signals[i]
        mz   = max_z_arr[i]

        if stop_active:
            cash_bars_held += 1
            cooldown_expired = (cash_bars_held >= min_cash_bars)
            if cooldown_expired and mz <= z_exit_threshold:
                stop_active    = False
                cash_bars_held = 0
                sig = msig
            else:
                sig = "CASH"
        else:
            if i >= training_days + 2:
                cum2 = (1 + strat_rets_arr[i-2]) * (1 + strat_rets_arr[i-1]) - 1
                if cum2 <= stop_loss_pct:
                    stop_active    = True
                    cash_bars_held = 0
                    sig = "CASH"
                else:
                    sig = msig
            else:
                sig = msig

        final_signals[i] = sig
        day_ret = dr_arr[i, asset_to_col[sig]] if sig != "CASH" else cash_arr[i]
        if i > training_days and final_signals[i] != final_signals[i-1]:
            day_ret -= t_cost_pct
        strat_rets_arr[i] = day_ret

    # ── Assemble output ────────────────────────────────────────────────────
    idx_slice   = prices_universe.index[training_days:]
    final_slice = final_signals[training_days:]
    model_slice = model_signals[training_days:]
    rets_slice  = strat_rets_arr[training_days:]
    maxz_slice  = max_z_arr[training_days:]

    strat_df = pd.DataFrame({
        'Signal':      final_slice,
        'Net_Return':  rets_slice,
        'ModelSignal': model_slice,
        'MaxZ':        maxz_slice,
    }, index=idx_slice)

    if len(rets_slice) == 0:
        return pd.DataFrame(), 0, 0, 0, 0

    returns_arr = rets_slice
    ann_ret = (np.prod(1 + returns_arr) ** (252 / len(returns_arr))) - 1
    rf_mean = cash_daily_yields.mean() * 252
    sharpe  = (ann_ret - rf_mean) / (np.std(returns_arr) * np.sqrt(252)) \
              if np.std(returns_arr) > 0 else 0

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
        .sdho-badge {
            background-color: #1a1f2e;
            border: 1px solid #2d4a6e;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 0.8rem;
            color: #4a9eff;
            margin-bottom: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

    # ── Data loading ──────────────────────────────────────────────────────
    df, load_error = load_data()
    if load_error:
        st.error(load_error)
        st.stop()
    if df is None:
        st.stop()

    df = df.sort_index().ffill()
    st.info(f"📁 Dataset updated till: **{df.index.max().date()}** | Source: **Hugging Face Dataset**")
    st.title("🚀 P2-ETF Momentum Maxima — SDHO Enhanced")

    UNIVERSE_FI = ['GLD', 'SLV', 'VNQ', 'TLT', 'LQD', 'HYG', 'VCIT']
    UNIVERSE_EQ = ['SPY', 'QQQ', 'XLV', 'XLF', 'XLE', 'XLI']
    benchmarks  = ['SPY', 'AGG']

    all_tickers   = list(dict.fromkeys(UNIVERSE_FI + UNIVERSE_EQ + benchmarks))
    prices        = df.xs('Close',  axis=1, level=1)[all_tickers].copy()
    volumes       = df.xs('Volume', axis=1, level=1)[all_tickers].copy()
    daily_returns = prices.pct_change()
    cash_daily_yields = df[('CASH', 'Daily_Rf')]

    rolling_vol_fi = compute_rolling_vol(daily_returns[UNIVERSE_FI])
    rolling_vol_eq = compute_rolling_vol(daily_returns[UNIVERSE_EQ])
    sma_200_fi     = compute_sma(prices[UNIVERSE_FI])
    sma_200_eq     = compute_sma(prices[UNIVERSE_EQ])

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Model Parameters")

        st.subheader("📂 Universe")
        selected_option = st.radio(
            "Select Universe",
            options=["Option A — Fixed Income", "Option B — Equities"],
            key="selected_option",
            label_visibility="collapsed"
        )
        universe     = UNIVERSE_FI if "Fixed Income" in selected_option else UNIVERSE_EQ
        option_label = "Fixed Income" if "Fixed Income" in selected_option else "Equities"
        rolling_vol  = rolling_vol_fi if "Fixed Income" in selected_option else rolling_vol_eq
        sma_200      = sma_200_fi     if "Fixed Income" in selected_option else sma_200_eq

        st.divider()
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
            help="If 2-day total return ≤ this value, switch to CASH."
        ) / 100.0

        z_exit_threshold = st.slider(
            "Z-score exit threshold",
            min_value=0.8, max_value=1.8, value=0.9, step=0.1,
            key="z_exit",
            help="Exit CASH when max Z-score > this value."
        )

        st.divider()
        st.subheader("📊 Additional Filters")
        use_vol_filter = st.checkbox(
            "Volatility filter",
            value=False,
            key="vol_filter"
        )
        vol_threshold_pct = st.slider(
            "Max annualized volatility (%)",
            min_value=20, max_value=50, value=40, step=5,
            disabled=not use_vol_filter,
            key="vol_threshold_pct"
        )
        vol_threshold = vol_threshold_pct / 100.0

        use_ma_filter = st.checkbox(
            "Moving average filter (price > 200d MA)",
            value=False,
            key="ma_filter"
        )

        # ── SDHO Enhancement Controls ─────────────────────────────────
        st.divider()
        st.subheader("🔬 SDHO Enhancements")
        st.caption("Based on Dean (2026) — Scale Invariant Dynamics in Market Price Momentum")

        use_phi_calibration = st.checkbox(
            "Φ-calibrated look-backs",
            value=True,
            key="phi_cal",
            help=(
                "Applies per-asset look-back windows derived from the scaled "
                "restoring force Φ. Energy ETFs (high Φ) use shorter windows; "
                "bond ETFs (low Φ) use longer windows."
            )
        )

        use_quadrant_filter = st.checkbox(
            "Phase quadrant filter",
            value=True,
            key="quad_filter",
            help=(
                "Penalises assets in the SDHO 'rally fading' quadrant "
                "(positive momentum, negative acceleration) where mean "
                "reversion is predicted to be imminent."
            )
        )

        if use_quadrant_filter:
            quadrant_penalty = float(st.slider(
                "Rally fading penalty",
                min_value=-6, max_value=-1, value=-3, step=1,
                key="quad_penalty",
                help="Rank penalty applied to assets with positive momentum but negative acceleration."
            ))
            quadrant_bonus = float(st.slider(
                "Crash recovery bonus",
                min_value=0, max_value=3, value=1, step=1,
                key="quad_bonus",
                help="Rank bonus applied to assets with negative momentum but positive acceleration."
            ))
        else:
            quadrant_penalty = QUADRANT_RALLY_FADING_PENALTY
            quadrant_bonus   = QUADRANT_CRASH_RECOVERY_BONUS

        use_omega_cooldown = st.checkbox(
            "Ω re-entry cooldown",
            value=True,
            key="omega_cool",
            help=(
                "After a stop-loss trigger, enforces a minimum 1-day CASH hold "
                "derived from the SDHO momentum half-life (Ω≈1.15, t½≈1 day "
                "at daily resolution). Prevents whipsaw re-entry."
            )
        )

    # ── Fragment: only reruns when controls change ─────────────────────────
    @st.fragment
    def update_dashboard():
        try:
            strat_df, ann_ret, sharpe, max_dd, daily_dd = run_backtest_with_stop(
                prices[universe], volumes[universe], cash_daily_yields,
                daily_returns[universe], daily_returns['SPY'], daily_returns['AGG'],
                rolling_vol, sma_200,
                training_days, t_cost_pct, stop_loss_pct, z_exit_threshold,
                use_vol_filter, use_ma_filter, vol_threshold,
                use_phi_calibration, use_quadrant_filter,
                quadrant_penalty, quadrant_bonus,
                use_omega_cooldown
            )

            bm = benchmark_metrics(daily_returns['SPY'], daily_returns['AGG'], cash_daily_yields)
            spy_ann, spy_sharpe = bm['SPY']
            agg_ann, agg_sharpe = bm['AGG']

            audit_df  = strat_df.tail(15)
            hit_ratio = len(audit_df[audit_df['Net_Return'] > 0]) / len(audit_df) \
                        if len(audit_df) > 0 else 0

            # ── Current signal (last-row recompute) ───────────────────
            last_idx    = len(prices) - 1
            actual_days = min(training_days, last_idx)
            fast_days   = max(5, actual_days // 3)

            if use_phi_calibration:
                rets_dict      = {}
                fast_rets_dict = {}
                for t in universe:
                    cal_days  = get_asset_training_days(t, actual_days)
                    fast_cal  = max(5, cal_days // 3)
                    s_idx     = max(0, last_idx - cal_days)
                    fs_idx    = max(0, last_idx - fast_cal)
                    wp        = prices[t].iloc[s_idx: last_idx + 1]
                    fwp       = prices[t].iloc[fs_idx: last_idx + 1]
                    rets_dict[t]      = (wp.iloc[-1] / wp.iloc[0]) - 1
                    fast_rets_dict[t] = (fwp.iloc[-1] / fwp.iloc[0]) - 1
                rets      = pd.Series(rets_dict)
                fast_rets = pd.Series(fast_rets_dict)
            else:
                start_idx      = last_idx - actual_days
                fast_start_idx = last_idx - min(fast_days, last_idx)
                window_prices      = prices[universe].iloc[start_idx: last_idx + 1]
                fast_window_prices = prices[universe].iloc[fast_start_idx: last_idx + 1]
                rets      = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
                fast_rets = (fast_window_prices.iloc[-1] / fast_window_prices.iloc[0]) - 1

            accel  = fast_rets - rets
            zs     = (rets - rets.mean()) / (rets.std() + 1e-6)

            start_idx_vol  = max(0, last_idx - actual_days)
            window_vols    = volumes[universe].iloc[start_idx_vol: last_idx + 1]
            window_daily_rf = cash_daily_yields.iloc[start_idx_vol: last_idx + 1]
            v_fuel = window_vols.iloc[-1] / window_vols.iloc[:-1].mean()

            ret_rank   = rets.rank(method='min', ascending=True).fillna(0).astype(int)
            z_rank     = zs.rank(method='min', ascending=True).fillna(0).astype(int)
            accel_rank = accel.rank(method='min', ascending=True).fillna(0).astype(int)
            vol_rank_raw = v_fuel.rank(method='min', ascending=True).fillna(0).astype(int)
            vol_rank   = vol_rank_raw.where(v_fuel > 1.0, 0)

            rank_sum = (ret_rank + z_rank + accel_rank + vol_rank).astype(float)

            if use_quadrant_filter:
                rally_fading   = (rets > 0) & (accel < 0)
                crash_recovery = (rets < 0) & (accel > 0)
                rank_sum += rally_fading.map({True: quadrant_penalty,  False: 0.0})
                rank_sum += crash_recovery.map({True: quadrant_bonus, False: 0.0})

            valid_assets = universe.copy()
            if use_vol_filter:
                current_vol = rolling_vol.iloc[last_idx]
                valid_assets = [a for a in valid_assets if current_vol[a] <= vol_threshold]
            if use_ma_filter:
                current_price = prices[universe].iloc[last_idx]
                current_sma   = sma_200.iloc[last_idx]
                valid_assets  = [a for a in valid_assets if current_price[a] > current_sma[a]]

            if not valid_assets:
                curr_sig = "CASH"
            else:
                valid_df = pd.DataFrame({
                    'rank_sum': rank_sum[valid_assets],
                    'return':   rets[valid_assets]
                }).sort_values(['rank_sum', 'return'], ascending=[False, False])
                top_asset = valid_df.index[0]
                rf_hurdle = np.prod(1 + window_daily_rf) - 1
                curr_sig  = "CASH" if rets[top_asset] < rf_hurdle else top_asset

            # ── Next trading day ──────────────────────────────────────
            holidays_2026 = [
                "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
                "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
                "2026-11-26", "2026-12-25"
            ]
            def next_trading_day(base):
                d = base + timedelta(days=1)
                while d.weekday() >= 5 or d.strftime('%Y-%m-%d') in holidays_2026:
                    d += timedelta(days=1)
                return d
            display_date = next_trading_day(df.index.max().date())

            # ── Render ────────────────────────────────────────────────
            container = st.container()
            with container:

                # SDHO status badges
                active_enhancements = []
                if use_phi_calibration:
                    active_enhancements.append("Φ-calibrated look-backs")
                if use_quadrant_filter:
                    active_enhancements.append(f"Quadrant filter (penalty={quadrant_penalty:+.0f})")
                if use_omega_cooldown:
                    active_enhancements.append("Ω cooldown (1-day min)")
                if active_enhancements:
                    badge_html = " &nbsp;|&nbsp; ".join(
                        [f"🔬 {e}" for e in active_enhancements]
                    )
                    st.markdown(
                        f'<div class="sdho-badge">SDHO active: {badge_html}</div>',
                        unsafe_allow_html=True
                    )

                st.subheader(
                    f"{'🏦' if option_label == 'Fixed Income' else '📈'} "
                    f"{option_label} Universe — {selected_option.split('—')[0].strip()}"
                )
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Strat Ann. Return", f"{ann_ret:.2%}")
                col1.metric("SPY Ann. Return",   f"{spy_ann:.2%}")
                col1.metric("AGG Ann. Return",   f"{agg_ann:.2%}")
                col2.metric("Strat Sharpe", f"{sharpe:.2f}")
                col2.metric("SPY Sharpe",   f"{spy_sharpe:.2f}")
                col2.metric("AGG Sharpe",   f"{agg_sharpe:.2f}")
                col3.metric("Max DD (P-T)",    f"{max_dd:.1%}")
                col4.metric("Max DD (Daily)",  f"{daily_dd:.1%}")
                col5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")

                bg = "#00d1b2" if curr_sig != "CASH" else "#ff4b4b"
                st.markdown(
                    f'<div class="signal-banner" style="background-color:{bg};">'
                    f'<div style="text-transform:uppercase;">Next Session: {display_date}</div>'
                    f'<div class="signal-text">{curr_sig}</div></div>',
                    unsafe_allow_html=True
                )

                # ── Hold Period Analysis ───────────────────────────────
                if curr_sig != "CASH" and not strat_df.empty:
                    st.subheader(f"⏱ Optimal Hold Period Analysis — {curr_sig}")

                    asset_dates  = strat_df[strat_df['Signal'] == curr_sig].index
                    asset_prices = prices[curr_sig]

                    rows = []
                    for hold_days, label in [(1, "1 Day"), (3, "3 Days"), (5, "5 Days")]:
                        gross_rets = []
                        for dt in asset_dates:
                            loc     = prices.index.get_loc(dt)
                            end_loc = loc + hold_days
                            if end_loc < len(prices):
                                g = asset_prices.iloc[end_loc] / asset_prices.iloc[loc] - 1
                                gross_rets.append(g)
                        if len(gross_rets) == 0:
                            continue

                        gross_arr   = np.array(gross_rets)
                        net_arr     = gross_arr - 2 * t_cost_pct
                        avg_net     = net_arr.mean()
                        per_day_net = avg_net / hold_days
                        win_rate    = (net_arr > 0).mean()
                        n_obs       = len(gross_rets)

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
                            hold_df[["Hold", "Avg Net Ret", "Per-Day Net Ret",
                                     "Win Rate", "Best"]].style.format({
                                "Avg Net Ret":     "{:.3%}",
                                "Per-Day Net Ret": "{:.3%}",
                                "Win Rate":        "{:.0%}",
                            }),
                            use_container_width=True,
                            key="hold_period_table"
                        )
                        best_row = hold_df.loc[best_idx]
                        st.caption(
                            f"Based on {int(best_row['# Signals'])} historical {curr_sig} signals. "
                            f"Net of 2× transaction cost ({t_costs_bps}bps in + {t_costs_bps}bps out)."
                        )
                        st.warning(
                            "⚠️ **Disclaimer:** Hold period returns are historical averages under "
                            "current parameter settings. Past performance is not indicative of "
                            "future results. This is informational only.",
                            icon=None
                        )

                # ── Ranking Matrix ────────────────────────────────────
                fast_days_display = max(5, training_days // 3)
                sdho_note = " · Φ-windows active" if use_phi_calibration else ""
                st.subheader(
                    f"📊 {training_months}M Ranking Matrix · {option_label} "
                    f"· Accel window: {fast_days_display}d{sdho_note}"
                )

                quadrant_labels = []
                for asset in universe:
                    r = rets[asset]
                    a = accel[asset]
                    if r > 0 and a > 0:
                        quadrant_labels.append("Q1 ↗ Accel")
                    elif r > 0 and a < 0:
                        quadrant_labels.append("Q2 ↘ Fading ⚠️")
                    elif r < 0 and a < 0:
                        quadrant_labels.append("Q3 ↙ Decline")
                    else:
                        quadrant_labels.append("Q4 ↖ Recovery ✅")

                rank_df = pd.DataFrame({
                    "ETF":        universe,
                    "Return":     rets,
                    "Ret Rank":   ret_rank,
                    "Z-Score":    zs,
                    "Z Rank":     z_rank,
                    "Accel":      accel,
                    "Accel Rank": accel_rank,
                    "Vol Fuel":   v_fuel,
                    "Vol Rank":   vol_rank,
                    "Rank Sum":   rank_sum,
                    "Quadrant":   pd.Series(quadrant_labels, index=universe),
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
                        "Rank Sum":   "{:.1f}",
                    }),
                    use_container_width=True,
                    key="rank_matrix"
                )

                # ── Audit Trail ───────────────────────────────────────
                st.subheader("📋 Audit Trail (Last 15 Trading Days)")
                def color_rets(v):
                    return f'color: {"#00d1b2" if v > 0 else "#ff4b4b"}'
                st.dataframe(
                    audit_df[['Signal', 'Net_Return']].style
                    .map(color_rets, subset=['Net_Return'])
                    .format({"Net_Return": "{:.2%}"}),
                    use_container_width=True,
                    key="audit_trail"
                )

                # ── Equity Curve ──────────────────────────────────────
                if not strat_df.empty:
                    st.subheader("📈 Equity Curve")
                    strat_series = strat_df['Net_Return'].copy()
                    spy_series   = daily_returns['SPY'].loc[strat_df.index].copy()
                    agg_series   = daily_returns['AGG'].loc[strat_df.index].copy()
                    fig = get_equity_curve_fig(strat_series, spy_series, agg_series)
                    st.pyplot(fig, clear_figure=False)

                # ── Methodology ───────────────────────────────────────
                st.divider()
                st.subheader("📖 Methodology")
                fast_days_disp = max(5, training_days // 3)
                if option_label == "Fixed Income":
                    universe_desc = "GLD · SLV · VNQ · TLT · LQD · HYG · VCIT"
                else:
                    universe_desc = "SPY · QQQ · XLV · XLF · XLE · XLI"

                st.markdown(f"""
**Universe ({option_label}):** {universe_desc}

**Objective:** Maximum absolute return via systematic momentum rotation.
One asset (or CASH) is held at a time.

**Ranking — 4 base factors, max score 20:**

| Factor | How it's computed | Max Rank |
|---|---|---|
| **Return Rank** | Total price return over the {training_months}-month training window | 5 |
| **Z-Score Rank** | Cross-sectional z-score of returns | 5 |
| **Momentum Acceleration** | Fast return ({fast_days_disp}d = ⅓ of window) minus slow return | 5 |
| **Volume Confirmation** | Ranks 1–5 only if volume fuel > 1.0× | 5 |

**SDHO Enhancements (Dean 2026):**

| Enhancement | Theory | Effect |
|---|---|---|
| **Φ-calibrated look-backs** | Scaled restoring force Φ varies by asset class | Energy uses shorter windows; bonds use longer windows |
| **Phase quadrant filter** | SDHO flow field: Q2 (x>0, y<0) predicts mean reversion | Rally fading assets receive rank penalty |
| **Ω re-entry cooldown** | Momentum half-life t½ ≈ 1 day (from Ω≈1.15) | Prevents whipsaw after stop-loss trigger |

**Optional Risk Brakes:** Volatility filter · 200-day MA filter · Trailing stop

**Data Source:** [Hugging Face Dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-momentum-maxima) (updated daily via GitHub Actions)
                """)

        except Exception as frag_err:
            st.error(f"❌ Fragment error:\n{traceback.format_exc()}")

    update_dashboard()

except Exception as e:
    st.error(f"❌ An unexpected error occurred:\n{traceback.format_exc()}")
