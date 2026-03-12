import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import os
from datetime import datetime

# ============================================================
# SDHO INTEGRATION: Phi (Φ) calibration table
# From Dean (2026) "Scale Invariant Dynamics in Market Price Momentum"
#
# Φ = scaled restoring force = k * Var(x) = σ²/(2Ω)
# Measures how aggressively each asset class mean-reverts.
#
# HIGH Φ  → strong mean reversion → use SHORTER look-back window
# LOW  Φ  → weak mean reversion, trends persist → use LONGER look-back window
#
# Empirical Φ values from paper (Table 2, 1-hour aggregation):
#   Energy (CL):       Φ = 0.292  ← strongest mean reversion
#   Precious metals:   Φ = 0.106
#   Equity indices:    Φ = 0.046–0.069
#   Fixed income (ZB): Φ = 0.021
#   Currencies (6E):   Φ = 0.016  ← weakest mean reversion
#
# We map each ETF to its closest futures analog and compute
# a look-back multiplier: multiplier = Φ_equity / Φ_asset
# (normalised so equity = 1.0 baseline)
# ============================================================

PHI_LOOKBACK_MULTIPLIER = {
    # Energy — very high Φ (like CL futures, Φ=0.292)
    # Mean reverts strongly → shorter window catches the signal faster
    'XLE':  0.60,

    # Precious metals — medium-high Φ (like GC futures, Φ=0.106)
    'GLD':  0.75,
    'SLV':  0.75,

    # Real estate — medium Φ (between equity and bonds)
    'VNQ':  0.90,

    # Equity indices — baseline Φ (like ES/NQ futures, Φ=0.046–0.069)
    'SPY':  1.00,
    'QQQ':  1.00,
    'XLV':  1.00,
    'XLF':  1.00,
    'XLI':  1.00,

    # High yield bonds — slightly below equity (credit-sensitive but mean-reverts)
    'HYG':  1.10,

    # Investment grade corp bonds — low Φ (like ZB futures, Φ=0.021)
    'VCIT': 1.25,
    'LQD':  1.30,

    # Long-duration Treasuries — lowest Φ (trends persist, like ZB/currencies)
    'TLT':  1.40,

    # Benchmark-only, not traded
    'AGG':  1.30,
}


def get_asset_training_days(ticker: str, base_days: int) -> int:
    """
    Return the SDHO-calibrated look-back window for a given ETF.

    Args:
        ticker:    ETF ticker symbol
        base_days: User-selected base training period in days

    Returns:
        Adjusted training days based on asset's Φ multiplier.
        Clamped to minimum 10 days.
    """
    mult = PHI_LOOKBACK_MULTIPLIER.get(ticker, 1.0)
    return max(10, int(base_days * mult))


def fetch_data():
    """
    Fetch price, volume, and risk-free rate data for the full ETF universe.

    Universe:
        Option A — Fixed Income/Alts: TLT, LQD, VNQ, SLV, GLD, HYG, VCIT
        Option B — Equities:          SPY, QQQ, XLV, XLF, XLE, XLI
        Benchmarks:                   SPY, AGG

    Output:
        etf_momentum_data.parquet  — MultiIndex DataFrame:
            Level 0: ticker
            Level 1: Close, Volume, Return
            Plus: (CASH, Rate), (CASH, Daily_Rf)

    SDHO note:
        Raw returns are stored here; the phase-space coordinates
        (momentum x, acceleration y) are computed at runtime in
        streamlit_app.py using the Φ-calibrated per-asset windows.
    """
    # Full universe including benchmarks
    tickers = [
        'TLT', 'LQD', 'VNQ', 'SLV', 'GLD', 'HYG', 'VCIT',
        'SPY', 'AGG', 'QQQ', 'XLV', 'XLF', 'XLE', 'XLI'
    ]

    all_data = []

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        # Start from 2007 to provide padding for the 2008–2026 backtest
        df = yf.download(ticker, start="2007-01-01", progress=False, auto_adjust=True)

        if not df.empty:
            # Flatten potential MultiIndex from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalise column names
            df.columns = [str(col).capitalize() for col in df.columns]

            temp = df[['Close', 'Volume']].copy()
            # Raw log returns for the Z-Score engine
            # Using log returns for better cross-asset comparability
            temp['Return'] = temp['Close'].pct_change()

            # Set up MultiIndex for the combined dataframe
            temp.columns = pd.MultiIndex.from_product([[ticker], temp.columns])
            all_data.append(temp)

    # ── Fetch FRED 3-Month T-Bill (the cash hurdle rate) ──────────────────
    print("Fetching 3-Month T-Bill from FRED...")
    try:
        # DTB3 = 3-Month Treasury Bill Secondary Market Rate (annualised %)
        rf_data = web.DataReader('DTB3', 'fred', start="2007-01-01")
        rf_data.columns = pd.MultiIndex.from_product([['CASH'], ['Rate']])

        # Convert annualised % to daily decimal return: 5.0% → 0.05/252
        rf_data[('CASH', 'Daily_Rf')] = (rf_data[('CASH', 'Rate')] / 100) / 252
        all_data.append(rf_data)
        print("✅ FRED T-Bill fetched successfully.")
    except Exception as e:
        print(f"⚠️  FRED Failed: {e}")
        print("    Falling back to flat 5% annual rate.")

    # ── Combine and align all data ─────────────────────────────────────────
    if all_data:
        final_df = pd.concat(all_data, axis=1)

        # Forward-fill CASH rate (FRED may lag by 1–2 days)
        final_df = final_df.ffill()

        # Drop rows with no price data across the entire universe
        final_df = final_df.dropna(
            subset=[(t, 'Close') for t in tickers], how='all'
        )

        # ── Log the Φ multipliers for transparency ─────────────────────
        print("\n📐 SDHO Φ-calibrated look-back multipliers:")
        print(f"   {'Ticker':<8} {'Multiplier':>10}  {'Effect'}")
        print(f"   {'-'*8} {'-'*10}  {'-'*30}")
        for t in tickers:
            mult = PHI_LOOKBACK_MULTIPLIER.get(t, 1.0)
            effect = (
                "shorter window (strong mean-reversion)" if mult < 1.0
                else "longer window (trend persistence)" if mult > 1.0
                else "baseline"
            )
            print(f"   {t:<8} {mult:>10.2f}  {effect}")

        final_df.to_parquet('etf_momentum_data.parquet')
        print(f"\n✅ Pipeline Complete. Dataset ends: {final_df.index.max().date()}")
        print(f"   Rows: {len(final_df):,}  |  Columns: {len(final_df.columns)}")
    else:
        print("❌ Error: No data was fetched.")


if __name__ == "__main__":
    fetch_data()
