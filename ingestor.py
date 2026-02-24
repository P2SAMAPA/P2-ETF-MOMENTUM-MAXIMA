import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import os
from datetime import datetime

def fetch_data():
    # 1. Define Universe (Core ETFs + Benchmarks)
    tickers = ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'SPY', 'AGG']
    all_data = []

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        # Start from 2007 to ensure lookback padding for the 2008-2026 model
        df = yf.download(ticker, start="2007-01-01", progress=False, auto_adjust=True)
        
        if not df.empty:
            # Flatten columns if yfinance returns a MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Explicitly capture Close and Volume for the Momentum Engine
            temp = df[['Close', 'Volume']].copy()
            temp['Return'] = temp['Close'].pct_change()
            
            # Structure as Multi-Index (Ticker, Metric)
            temp.columns = pd.MultiIndex.from_product([[ticker], temp.columns])
            all_data.append(temp)

    # 2. Fetch FRED 3-Month T-Bill (The Absolute Momentum Hurdle)
    print("Fetching 3-Month T-Bill from FRED...")
    try:
        rf_data = web.DataReader('DTB3', 'fred', start="2007-01-01")
        rf_data.columns = pd.MultiIndex.from_product([['CASH'], ['Rate']])
        
        # Convert % rate to daily decimal for cost-adjusted optimization
        rf_data[('CASH', 'Daily_Rf')] = (rf_data[('CASH', 'Rate')] / 100) / 252
        all_data.append(rf_data)
    except Exception as e:
        print(f"⚠️ FRED Failed: {e}")

    # 3. Combine and Save
    if all_data:
        # Join dataframes along the date index
        final_df = pd.concat(all_data, axis=1)
        
        # RECTIFICATION: Forward fill T-Bill rates to align with latest prices
        final_df = final_df.ffill()
        
        # Only drop rows where ETF price data is missing (preserving the latest days)
        final_df = final_df.dropna(subset=[(t, 'Close') for t in tickers], how='all')

        # Save to Parquet for Streamlit consumption
        final_df.to_parquet('etf_momentum_data.parquet')
        print(f"✅ Pipeline Complete: Dataset ends at {final_df.index.max().date()}")
    else:
        print("❌ Error: No data was fetched.")

if __name__ == "__main__":
    fetch_data()
