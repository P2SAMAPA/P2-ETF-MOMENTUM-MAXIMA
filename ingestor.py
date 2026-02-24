import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import os
from datetime import datetime

def fetch_data():
    # 1. Define Universe
    tickers = ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'SPY', 'AGG']
    all_data = []

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        # Start from 2007 to ensure the 18-month lookback works for the 2008 start
        df = yf.download(ticker, start="2007-01-01", progress=False, auto_adjust=True)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            temp = df[['Close', 'Volume']].copy()
            # Ensure we have the raw Return for the Z-Score engine
            temp['Return'] = temp['Close'].pct_change()
            
            temp.columns = pd.MultiIndex.from_product([[ticker], temp.columns])
            all_data.append(temp)

    # 2. Fetch FRED 3-Month T-Bill (The Hurdle Rate)
    print("Fetching 3-Month T-Bill from FRED...")
    try:
        rf_data = web.DataReader('DTB3', 'fred', start="2007-01-01")
        rf_data.columns = pd.MultiIndex.from_product([['CASH'], ['Rate']])
        # Daily RF used for the Absolute Momentum Filter
        rf_data[('CASH', 'Daily_Rf')] = (rf_data[('CASH', 'Rate')] / 100) / 252
        all_data.append(rf_data)
    except Exception as e:
        print(f"⚠️ FRED Failed: {e}")

    # 3. Combine and Save
    if all_data:
        # We join on the index of the price data to prevent dropping recent days
        final_df = pd.concat(all_data, axis=1)
        
        # FIX: Forward fill the CASH rate so it doesn't kill the latest price rows
        final_df = final_df.ffill()
        
        # Only drop rows where we have NO price data at all
        final_df = final_df.dropna(subset=[(t, 'Close') for t in tickers], how='all')

        final_df.to_parquet('etf_momentum_data.parquet')
        print(f"✅ Pipeline Complete: Dataset ends at {final_df.index.max().date()}")
    else:
        print("❌ Error: No data was fetched.")

if __name__ == "__main__":
    fetch_data()
