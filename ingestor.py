import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import os
from datetime import datetime

def fetch_data():
    # 1. Define Universe (GLD, SLV, VNQ, TLT, TBT + Benchmarks)
    tickers = ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'SPY', 'AGG']
    
    all_data = []

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        # auto_adjust=True handles splits/dividends into the Close price
        df = yf.download(ticker, progress=False, auto_adjust=True)
        
        if not df.empty:
            # Flatten MultiIndex if yfinance returns one
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Explicitly keep Close and Volume
            temp = df[['Close', 'Volume']].copy()
            temp['Return'] = temp['Close'].pct_change()
            
            # Create Multi-Index (Ticker, Metric)
            temp.columns = pd.MultiIndex.from_product([[ticker], temp.columns])
            all_data.append(temp)

    # 2. Fetch FRED 3-Month T-Bill (Risk-Free Rate)
    print("Fetching 3-Month T-Bill from FRED...")
    try:
        rf_data = web.DataReader('DTB3', 'fred')
        rf_data.columns = pd.MultiIndex.from_product([['CASH'], ['Rate']])
        # Convert % rate to daily decimal for calculations
        rf_data[('CASH', 'Daily_Rf')] = (rf_data[('CASH', 'Rate')] / 100) / 252
        all_data.append(rf_data)
    except Exception as e:
        print(f"FRED Failed: {e}")

    # 3. Combine and Save
    if all_data:
        final_df = pd.concat(all_data, axis=1)
        final_df = final_df.ffill().dropna()

        # Save to Parquet for the Streamlit App to consume
        final_df.to_parquet('etf_momentum_data.parquet')
        print(f"Pipeline Complete: Data stored for {tickers} + CASH")
    else:
        print("Error: No data was fetched.")

if __name__ == "__main__":
    fetch_data()
