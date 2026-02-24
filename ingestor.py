import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import os
import pandas_market_calendars as mcal
from datetime import datetime

def fetch_data():
    # 1. Define Universe
    tickers = ['TLT', 'TBT', 'VNQ', 'SLV', 'GLD', 'SPY', 'AGG']
    all_data = []

    # 2. Integrate NYSE Calendar for Date Validation
    nyse = mcal.get_calendar('NYSE')
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        # Start from 2007 to provide padding for the 2008-2026 backtest
        df = yf.download(ticker, start="2007-01-01", progress=False, auto_adjust=True)
        
        if not df.empty:
            # RECTIFIED: Flatten potential MultiIndex from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure columns are standard strings before slicing
            df.columns = [str(col).capitalize() for col in df.columns]
            
            temp = df[['Close', 'Volume']].copy()
            # Raw returns for the Z-Score engine
            temp['Return'] = temp['Close'].pct_change()
            
            # Set up MultiIndex for the main dataframe
            temp.columns = pd.MultiIndex.from_product([[ticker], temp.columns])
            all_data.append(temp)

    # 3. Fetch FRED 3-Month T-Bill (The Hurdle)
    print("Fetching 3-Month T-Bill from FRED...")
    try:
        # DTB3 is the 3-Month Treasury Bill Secondary Market Rate
        rf_data = web.DataReader('DTB3', 'fred', start="2007-01-01")
        rf_data.columns = pd.MultiIndex.from_product([['CASH'], ['Rate']])
        
        # RECTIFIED: Calculate daily yield for the Audit Trail
        # Annual Rate / 100 / 252 trading days
        rf_data[('CASH', 'Daily_Rf')] = (rf_data[('CASH', 'Rate')] / 100) / 252
        all_data.append(rf_data)
    except Exception as e:
        print(f"⚠️ FRED Failed: {e}")

    # 4. Combine and Rectify Alignment
    if all_data:
        # Join all tickers and CASH data on the date index
        final_df = pd.concat(all_data, axis=1)
        
        # FIX: Forward fill CASH so it doesn't drop latest price days if FRED lags
        final_df = final_df.ffill()
        
        # Drop rows where we have no price data for our universe
        final_df = final_df.dropna(subset=[(t, 'Close') for t in tickers], how='all')

        # Save as Parquet for the Streamlit app
        final_df.to_parquet('etf_momentum_data.parquet')
        print(f"✅ Pipeline Complete: Dataset ends at {final_df.index.max().date()}")
    else:
        print("❌ Error: No data was fetched.")

if __name__ == "__main__":
    fetch_data()
