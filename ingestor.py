#!/usr/bin/env python3
"""
Daily incremental update for etf_momentum_data.parquet on Hugging Face.
Fetches only new data since last update and appends to existing dataset.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, HfApi
from io import BytesIO

REPO_ID = "P2SAMAPA/p2-etf-momentum-maxima"
FILENAME = "etf_momentum_data.parquet"

# Full universe - REMOVED XNT (delisted)
UNIVERSE_FI = ['GLD', 'SLV', 'VNQ', 'TLT', 'LQD', 'HYG', 'VCIT']
UNIVERSE_EQ = ['SPY', 'QQQ', 'XLV', 'XLF', 'XLE', 'XLI', 'XLK', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU', 'XLC', 'XBI', 'XME', 'XHB', 'XSD', 'XRT', 'XAR', 'XNTK']
BENCHMARKS = ['SPY', 'AGG']
ALL_TICKERS = list(dict.fromkeys(UNIVERSE_FI + UNIVERSE_EQ + BENCHMARKS))


def fetch_incremental_data(start_date: datetime):
    """Fetch only new data from start_date to today."""
    if start_date > datetime.now():
        return None, None
        
    print(f"⏳ Fetching incremental data from {start_date.date()}...")
    
    # Download price and volume
    data = yf.download(
        ALL_TICKERS,
        start=start_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True
    )
    
    if data.empty:
        return None, None
    
    # Fetch FRED data for the same period (with buffer)
    try:
        rf_recent = web.DataReader('DTB3', 'fred', start=start_date - timedelta(days=5))
        rf_daily = rf_recent / 100 / 252
        rf_daily.columns = ['Daily_Rf']
    except Exception as e:
        print(f"⚠️ FRED failed: {e}")
        rf_daily = pd.DataFrame({'Daily_Rf': 0.0001}, index=data.index)
    
    return data, rf_daily


def build_dataframe(price_vol_data, rf_data):
    """Build the exact same MultiIndex structure."""
    price_df = price_vol_data['Close'].copy()
    volume_df = price_vol_data['Volume'].copy()
    
    combined = pd.concat([price_df, volume_df], axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [(t, 'Close') for t in ALL_TICKERS] + [(t, 'Volume') for t in ALL_TICKERS]
    )
    
    combined[('CASH', 'Daily_Rf')] = rf_data.reindex(combined.index).ffill()
    
    return combined


def update_dataset():
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # 1. Download existing dataset
    print("⏳ Downloading existing HF dataset...")
    existing_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    df_existing = pd.read_parquet(existing_path)
    last_date = df_existing.index.max()
    print(f"📊 Existing data ends: {last_date.date()}")
    
    # 2. Determine new start date (last_date + 1 day)
    start_date = last_date + timedelta(days=1)
    
    # Skip weekends
    while start_date.weekday() >= 5:
        start_date += timedelta(days=1)
    
    # Check if we actually need to update
    if start_date.date() > datetime.now().date():
        print("✅ Dataset is already up to date")
        return
    
    # 3. Fetch new data
    new_data, new_rf = fetch_incremental_data(start_date)
    if new_data is None or new_data.empty:
        print("⚠️ No new market data available (markets closed or data delayed)")
        return
    
    df_new = build_dataframe(new_data, new_rf)
    print(f"📈 New data shape: {df_new.shape}")
    
    # 4. Combine (handle any overlap/duplicates)
    df_combined = pd.concat([df_existing, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined = df_combined.sort_index()
    
    # 5. Upload back to HF
    print("⏳ Uploading to Hugging Face...")
    buffer = BytesIO()
    df_combined.to_parquet(buffer, index=True)
    buffer.seek(0)
    
    api.upload_file(
        path_or_fileobj=buffer.getvalue(),
        path_in_repo=FILENAME,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Daily update: {start_date.date()} to {df_new.index.max().date()}"
    )
    
    print(f"✅ Success! Dataset now covers: {df_combined.index.min().date()} to {df_combined.index.max().date()}")


if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        print("❌ HF_TOKEN not set!")
        exit(1)
    update_dataset()
