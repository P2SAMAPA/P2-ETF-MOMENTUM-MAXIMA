#!/usr/bin/env python3
"""
Rebuild etf_momentum_data.parquet with updated tickers and upload to GitLab.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import urllib.parse
from io import BytesIO

# GitLab config
GITLAB_TOKEN = os.getenv("GITLAB_API_TOKEN")
PROJECT_ID = "p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA"
FILE_PATH = "etf_momentum_data.parquet"
BRANCH = "main"

# Updated tickers (matching your streamlit_app.py)
UNIVERSE_FI = ['GLD', 'SLV', 'VNQ', 'TLT', 'LQD', 'HYG', 'VCIT']
UNIVERSE_EQ = ['SPY', 'QQQ', 'XLV', 'XLF', 'XLE', 'XLI']
BENCHMARKS = ['SPY', 'AGG']
ALL_TICKERS = list(dict.fromkeys(UNIVERSE_FI + UNIVERSE_EQ + BENCHMARKS))

def fetch_data():
    """Fetch price and volume data from Yahoo Finance."""
    print(f"⏳ Downloading data for {len(ALL_TICKERS)} tickers...")
    
    # Download price and volume data
    data = yf.download(
        ALL_TICKERS,
        start="2008-01-01",
        progress=False,
        auto_adjust=True
    )
    
    # Fetch risk-free rate (3-month T-bill)
    try:
        import pandas_datareader.data as web
        rf = web.DataReader('DTB3', 'fred', start="2008-01-01")
        rf_daily = rf / 100 / 252  # Convert to daily yield
        rf_daily.columns = ['Daily_Rf']
    except Exception as e:
        print(f"⚠️ Could not fetch T-bill data: {e}")
        # Create dummy cash data
        rf_daily = pd.DataFrame({'Daily_Rf': 0.0001}, index=data.index)
    
    # Build MultiIndex DataFrame
    price_df = data['Close'].copy()
    volume_df = data['Volume'].copy()
    
    # Create MultiIndex columns
    tuples = []
    for ticker in ALL_TICKERS:
        tuples.append((ticker, 'Close'))
        tuples.append((ticker, 'Volume'))
    
    # Combine price and volume
    combined = pd.concat([price_df, volume_df], axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [(t, 'Close') for t in ALL_TICKERS] + [(t, 'Volume') for t in ALL_TICKERS]
    )
    
    # Add CASH column
    combined[('CASH', 'Daily_Rf')] = rf_daily.reindex(combined.index).fillna(method='ffill')
    
    print(f"✅ Data fetched: {combined.shape}")
    print(f"📊 Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"📈 Tickers: {ALL_TICKERS}")
    
    return combined

def upload_to_gitlab(df):
    """Upload parquet file to GitLab repository."""
    print("⏳ Uploading to GitLab...")
    
    # Convert to parquet bytes
    buffer = BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)
    content = buffer.read()
    
    # GitLab API endpoint
    proj_enc = urllib.parse.quote(PROJECT_ID, safe='')
    file_enc = urllib.parse.quote(FILE_PATH, safe='')
    url = f"https://gitlab.com/api/v4/projects/{proj_enc}/repository/files/{file_enc}"
    
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    
    # Check if file exists
    get_url = f"{url}?ref={BRANCH}"
    resp = requests.get(get_url, headers=headers)
    file_exists = resp.status_code == 200
    
    # Prepare content (base64 encode)
    import base64
    content_b64 = base64.b64encode(content).decode('utf-8')
    
    # Create or update
    if file_exists:
        # Update existing file
        data = {
            "branch": BRANCH,
            "content": content_b64,
            "encoding": "base64",
            "commit_message": f"Update dataset: Added VCIT, LQD, HYG; Removed TBT equivalent | {pd.Timestamp.now().strftime('%Y-%m-%d')}"
        }
        resp = requests.put(url, headers=headers, json=data)
    else:
        # Create new file
        data = {
            "branch": BRANCH,
            "content": content_b64,
            "encoding": "base64",
            "commit_message": f"Create dataset with tickers: {ALL_TICKERS}"
        }
        resp = requests.post(url, headers=headers, json=data)
    
    if resp.status_code in [200, 201]:
        print("✅ Upload successful!")
        print(f"🌐 File URL: https://gitlab.com/{PROJECT_ID}/-/blob/{BRANCH}/{FILE_PATH}")
    else:
        print(f"❌ Upload failed: {resp.status_code}")
        print(resp.text)

def main():
    if not GITLAB_TOKEN:
        print("❌ GITLAB_API_TOKEN not set!")
        return
    
    df = fetch_data()
    upload_to_gitlab(df)
    print("🎉 Done!")

if __name__ == "__main__":
    main()
