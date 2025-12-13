import json
import os
import yfinance as yf
import time
from tqdm import tqdm

base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_path)
data_path = os.path.join(project_root, 'data', 'processed')
sector_path = os.path.join(data_path, 'sector_map.json')
output_path = os.path.join(data_path, 'ticker_names.json')

def fetch_names():
    if not os.path.exists(sector_path):
        print("Sector map not found.")
        return

    with open(sector_path, 'r') as f:
        sector_map = json.load(f)
        
    tickers = list(sector_map.keys())
    print(f"Fetching names for {len(tickers)} tickers...")
    
    ticker_names = {}
    
    # Check if we already have some names to avoid refetching everything if it failed
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                ticker_names = json.load(f)
        except:
            pass
            
    tickers_to_fetch = [t for t in tickers if t not in ticker_names]
    
    # Fetch in chunks using yfinance Tickers object for better performance if possible, 
    # but Tickers object might not give details for all easily. Let's try individual for safety/simplicity
    # yfinance often works better with space separated string for multiple tickers
    
    # Let's try batching in groups of 10
    batch_size = 10
    
    for i in tqdm(range(0, len(tickers_to_fetch), batch_size)):
        batch = tickers_to_fetch[i:i+batch_size]
        batch_str = " ".join(batch)
        
        try:
            tickers_data = yf.Tickers(batch_str)
            
            for ticker in batch:
                try:
                    info = tickers_data.tickers[ticker].info
                    name = info.get('longName') or info.get('shortName') or ticker
                    ticker_names[ticker] = name
                except Exception as e:
                    print(f"\nCould not fetch for {ticker}: {e}")
                    ticker_names[ticker] = ticker
                    
        except Exception as e:
            print(f"Batch error: {e}")
            
        # Save progress periodically
        with open(output_path, 'w') as f:
            json.dump(ticker_names, f, indent=4)
            
    print("Done!")

if __name__ == "__main__":
    fetch_names()
