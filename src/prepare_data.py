import pandas as pd
import numpy as np
import os
import sys
import json
import scipy.optimize as sco

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_prep import all_sectors
from src.fetch_names import fetch_names

data_dir = os.path.join(project_root, 'data', 'processed')
raw_dir = os.path.join(project_root, 'data', 'raw')

# Ensure directories exist
os.makedirs(data_dir, exist_ok=True)

def prepare_all_data():
    print("=== Starting Data Preparation Pipeline ===")
    
    # Load and Clean Data (Generates all_sectors.csv and sector_map.json)
    print("\n[1/4] Processing raw sector data...")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print("Error: No raw data found in data/raw. Cannot proceed.")
        # If files exist in processed, we might survive, otherwise fail
        if not os.path.exists(os.path.join(data_dir, 'daily_returns.csv')):
             sys.exit(1)
    else:
        df = all_sectors()
        if df is None:
            print("Failed to load sectors.")
            sys.exit(1)

        # Calculate Returns (Generates daily_returns.csv)
        print("\n[2/4] Calculating daily returns...")
        returns = df.pct_change()
        returns.drop(returns.index[0], inplace=True)
        
        # Clean columns with NaNs
        returns_clean = returns.dropna(axis=1)
        print(f"Cleaned returns shape: {returns_clean.shape}")
        
        returns_path = os.path.join(data_dir, 'daily_returns.csv')
        returns_clean.to_csv(returns_path)
        print(f"Saved: {returns_path}")

        # Markowitz Optimization (Generates optimal_weights_level1.csv)
        print("\n[3/4] Running Level 1 Markowitz Optimization...")
        run_optimization(returns_clean)

    # Fetch Ticker Names (Generates ticker_names.json)
    print("\n[4/4] Checking/Fetching ticker names...")
    # This checks if file exists inside the function, safe to call
    fetch_names()
    
    print("\n=== Data Preparation Complete ===")

def run_optimization(df):
    # Annualized stats
    mu = df.mean() * 252
    sigma = df.cov() * 252
    
    num_assets = len(mu)
    
    # Constraints
    constraints_base = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    w0 = np.ones(num_assets) / num_assets
    
    # Objective: Minimize Negative Sharpe
    def min_neg_sharpe(w, mu, sigma):
        rend = w @ mu
        vol = np.sqrt(w.T @ sigma @ w)
        if vol == 0: return 0
        return -rend / vol

    print("Optimizing portfolio (Max Sharpe)... this may take a moment.")
    res_sharpe = sco.minimize(
        min_neg_sharpe, 
        w0, 
        args=(mu, sigma), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints_base
    )
    
    w_sharpe = res_sharpe.x
    
    # Save weights
    w_sharpe_df = pd.DataFrame({'Ticker': df.columns, 'Weight': w_sharpe})
    
    # Filter small weights for cleanliness
    w_sharpe_df = w_sharpe_df[w_sharpe_df['Weight'] > 1e-5]
    
    output_path = os.path.join(data_dir, 'optimal_weights_level1.csv')
    w_sharpe_df.to_csv(output_path, header=False, index=False)
    print(f"Saved optimal weights to: {output_path}")

if __name__ == "__main__":
    prepare_all_data()
