import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import Rbf

def fetch_historical_surface_sequence(ticker_symbol="AAPL", lookback_days=10):
    print(f"Fetching historical data for {ticker_symbol} over the last {lookback_days} days...")
    ticker = yf.Ticker(ticker_symbol)
    
    # Get historical stock prices for the lookback window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2) # Pull extra days to ensure we get enough trading days
    history_df = ticker.history(start=start_date, end=end_date)
    
    trading_days = history_df.index[-lookback_days:]
    print(f"Processing {len(trading_days)} trading days...")
    
    # Define a standardized reference grid for the LSTM to read
    # 15 strike points x 5 maturity points = 75 features per day
    grid_k = np.linspace(-0.2, 0.2, 15)
    grid_T = np.linspace(0.1, 1.5, 5)
    K_mesh, T_mesh = np.meshgrid(grid_k, grid_T)
    
    sequence_data = []
    
    # Fetch live options chain to simulate historical surface deforming
    try:
        expirations = ticker.options[:5]
        all_options = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            opt_chain = ticker.option_chain(exp_str)
            calls = opt_chain.calls[['strike', 'impliedVolatility', 'volume']].copy()
            puts = opt_chain.puts[['strike', 'impliedVolatility', 'volume']].copy()
            chain = pd.concat([calls, puts])
            chain = chain[(chain['volume'] > 5) & (chain['impliedVolatility'] > 0.01)]
            chain['exp_date'] = exp_date
            all_options.append(chain)
        df_base = pd.concat(all_options)
    except Exception as e:
        raise ValueError(f"Failed to fetch baseline options chain: {e}")

    for date in trading_days:
        S0 = history_df.loc[date, 'Close']
        date_naive = date.to_pydatetime().replace(tzinfo=None)
        
        # Recalculate k and T for this specific historical day's snapshot
        df_day = df_base.copy()
        
        # FIXED: Added .dt to access vectorized days from the Timedelta Series
        df_day['T'] = (df_day['exp_date'] - date_naive).dt.days / 365.25
        
        df_day = df_day[df_day['T'] > 0.05] # Drop expiring options
        df_day['k'] = np.log(df_day['strike'] / S0)
        df_day = df_day[(df_day['k'] >= -0.25) & (df_day['k'] <= 0.25)]
        
        if len(df_day) < 10:
            continue
            
        # Standardize the messy market points onto our fixed grid using Radial Basis Functions (RBF)
        try:
            rbf = Rbf(df_day['k'].values, df_day['T'].values, df_day['impliedVolatility'].values, function='thin_plate', smooth=0.01)
            grid_vol = rbf(K_mesh, T_mesh)
            grid_vol = np.clip(grid_vol, 0.05, 1.0) # Keep within realistic boundaries
            
            # Total Variance: w = sigma^2 * T
            grid_w = (grid_vol ** 2) * T_mesh
            sequence_data.append(grid_w.flatten())
        except:
            continue
            
    if len(sequence_data) < 3:
        raise ValueError("Not enough historical data frames could be generated.")
        
    # Shape: (Num_Days, Grid_Size) -> e.g., (10, 75)
    sequence_tensor = torch.tensor(np.array(sequence_data), dtype=torch.float32)
    print(f"Successfully constructed historical sequence tensor with shape: {sequence_tensor.shape}")
    
    return sequence_tensor, torch.tensor(K_mesh.flatten(), dtype=torch.float32), torch.tensor(T_mesh.flatten(), dtype=torch.float32), S0