import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our modular toolkit
from data_loader import fetch_historical_surface_sequence
from forecaster import VolatilitySurfaceForecaster
from loss import grid_pinn_loss

# Config
TICKER = "AAPL"
LOOKBACK_DAYS = 15  # Increased lookback for a richer time-series memory

# 1. Load the Historical Sequence
sequence_data, k_grid, T_grid, S0 = fetch_historical_surface_sequence(TICKER, lookback_days=LOOKBACK_DAYS)

# --- THE PRODUCTION SPLIT ---
# We train the LSTM using days up to the second-to-last day, 
# and hold out the absolute final day as our "Unseen Future Test Day"
train_sequence = sequence_data[:-2].unsqueeze(0)   # History for training
train_target = sequence_data[-2].unsqueeze(0)     # Target for training

test_sequence = sequence_data[:-1].unsqueeze(0)    # History up to today
test_target_tomorrow = sequence_data[-1].unsqueeze(0) # The True Unseen Future (Tomorrow)

# 2. Instantiate Model and Optimizer
model = VolatilitySurfaceForecaster(grid_size=75, latent_dim=3, hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("\n[Phase 1] Training on Historical Split...")
for epoch in range(601):
    optimizer.zero_grad()
    
    reconstructed_seq, forecasted_surface = model(train_sequence)
    loss_reconstruction = nn.MSELoss()(reconstructed_seq, train_sequence)
    
    total_forecast_loss, _, _, _ = grid_pinn_loss(
        forecasted_surface=forecasted_surface,
        target_tomorrow=train_target,
        num_k=15, num_t=5, lambda_cal=15.0, lambda_but=15.0
    )
    
    total_loss = loss_reconstruction + total_forecast_loss
    total_loss.backward()
    optimizer.step()

print("Training complete.")

# --- THE TRUE OUT-OF-SAMPLE TEST ---
print("\n[Phase 2] Evaluating on COMPLETELY UNSEEN Future Market Data...")
model.eval() # Put model in evaluation mode
with torch.no_grad():
    _, true_future_forecast = model(test_sequence)
    
    # Calculate Out-of-Sample Performance Metrics
    test_mse = nn.MSELoss()(true_future_forecast, test_target_tomorrow).item()
    test_rmse = np.sqrt(test_mse)
    
    tomorrow_pred_w = true_future_forecast.squeeze().numpy()
    tomorrow_actual_w = test_target_tomorrow.squeeze().numpy()
    
    # Convert back to standard implied volatility scale
    vol_pred = np.sqrt(tomorrow_pred_w / T_grid.numpy())
    vol_actual = np.sqrt(tomorrow_actual_w / T_grid.numpy())

print(f"\n=== OUT-OF-SAMPLE PERFORMANCE ===")
print(f"True Future Forecast RMSE: {test_rmse:.6f}")
print("==================================\n")

# 3. Generate the Out-of-Sample 3D Plot
K_mesh = k_grid.numpy().reshape(5, 15)
T_mesh = T_grid.numpy().reshape(5, 15)
Vol_pred_mesh = vol_pred.reshape(5, 15)
Vol_actual_mesh = vol_actual.reshape(5, 15)
Strike_mesh = S0 * np.exp(K_mesh)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Strike_mesh, T_mesh, Vol_pred_mesh, cmap='viridis', edgecolor='none', alpha=0.7, label='PINN Out-of-Sample Forecast')
wire = ax.plot_wireframe(Strike_mesh, T_mesh, Vol_actual_mesh, color='red', linewidth=0.8, alpha=0.5, label='Actual Unseen Realization')

ax.set_title(f"TRUE OUT-OF-SAMPLE FORECAST: Predicted vs Actual Volatility Surface ({TICKER})")
ax.set_xlabel("Strike Price ($)")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Implied Volatility (sigma)")

#surf._edgecolors2d = surf._facecolors2d
#wire._edgecolors2d = wire._edgecolors2d
ax.legend()

plt.show()