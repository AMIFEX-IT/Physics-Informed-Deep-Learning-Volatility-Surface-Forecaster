import torch
import torch.nn as nn  
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our custom modules
from data_loader import fetch_real_market_data
from model import VolatilityPINN
from loss import exact_pinn_loss

# Config
TICKER = "AAPL"

# 1. Load live market data
k_train, T_train, w_train, S0 = fetch_real_market_data(TICKER)

# 2. Instantiate Model and Optimizer
model = VolatilityPINN()
optimizer = optim.Adam(model.parameters(), lr=0.005)

history = []

# 3. Training Loop
print("\nTraining PINN on Real Market Data...")
for epoch in range(1001):
    optimizer.zero_grad()
    
    loss, l_data, l_cal, l_but = exact_pinn_loss(model, k_train, T_train, w_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        history.append([epoch, loss.item(), l_data.item(), l_cal.item(), l_but.item()])

# 4. Display Summary Table
df_history = pd.DataFrame(history, columns=["Epoch", "Total Loss", "Market Data Error", "Calendar Penalty", "Butterfly Penalty"])
print("\n=== LIVE TRAINING PROGRESS SUMMARY ===")
print(df_history.to_string(index=False, formatters={
    "Total Loss": "{:.6f}".format,
    "Market Data Error": "{:.6f}".format,
    "Calendar Penalty": "{:.6f}".format,
    "Butterfly Penalty": "{:.6f}".format
}))
print("======================================\n")

# 5. Visualizations (Bug Fixed)
print("Generating 3D Real Volatility Surface Plot...")
k_grid = np.linspace(k_train.min().item(), k_train.max().item(), 100)
T_grid = np.linspace(T_train.min().item(), T_train.max().item(), 100)
K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)

k_flat = torch.tensor(K_mesh.flatten()[:, None], dtype=torch.float32)
T_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32)

with torch.no_grad():
    w_pred_flat = model(k_flat, T_flat).numpy()
    vol_pred_flat = np.sqrt(w_pred_flat / T_mesh.flatten()[:, None])

Vol_mesh = vol_pred_flat.reshape(K_mesh.shape)
Strike_mesh = S0 * np.exp(K_mesh)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Strike_mesh, T_mesh, Vol_mesh, cmap='plasma', edgecolor='none', alpha=0.8)

# Overlay raw data points
with torch.no_grad():
    raw_strikes = S0 * np.exp(k_train.numpy())
    raw_vols = np.sqrt(w_train.numpy() / T_train.numpy())
    ax.scatter(raw_strikes, T_train.numpy(), raw_vols, color='black', s=2, alpha=0.5, label='Raw Market Data')

ax.set_title(f"Live Arbitrage-Free Implied Volatility Surface ({TICKER})")
ax.set_xlabel("Strike Price ($)")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Implied Volatility (sigma)")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()