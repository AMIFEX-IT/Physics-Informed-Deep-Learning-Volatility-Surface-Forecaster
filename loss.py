import torch
import torch.nn as nn

def grid_pinn_loss(forecasted_surface, target_tomorrow, num_k=15, num_t=5, lambda_cal=10.0, lambda_but=10.0):
    """
    Computes data MSE and physics penalties directly on a structured grid 
    using finite differences.
    """
    # Data Fit Loss
    loss_data = nn.MSELoss()(forecasted_surface, target_tomorrow)
    
    # Reshape flat grid to 2D matrix: [Batch(1), Rows(Maturities), Cols(Log-Strikes)]
    W = forecasted_surface.view(1, num_t, num_k)
    
    # Grid stepsizes (based on data_loader np.linspace boundaries)
    dk = 0.4 / (num_k - 1)  # range from -0.2 to 0.2
    dT = 1.4 / (num_t - 1)  # range from 0.1 to 1.5
    
    # 1. Calendar Arbitrage: dW/dT >= 0 (Moving down rows)
    # Forward difference along the T dimension (axis 1)
    dW_dT = (W[:, 1:, :] - W[:, :-1, :]) / dT
    loss_calendar = torch.mean(torch.relu(-dW_dT)**2)
    
    # 2. Butterfly Arbitrage: Numerical Durrleman Condition
    # First and second derivatives along the Strike dimension (axis 2)
    dW_dk = (W[:, :, 2:] - W[:, :, :-2]) / (2.0 * dk) # Centered difference
    dW_dkk = (W[:, :, 2:] - 2.0 * W[:, :, 1:-1] + W[:, :, :-2]) / (dk**2)
    
    # Crop the original surface to match the interior grid dimensions [:, :, 1:-1]
    W_interior = W[:, :, 1:-1]
    
    # Set up standard grid arrays for k evaluation matching the interior
    k_grid = torch.linspace(-0.2, 0.2, num_k, device=forecasted_surface.device)
    k_interior = k_grid[1:-1].view(1, 1, -1)
    
    # Exact Durrleman equation applied over the mesh pixels
    term1 = (1.0 - (k_interior * dW_dk) / (2.0 * W_interior))**2
    term2 = (dW_dk**2 / 4.0) * ((1.0 / W_interior) + 0.25)
    term3 = 0.5 * dW_dkk
    g_k = term1 - term2 + term3
    
    loss_butterfly = torch.mean(torch.relu(-g_k)**2)
    
    total_loss = loss_data + (lambda_cal * loss_calendar) + (lambda_but * loss_butterfly)
    
    return total_loss, loss_data, loss_calendar, loss_butterfly