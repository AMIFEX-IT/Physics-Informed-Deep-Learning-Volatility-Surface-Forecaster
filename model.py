import torch
import torch.nn as nn

class VolatilityPINN(nn.Module):
    def __init__(self):
        super(VolatilityPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Softplus(),  # Smooth activation allowing continuous derivatives
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
            nn.Softplus()   # Outputs total variance (w), strictly positive
        )
        
    def forward(self, k, T):
        inputs = torch.cat([k, T], dim=1)
        return self.net(inputs)