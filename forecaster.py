import torch
import torch.nn as nn

class VolatilitySurfaceForecaster(nn.Module):
    def __init__(self, grid_size=75, latent_dim=3, hidden_dim=32, num_layers=1):
        super(VolatilitySurfaceForecaster, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 1. THE ENCODER: Compresses the 2D surface grid into a low-dimensional latent vector
        # (Captures structural features like parallel shifts, skew changes, or term structure twists)
        self.encoder = nn.Sequential(
            nn.Linear(grid_size, 16),
            nn.Tanh(),
            nn.Linear(16, latent_dim)
        )
        
        # 2. THE DECODER: Reconstructs the compressed vector back into a full 75-point grid surface
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, grid_size),
            nn.Softplus() # Ensures reconstructed variance values remain positive
        )
        
        # 3. THE TIME-SERIES ENGINE (LSTM): Learns historical trajectories of the latent factors
        self.lstm = nn.ModuleList([
            nn.LSTMCell(input_size=latent_dim, hidden_size=hidden_dim)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 4. FORECAST PROJECTION: Maps LSTM hidden state to *tomorrow's* predicted latent factors
        self.forecast_head = nn.Linear(hidden_dim, latent_dim)
        
    def encode_surface(self, surface_grid):
        """Compresses a batch of surface frames into their latent coordinates."""
        return self.encoder(surface_grid)
        
    def decode_latent(self, latent_vectors):
        """Expands latent coordinates back into full 2D surface grids."""
        return self.decoder(latent_vectors)
        
    def forward(self, sequence):
        """
        Input shape: (batch, sequence_length, grid_size)
        Outputs:
            - reconstructed_seq: Autoencoder reconstructions for logging
            - next_day_surface: Predicted surface grid for tomorrow
        """
        batch_size, seq_len, grid_size = sequence.size()
        
        # Step A: Pass every historical day through the encoder to get historical latent paths
        # Flatten time and batch dimensions together for efficiency
        flat_seq = sequence.view(-1, grid_size)
        flat_latent = self.encode_surface(flat_seq)
        latent_seq = flat_latent.view(batch_size, seq_len, self.latent_dim)
        
        # Step B: Get autoencoder reconstructions to measure structural loss
        flat_recon = self.decode_latent(flat_latent)
        reconstructed_seq = flat_recon.view(batch_size, seq_len, grid_size)
        
        # Step C: Feed the path of latent points sequentially into the LSTM cells
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, device=sequence.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=sequence.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            x_t = latent_seq[:, t, :] # Latent vector for day 't'
            
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.lstm[layer](x_t, (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.lstm[layer](h[layer-1], (h[layer], c[layer]))
                    
        # Step D: Use final hidden state to project tomorrow's latent state
        next_latent = self.forecast_head(h[-1])
        
        # Step E: Decode tomorrow's predicted latent factor back to a full 75-point surface grid
        next_day_surface = self.decode_latent(next_latent)
        
        return reconstructed_seq, next_day_surface