import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    """
    Decoder for 28x28 grayscale images (e.g., MNIST).
    Sizes are parameterized to allow experimentation, but defaults 
    (128, 256) are matched with the standard MNIST training script.
    """
    def __init__(self, latent_dim=2, hidden_dim1=128, hidden_dim2=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 28 * 28)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, 28, 28)

class ConvolutionalDecoder(nn.Module):
    """Decoder for 256x256 RGB images (e.g., Dog dataset)."""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.decoder_input = nn.Linear(latent_dim, 65536)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # -> 128x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> 3x256x256
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 16, 16)
        x = self.decoder(h)
        # Returns (B, 3, 256, 256), we want (B, 256, 256, 3) for plotting
        return x.permute(0, 2, 3, 1)
