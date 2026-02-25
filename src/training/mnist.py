import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")
# 1. Override the broken official mirrors with a reliable AWS mirror
MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

# Load the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. Define the VAE Architecture
# ==========================================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ENCODER: Compresses 28x28 image down to hidden layers
        self.enc_fc1 = nn.Linear(28 * 28, 256)
        self.enc_fc2 = nn.Linear(256, 128)
        
        # Latent Space (2D): Mean and Log-Variance
        self.fc_mu = nn.Linear(128, 2)
        self.fc_logvar = nn.Linear(128, 2)
        
        # DECODER: Matches the SimpleDecoder from your Streamlit app
        self.dec_fc1 = nn.Linear(2, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_fc3 = nn.Linear(256, 28 * 28)

    def encode(self, x):
        h = torch.relu(self.enc_fc1(x))
        h = torch.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        # Use sigmoid to ensure pixel values are between 0 and 1
        return torch.sigmoid(self.dec_fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

# ==========================================
# 3. Loss Function & Optimizer
# ==========================================
def vae_loss(reconstructed_x, x, mu, logvar):
    # 1. Reconstruction Loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(reconstructed_x, x.view(-1, 28 * 28), reduction='sum')
    
    # 2. KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. Training Loop
# ==========================================
model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        
        optimizer.zero_grad()
        reconstructed_batch, mu, logvar = model(data)
        
        # Calculate loss and backpropagate
        loss = vae_loss(reconstructed_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

# ==========================================
# 5. Extract and Save Decoder Weights
# ==========================================
# We only want to save the decoder parts to load into the Streamlit app
decoder_state_dict = {
    'fc1.weight': model.dec_fc1.weight, 'fc1.bias': model.dec_fc1.bias,
    'fc2.weight': model.dec_fc2.weight, 'fc2.bias': model.dec_fc2.bias,
    'fc3.weight': model.dec_fc3.weight, 'fc3.bias': model.dec_fc3.bias,
}

torch.save(decoder_state_dict, 'decoder_weights.pth')
print("\n✅ Training complete! Saved 'decoder_weights.pth'.")
