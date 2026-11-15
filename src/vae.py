'''
Module containing the VAE implementation
'''

# Import required libraries
import torch
from torch import nn
from torch.nn import functional as F

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential( nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    # Encode the input
    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    # Reparameterize the latent vector
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + std * noise

    # Decode the latent vector
    def decode(self, z):
        return self.decoder(z)

    # Forward pass
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the reconstruction loss function
def reconstruction_nll(logits, x):
    pixel_losses = F.binary_cross_entropy_with_logits(logits, x, reduction = 'none')
    return pixel_losses.sum(dim = 1)

# Define the KL loss function
def kl_to_standard_normal(mu, logvar):
    kl_per_dim = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
    return kl_per_dim.sum(dim = 1)

# Define the VAE loss function
def vae_loss(model, x_batch, _y_batch = None, *, beta = 1.0):
    logits, mu, logvar = model(x_batch)
    reconstruction = reconstruction_nll(logits, x_batch)
    kl = kl_to_standard_normal(mu, logvar)
    return (reconstruction + beta * kl).mean()