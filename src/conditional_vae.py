'''
This file contains the code for the conditional VAE.
'''
# Import required libraries
import torch
from torch import nn
from src.vae import kl_to_standard_normal, reconstruction_nll

# Define the conditional VAE class
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes = 10):
        super().__init__()
        # Define the encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim + n_classes, hidden_dim), nn.ReLU())
        # Define the mu head
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        # Define the logvar head
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + n_classes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    # Define the encode method
    def encode(self, x, labels):
        hidden = self.encoder(torch.cat((x, labels), dim = -1))
        return self.mu_head(hidden), self.logvar_head(hidden)
    # Define the decode method
    def decode(self, z, labels):
        return self.decoder(torch.cat((z, labels), dim = -1))
    # Define the forward method
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, labels), mu, logvar

# Define the conditional VAE loss function
def conditional_vae_loss(model, x_batch, label_batch, *, beta = 1.0):
    # Forward pass through the model
    logits, mu, logvar = model(x_batch, label_batch)
    # Calculate the reconstruction loss
    reconstruction = reconstruction_nll(logits, x_batch)
    # Calculate the KL loss
    kl = kl_to_standard_normal(mu, logvar)
    return (reconstruction + beta * kl).mean()

# Define the evaluate conditioning method for the conditional VAE
@torch.no_grad()
def evaluate_conditioning(conditional_model, unconditional_model, feature_net, latent_dim, seed, n_classes = 10, n_per_digit = 200, n_unconditional = 1_000):
    
    # Set the models to evaluation mode
    conditional_model.eval()
    unconditional_model.eval()

    # Create a generator, requested digits, and labels
    generator = torch.Generator().manual_seed(seed)
    requested = torch.arange(n_classes).repeat_interleave(n_per_digit)
    labels = nn.functional.one_hot(requested, num_classes = n_classes).float()

    # Create a tensor of the latent samples
    z = torch.randn(len(requested), latent_dim, generator = generator)
    # Decode the latent samples
    conditional_samples = conditional_model.decode(z, labels).sigmoid()
    # Predict the class of the conditional samples
    predicted = feature_net(conditional_samples).argmax(dim = 1)
    # Calculate the match rates
    match_rates = (predicted == requested).reshape(n_classes, n_per_digit)
    match_rates = match_rates.float().mean(dim = 1)
    # Create a tensor of the latent samples
    z = torch.randn(n_unconditional, latent_dim, generator = generator)
    # Decode the latent samples
    unconditional_samples = unconditional_model.decode(z).sigmoid()
    # Predict the class of the unconditional samples
    predicted = feature_net(unconditional_samples).argmax(dim = 1)
    # Calculate the class counts
    class_counts = predicted.bincount(minlength = n_classes)
    # Return the match rates, class counts, and conditional samples
    return match_rates, class_counts, conditional_samples