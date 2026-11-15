'''
This file contains the code for the KL-weight sweep experiment.
'''

# Import required libraries
from functools import partial
import torch
from scripts import lab07_starter as S
from src.vae import VAE, kl_to_standard_normal, reconstruction_nll, vae_loss

# Define the function to train the sweep VAE
def train_sweep_vae(x_train, input_dim, hidden_dim, latent_dim, beta, seed, *, epochs, lr, batch_size):
    # Set the seed
    torch.manual_seed(seed)
    # Define the VAE model
    model = VAE(input_dim = input_dim, hidden_dim = hidden_dim, latent_dim = latent_dim)
    # Define the loss function
    loss_fn = partial(vae_loss, beta = beta)
    # Train the VAE
    S.train(model, loss_fn, x_train, epochs = epochs, lr = lr, batch_size = batch_size, seed = seed)
    return model

# Define the function to evaluate the sweep VAE
@torch.no_grad()
def evaluate_sweep_vae(model, x_train, x_test, feature_net, test_features, seed):
    model.eval()

    # Use the same evaluation seed across beta values within each training seed
    generator = torch.Generator().manual_seed(seed)
    mu, logvar = model.encode(x_test)
    noise = torch.randn(mu.shape, generator = generator)
    z = mu + torch.exp(0.5 * logvar) * noise
    reconstruction = reconstruction_nll(model.decode(z), x_test).mean().item()
    kl = kl_to_standard_normal(mu, logvar).mean().item()

    train_mu, _ = model.encode(x_train)
    active_units = (train_mu.var(dim = 0) > 0.01).sum().item()

    # Match the prior draws used in Section 3 for this seed
    prior_generator = torch.Generator().manual_seed(seed)
    prior_codes = torch.randn(mu.shape, generator = prior_generator)
    samples = model.decode(prior_codes).sigmoid()

    return {
        'reconstruction': reconstruction,
        'kl': kl,
        'active_units': active_units,
        'samples': samples,
        **S.score_samples(samples, test_features, feature_net, x_train)
        }