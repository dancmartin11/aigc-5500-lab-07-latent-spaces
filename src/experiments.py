'''
This file contains the code for the experiments in the lab.
'''

# Import required libraries
from functools import partial
import torch
from scripts import lab07_starter as S
from src.vae import VAE, vae_loss

# Define the function to train the matched models
def train_matched_models(x_train, input_dim, hidden_dim, latent_dim, seed, *, epochs, lr, batch_size):

    # Define the training settings
    training_settings = {'epochs': epochs, 'lr': lr, 'batch_size': batch_size, 'seed': seed}
    torch.manual_seed(seed)

    # Train the autoencoder
    ae = S.AE(d_in = input_dim, d_hidden = hidden_dim, d_latent = latent_dim)
    S.train(ae, S.ae_loss, x_train, **training_settings)

    # Train the VAE
    torch.manual_seed(seed)
    vae = VAE(input_dim = input_dim, hidden_dim = hidden_dim, latent_dim = latent_dim)
    # Define the VAE loss with beta = 1.0
    beta_1_loss = partial(vae_loss, beta = 1.0)
    # Train the VAE
    S.train(vae, beta_1_loss, x_train, **training_settings)

    return ae, vae