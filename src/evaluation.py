
'''
This file contains the code for the evaluation of the matched models.
'''

# Import required libraries
import torch
from scripts import lab07_starter as S

# Define the function to evaluate the matched models
@torch.no_grad()
def evaluate_matched_models(ae, vae, x_train, x_test, feature_net, test_features, seed):
    ae.eval()
    vae.eval()

    # Use identical prior draws for both decoders
    test_means, _ = vae.encode(x_test)
    generator = torch.Generator().manual_seed(seed)
    prior_codes = torch.randn(test_means.shape, generator = generator)

    # Encode the training images
    vae_train_means, _ = vae.encode(x_train)
    model_outputs = (
        ('AE', ae(x_test), ae.decode(prior_codes), ae.encode(x_train)),
        ('VAE', vae.decode(test_means).sigmoid(), vae.decode(prior_codes).sigmoid(), vae_train_means)
        )

    # Evaluate the models
    results = {}
    for name, reconstruction, samples, train_codes in model_outputs:
        results[name] = {
            'test_mse': (reconstruction - x_test).square().mean().item(),
            'code_std': train_codes.std(dim = 0).tolist(),
            'samples': samples,
            **S.score_samples(samples, test_features, feature_net, x_train),
        }

    return results