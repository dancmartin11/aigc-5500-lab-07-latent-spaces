'''
This file contains the code for the interpolation experiment.
'''

# Import required libraries
import torch

# Define the function to sample test pairs
def sample_test_pairs(y_test, n_pairs = 40, seed = 0):
    left, right = torch.triu_indices(len(y_test), len(y_test), offset = 1)
    different_digits = y_test[left] != y_test[right]
    pairs = torch.stack((left[different_digits], right[different_digits]), dim = 1)

    # Sample the test pairs
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(pairs), generator = generator)[:n_pairs]
    return pairs[selected]

# Define the function to evaluate the interpolation
@torch.no_grad()
def evaluate_interpolation(ae, vae, feature_net, x_test, pairs, n_steps = 10):

    # Get the start and end images
    x_start = x_test[pairs[:, 0]]
    x_end = x_test[pairs[:, 1]]
    weights = torch.linspace(0, 1, n_steps).view(1, n_steps, 1)

    # Interpolate the codes
    ae_codes = torch.lerp(ae.encode(x_start)[:, None, :], ae.encode(x_end)[:, None, :], weights)
    # Encode the start and end images
    vae_start, _ = vae.encode(x_start)
    # Encode the end images
    vae_end, _ = vae.encode(x_end)
    # Interpolate the codes
    vae_codes = torch.lerp(vae_start[:, None, :], vae_end[:, None, :], weights)

    # Define the paths
    paths = {
        'AE latent': ae.decode(ae_codes),
        'VAE latent': vae.decode(vae_codes).sigmoid(),
        'Pixel cross-fade': torch.lerp(x_start[:, None, :], x_end[:, None, :], weights)
        }

    # Evaluate the paths
    results = []
    for method, frames in paths.items():
        pixels = frames.flatten(0, 1).clamp(0, 1)
        probabilities = feature_net(pixels).softmax(dim = -1)
        confidence, predicted_class = probabilities.max(dim = -1)
        predicted_class = predicted_class.reshape(len(pairs), n_steps)
        class_changes = (predicted_class[:, 1:] != predicted_class[:, :-1]).sum(dim = 1)
        # Append the results
        results.append({
            'Method': method,
            'Mean confidence': confidence.mean().item(),
            'Mean class changes': class_changes.float().mean().item()
            })

    return results