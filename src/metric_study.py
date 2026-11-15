'''
This script is used to score the sample sets.
'''

# Import required libraries
import torch
from scripts import lab07_starter as S

# Define the function to score the sample sets
def score_sample_sets(x_train, feature_net, test_features, matched, collapsed, seed):
    generator = torch.Generator().manual_seed(seed)
    n_samples = len(test_features)

    # Use the same training images for the real and noisy-copy baselines
    indices = torch.randperm(len(x_train), generator = generator)[:n_samples]
    real_images = x_train[indices]
    copy_noise = torch.randn(real_images.shape, generator = generator)
    mean_noise = torch.randn(real_images.shape, generator = generator)
    mean_image = x_train.mean(dim = 0, keepdim = True)

    # Define the sample sets
    sample_sets = {
        'Real training images': real_images,
        'Noisy training copies': real_images + 0.02 * copy_noise,
        'Noisy mean image': mean_image + 0.1 * mean_noise,
        'Collapsed VAE (beta = 4)': collapsed['samples'],
        'VAE (beta = 1)': matched['VAE']['samples'],
        'Plain AE': matched['AE']['samples']
        }

    # Return the scores
    return [{
        'Seed': seed,
        'Sample set': name,
        **S.score_samples(samples, test_features, feature_net, x_train)
        } for name, samples in sample_sets.items()
        ]

# Define the function to measure the Fréchet distance by size
def measure_frechet_by_size(x_train, feature_net, test_features, sample_sizes, n_repeats = 10, seed = 0):
    # Set the seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    # Get the train features
    train_features = S.get_features(feature_net, x_train)
    rows = []

    for size in sample_sizes:
        for repeat in range(1, n_repeats + 1):
            indices = torch.randperm(len(x_train), generator = generator)[:size]
            distance = S.frechet_distance(train_features[indices.numpy()], test_features)
            rows.append({'Subset size': size, 'Repeat': repeat, 'Fréchet': distance})

    return rows