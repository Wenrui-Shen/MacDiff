
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm

# from MDM Code
def evaluate_diversity(activations, diversity_times=200):
    print('Start diversity evaluation...')
    num_motions = len(activations)
    diversity_times = min(diversity_times, num_motions)
    print(f'diversity_times: {diversity_times}')

    diversity = 0

    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += np.linalg.norm(activations[first_idx,:] - activations[second_idx,:])

    diversity /= diversity_times
    return diversity

# from MDM Code
def evaluate_precision_recall(*, generated_features, real_features, k=3, data_num=None):
    print('Start precision recall evaluation...')
    if data_num is None:
        data_num = min(len(generated_features), len(real_features))
    else:
        data_num = min(data_num, len(generated_features), len(real_features))

    print(f'data num: {data_num}')

    if data_num <= 0:
        print("there is no data")
        return
    
    if isinstance(real_features, np.ndarray): real_features = torch.from_numpy(real_features)
    if isinstance(generated_features, np.ndarray): generated_features = torch.from_numpy(generated_features)
    generated_features = generated_features[:data_num]
    real_features = real_features[:data_num]

    # get precision and recall
    precision = manifold_estimate(real_features, generated_features, k)
    recall = manifold_estimate(generated_features, real_features, k)

    return precision, recall

def manifold_estimate(A_features, B_features, k):
    A_features = list(A_features)
    B_features = list(B_features)
    KNN_list_in_A = {}
    for A in tqdm(A_features, ncols=80, miniters=10):
        pairwise_distances = np.zeros(shape=(len(A_features)))

        for i, A_prime in enumerate(A_features):
            d = torch.norm((A - A_prime), 2)
            pairwise_distances[i] = d

        v = np.partition(pairwise_distances, k)[k]
        KNN_list_in_A[A] = v

    n = 0
    for B in tqdm(B_features, ncols=80, miniters=10):
        for A_prime in A_features:
            d = torch.norm((B - A_prime), 2)
            if d <= KNN_list_in_A[A_prime]:
                n += 1
                break
    return n / len(B_features)

# from Action2Motion Code
def evaluate_fid(activations, gt_activations):
    print('Start fid evaluation...')
    print(f'activations shape: {activations.shape}, gt_activations shape: {gt_activations.shape}')
    mu1, sigma1 = calculate_activation_statistics(activations)
    mu2, sigma2 = calculate_activation_statistics(gt_activations)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

def calculate_activation_statistics(activations):
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)