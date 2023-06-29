import torch
import copy
import numpy as np

from compute_fid import calculate_frechet_distance
from compute_fid_statistics import get_activations
from utils.util import average_tensor


def compute_fid(n_samples, n_gpus, sampling_shape, sampler, inception_model, stats_paths, device, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))

    def generator(num_samples):
        num_sampling_rounds = int(
            np.ceil(num_samples / sampling_shape[0]))
        for n in range(num_sampling_rounds):
            x = torch.randn(sampling_shape, device=device)

            if n_classes is not None:
                y = torch.randint(n_classes, size=(
                    sampling_shape[0],), dtype=torch.int32, device=device)
                x, _ = sampler(x, y=y)

            else:
                x, _ = sampler(x)

            x = (x / 2. + .5).clip(0., 1.)
            x = (x * 255.).to(torch.uint8)
            yield x

    act = get_activations(generator(num_samples_per_gpu), inception_model,
                          sampling_shape[0], device=device, max_samples=n_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()
    average_tensor(m)
    average_tensor(s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    fid = []
    for stats_path in stats_paths:
        stats = np.load(stats_path)
        data_pools_mean = stats['mu']
        data_pools_sigma = stats['sigma']
        fid.append(calculate_frechet_distance(data_pools_mean,
                   data_pools_sigma, all_pool_mean, all_pool_sigma))
    return fid


def compute_ppl(n_samples, n_gpus, sampling_shape, sampler, vgg_model, epsilon, device, n_classes=None):
    
    def generator(num_samples):
        num_sampling_rounds = int(
            np.ceil(num_samples / sampling_shape[0]))
        for n in range(num_sampling_rounds):
            z0 = torch.randn(sampling_shape, device=device)
            z1 = torch.randn(sampling_shape, device=device)
            t = torch.randn(sampling_shape.shape[0], device=device)

            zt0 = z0.lerp(z1, t)

            if n_classes is not None:
                y = torch.randint(n_classes, size=(
                    sampling_shape[0],), dtype=torch.int32, device=device)
                x, _ = sampler(x, y=y)

            else:
                x, _ = sampler(x)

            x = (x / 2. + .5).clip(0., 1.)
            x = (x * 255.).to(torch.uint8)
            yield x

    # Compute PPL.
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()

    return float(ppl)