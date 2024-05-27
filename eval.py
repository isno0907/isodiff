import torch
import copy
import numpy as np
import torch.distributed as dist
import pickle
import dnnlib
import numpy as np
import time
import os
import random
from scipy import linalg
from dnnlib.util import open_url
from torchvision.transforms import ToTensor, ToPILImage, Compose
from diffusers.utils import randn_tensor
from PIL import Image
from tqdm import tqdm
from utils import local_basis

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
import torch.autograd.functional as A
from einops import rearrange

from h_unet import UNet2DModel_G

def get_activations(dl, model, batch_size, device, max_samples, dl_include_step=True):
    pred_arr = []
    total_processed = 0

    # print('Starting to sample.')
    if dl_include_step:
        for step, batch in tqdm(enumerate(dl)):
            batch = batch["pixel_values"].to(torch.float16)
            # ignore labels
            if isinstance(batch, list):
                batch = batch[0]

            batch = batch.to(device)
            if batch.shape[1] == 1:  # if image is gray scale
                batch = batch.repeat(1, 3, 1, 1)
            elif len(batch.shape) == 3:  # if image is gray scale
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

            with torch.no_grad():
                batch = (batch / 2. + .5).clamp(0., 1.)
                batch = (batch * 255.).to(torch.uint8)
                pred = model(batch, return_features=True).unsqueeze(-1).unsqueeze(-1)

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
            total_processed += pred.shape[0]
            if max_samples is not None and total_processed > max_samples:
                print('Max of %d samples reached.' % max_samples)
                break

        pred_arr = np.concatenate(pred_arr, axis=0)
        if max_samples is not None:
            pred_arr = pred_arr[:max_samples]
    else:
        for batch in dl:
            # ignore labels
            if isinstance(batch, list):
                batch = batch[0]

            batch = batch.to(device)
            if batch.shape[1] == 1:  # if image is gray scale
                batch = batch.repeat(1, 3, 1, 1)
            elif len(batch.shape) == 3:  # if image is gray scale
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)
            with torch.no_grad():
                pred = model(batch, return_features=True).unsqueeze(-1).unsqueeze(-1)

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
            total_processed += pred.shape[0]
            if max_samples is not None and total_processed > max_samples:
                print('Max of %d samples reached.' % max_samples)
                break

        pred_arr = np.concatenate(pred_arr, axis=0)
        if max_samples is not None:
            pred_arr = pred_arr[:max_samples]

    return pred_arr

def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size

def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)
    return x

def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir='.stylegan2-cache')
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()
        t = t.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

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

    # print("s1 dot s2", sigma1.dot(sigma2))
    # print("covmean", covmean)
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_fid(n_samples, n_gpus, sampling_shape, num_inference_steps, sampler, gen, stats_path, device, accelerator=None, text=None, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    tf_toTensor = ToTensor()
    transform = Compose([ToTensor()])
    
    def generator(num_samples):
        num_sampling_rounds = int(np.ceil(num_samples / sampling_shape[0]))
        with torch.autocast("cuda"):
            pbar = tqdm(total = num_sampling_rounds, disable=not accelerator.is_local_main_process)
            for _ in range(num_sampling_rounds):
                x = sampler(batch_size= sampling_shape[0], num_inference_steps=num_inference_steps, generator=gen, output_type='pt')
                x = ((x+1)/2 * 255).to(torch.uint8) # Range Debugging
                pbar.update(1)
                yield x

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(device)
        inception_model.eval()

    act = get_activations(generator(n_samples), inception_model,
                          sampling_shape[0], device=device, max_samples=None, dl_include_step=False)
    if accelerator.num_processes > 1:
        return act
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()

    # print(m.shape, s.shape)
    # print("before = ", m, s)
    # average_tensor(m)
    # average_tensor(s)
    # print("after = ", m, s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    # print(all_pool_mean, all_pool_sigma)
    # print()

    stats = np.load(stats_path)
    data_pools_mean = stats['mu']
    data_pools_sigma = stats['sigma']

    # print(data_pools_mean, data_pools_sigma)

    fid = calculate_frechet_distance(data_pools_mean,
                data_pools_sigma, all_pool_mean, all_pool_sigma)
    return fid

def compute_ppl(n_samples, n_gpus, sampling_shape, num_inference_steps, sampler, gen, device, accelerator=None, text=None, n_classes=None, epsilon=1e-4, in_latent=False):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    sampling_shape[0] = sampling_shape[0] // 2
    epsilon = epsilon
    tf_toTensor = ToTensor()
    if type(text) is list:
        text = np.asarray(text)

    def generator(sampling_shape, gen):
        with torch.autocast("cuda"):
            with torch.no_grad():
                s_shape = (sampling_shape[0] * 2, sampling_shape[1], sampling_shape[2], sampling_shape[3])
                z = torch.randn(s_shape, device=device, generator=gen) #dtype=sampler.unet.dtype, 
                z0, z1 = z.split(split_size=sampling_shape[0], dim=0)

                w = torch.rand(sampling_shape[0], device=device)
                w = add_dimensions(w, 3)

                zt0 = slerp(z0, z1, w)
                zt1 = slerp(z0, z1, w + epsilon)

                # sampler.scheduler.set_timesteps(num_inference_steps=20)

                # For X-space interpolation
                
                # x0 = sampler(latents=zt0, num_inference_steps=20, generator=gen, output_type='pt')
                # x1 = sampler(latents=zt1, num_inference_steps=20, generator=gen, output_type='pt')
                if in_latent:
                    x = sampler(latents=torch.cat([zt0, zt1]), num_inference_steps=num_inference_steps, output_type='latent')
                else:
                    x = sampler(latents=torch.cat([zt0, zt1]), num_inference_steps=num_inference_steps, output_type='pt')
                x0, x1 = x[:sampling_shape[0]], x[sampling_shape[0]:]

                # x = sampler(latents=torch.cat([z0, z1]), num_inference_steps=20, output_type='pt')
                # x0_end, x1_end = x[:sampling_shape[0]], x[sampling_shape[0]:]

                # For H-space interpolation
                # x0, x1 = zt0, zt1
                # h_features_x0, h_features_x1 = [], []

                # for t in sampler.scheduler.timesteps:
                #     residual_x0 = sampler.unet(x0, t)[0].sample
                #     residual_x1 = sampler.unet(x1, t)[0].sample

                #     h_feature_x0 = sampler.unet(x0, t)[1]
                #     h_feature_x1 = sampler.unet(x1, t)[1]

                #     h_features_x0.append(h_feature_x0)
                #     h_features_x1.append(h_feature_x1)

                #     prev_image_x0 = sampler.scheduler.step(residual_x0, t, x0, eta=0.0).prev_sample
                #     prev_image_x1 = sampler.scheduler.step(residual_x1, t, x1, eta=0.0).prev_sample

                #     x0 = prev_image_x0
                #     x1 = prev_image_x1

                # x0, x1 = zt0, zt1
                # w = w.type(h_features_x0[0].dtype)

                # for i, t in enumerate(sampler.scheduler.timesteps[:15]):

                #     residual_x0 = sampler.unet(x0, t)[0].sample
                #     residual_x1 = sampler.unet(x1, t)[0].sample

                #     x0 = sampler.scheduler.step(residual_x0, t, x0, eta=0.0).prev_sample
                #     x1 = sampler.scheduler.step(residual_x1, t, x1, eta=0.0).prev_sample

                # for i, t in enumerate(sampler.scheduler.timesteps[15:]):

                #     residual_x0 = sampler.unet(x0, t, h_feature_in= torch.lerp(h_features_x0[i], h_features_x1[i], w))[0].sample
                #     residual_x1 = sampler.unet(x1, t, h_feature_in= torch.lerp(h_features_x0[i], h_features_x1[i], w + epsilon))[0].sample

                #     x0 = sampler.scheduler.step(residual_x0, t, x0, eta=0.0).prev_sample
                #     x1 = sampler.scheduler.step(residual_x1, t, x1, eta=0.0).prev_sample

                # print(x0.max(), x0.min())
                # save_image(x0, "x0.png")
                # save_image(x1, "x1.png")

        return x0, x1, w #, x0_end, x1_end

    def calculate_lpips(x0, x1):
        # loss_fn_alex = lpips.LPIPS(net = 'alex', verbose = False).to(device)
        # dist = loss_fn_alex(x0, x1).square().sum((1,2,3)) / epsilon ** 2
        # print(loss_fn_alex(x0, x1).shape)

        # epsilon = 1e-6
        # distance_measure = load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/vgg16_zhang_perceptual.pkl')
        # dist = distance_measure.get_output_for(x0, x1) * (1 / epsilon**2)

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device) # input detail is described in "torchmetric docs"
        dist = lpips(x0, x1) / epsilon ** 2

        return dist

    dist_list = []
    # if accelerator.local_process_index == 0:
    pbar = tqdm(total = int(np.ceil(n_samples / sampling_shape[0])), disable=not accelerator.is_local_main_process )
    for i in range(0, n_samples, sampling_shape[0]):
        gen = torch.Generator(device='cuda')
        gen.manual_seed(i)
        x0, x1, w = generator(sampling_shape, gen)

        if in_latent:
            dist = ((x0 - x1)**2).mean() / epsilon ** 2
        else:
            dist = calculate_lpips(x0, x1)

        if dist.detach().cpu().item() > 10000:
            # x0 = (x0 + 1)/2
            # x1 = (x1 + 1)/2
            # x0_end = (x0_end + 1)/2
            # x1_end = (x1_end + 1)/2

            # print(f'dist={dist}, seed={i}, w={w.item():.2f}')
            # save_image(x0, f"output_jae/ppl_div_seeds/x0_seed={i}_w={w.item():.2f}.png")

            # save_image(x0_end, f"output_jae/ppl_div_seeds/x0_end_seed={i}.png")
            # save_image(x1, f"output_jae/ppl_div_seeds/x1_seed={i}.png")
            # save_image(x1_end, f"output_jae/ppl_div_seeds/x1_end_seed={i}.png")
            # save_image(abs(x0-x1) * 10, f"output_jae/ppl_div_seeds/gap_seed={i}.png")

            pass
        dist_list.append(dist.detach().cpu())
        pbar.update(1)

    accelerator.wait_for_everyone()
    # Compute PPL.
    dist_list = np.array(dist_list)
    lo = np.percentile(dist_list, 1, interpolation='lower') #1
    hi = np.percentile(dist_list, 99, interpolation='higher') #99
    ppl = np.extract(np.logical_and(dist_list >= lo, dist_list <= hi), dist_list).mean()

    return float(ppl)


#################################################


def compute_ppl_end(n_samples, n_gpus, sampling_shape, num_inference_steps, sampler, gen, device, accelerator=None, text=None, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    sampling_shape[0] = sampling_shape[0] // 2
    epsilon = 1e-4
    tf_toTensor = ToTensor()
    if type(text) is list:
        text = np.asarray(text)

    def generator(sampling_shape, gen):
        with torch.autocast("cuda"):
            with torch.no_grad():
                s_shape = (sampling_shape[0] * 2, sampling_shape[1], sampling_shape[2], sampling_shape[3])
                z = torch.randn(s_shape, device=device, generator=gen) #dtype=sampler.unet.dtype, 
                z0, z1 = z.split(split_size=sampling_shape[0], dim=0)

                w = 0.5 * torch.ones(sampling_shape[0], device=device)
                w = (1 - epsilon) * torch.bernoulli(w)
                w = add_dimensions(w, 3)

                zt0 = slerp(z0, z1, w)
                zt1 = slerp(z0, z1, w + epsilon)

                x = sampler(latents=torch.cat([zt0, zt1]), num_inference_steps=num_inference_steps, output_type='pt')
                x0, x1 = x[:sampling_shape[0]], x[sampling_shape[0]:]


        return x0, x1, w

    def calculate_lpips(x0, x1):
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device) # input detail is described in "torchmetric docs"
        dist = lpips(x0, x1) / epsilon ** 2
        return dist

    dist_list = []
    # if accelerator.local_process_index == 0:
    pbar = tqdm(total = int(np.ceil(n_samples / sampling_shape[0])), disable=not accelerator.is_local_main_process )
    for i in range(0, n_samples, sampling_shape[0]):
        gen = torch.Generator(device='cuda')
        gen.manual_seed(i)
        x0, x1, w = generator(sampling_shape, gen)
        dist = calculate_lpips(x0, x1)
        dist_list.append(dist.detach().cpu())
        pbar.update(1)
    accelerator.wait_for_everyone()
    dist_list = np.array(dist_list)
    lo = np.percentile(dist_list, 1, interpolation='lower')
    hi = np.percentile(dist_list, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist_list >= lo, dist_list <= hi), dist_list).mean()

    return float(ppl)


###########################

def find_seeds_2d(n_samples, n_gpus, sampling_shape, num_inference_steps, sampler, gen=None, device=None, accelerator=None, text=None, n_classes=None):
    print('Finding unstable seeds for 2d_grid.')
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    sampling_shape[0] = sampling_shape[0] // 3
    epsilon = 1e-4
    tf_toTensor = ToTensor()
    if type(text) is list:
        text = np.asarray(text)

    def generator(sampling_shape, gen):
        with torch.autocast("cuda"):
            with torch.no_grad():
                s_shape = (sampling_shape[0] * 3, sampling_shape[1], sampling_shape[2], sampling_shape[3])
                z = torch.randn(s_shape, device=device, generator=gen) #dtype=sampler.unet.dtype, 
                z0, z1, z2 = z.split(split_size=sampling_shape[0], dim=0)

                w = torch.rand(sampling_shape[0], device=device)
                w = add_dimensions(w, 3)

                zt0_w = slerp(z0, z1, w)
                zt1_w = slerp(z0, z1, w + epsilon)

                v = torch.rand(sampling_shape[0], device=device)
                v = add_dimensions(w, 3)

                zt0_v = slerp(z0, z2, w)
                zt1_v = slerp(z0, z2, w + epsilon)

                x = sampler(latents=torch.cat([zt0_w, zt1_w, zt0_v, zt1_v]), num_inference_steps=num_inference_steps, output_type='pt')
                x0_w, x1_w, x0_v, x1_v = x.split(sampling_shape[0], 0)

        return x0_w, x1_w, x0_v, x1_v
    
    def calculate_lpips(x0, x1):
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device) # input detail is described in "torchmetric docs"
        dist = lpips(x0, x1) / epsilon ** 2
        return dist

    dist_list = []
    # if accelerator.local_process_index == 0:
    # pbar = tqdm(total = int(np.ceil(n_samples / sampling_shape[0])))
    for i in tqdm(range(0, n_samples, sampling_shape[0])):
        gen = torch.Generator(device='cuda')
        gen.manual_seed(i)
        x0_w, x1_w, x0_v, x1_v = generator(sampling_shape, gen)
        dist_w = calculate_lpips(x0_w, x1_w)
        dist_v = calculate_lpips(x0_v, x1_v)
        
        # print(i, f'dist_w={dist_w}, dist_v={dist_v}')
              
        if  dist_w.detach().cpu().item() > 2000 or dist_v.detach().cpu().item() > 2000:
            # x0 = (x0 + 1)/2
            # x1 = (x1 + 1)/2

            print(f'dist_w={dist_w}, dist_v={dist_v}, seed={i}')
            # save_image(x0, f"output_jae/ppl_div_seeds/x0_seed={i}_w={w.item():.2f}.png")

            # save_image(x0_end, f"output_jae/ppl_div_seeds/x0_end_seed={i}.png")
            # save_image(x1, f"output_jae/ppl_div_seeds/x1_seed={i}.png")
            # save_image(x1_end, f"output_jae/ppl_div_seeds/x1_end_seed={i}.png")
            # save_image(abs(x0-x1) * 10, f"output_jae/ppl_div_seeds/gap_seed={i}.png")

            pass


##############################################

def compute_distortion_per_timesteps(n_samples, n_gpus, sampling_shape, sampler, gen, device, text=None, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    epsilon = 1e-3
    tf_toTensor = ToTensor()
    if type(text) is list:
        text = np.asarray(text)

    def generator(sampling_shape, interrupt_step):
        with torch.autocast("cuda"):
            with torch.no_grad():
                z0 = torch.randn(sampling_shape, device=device, dtype=sampler.text_encoder.dtype)
                z1 = torch.randn(sampling_shape, device=device, dtype=sampler.text_encoder.dtype)
                y0 = sampler._encode_prompt(list(text[torch.randint(0,len(text),[sampling_shape[0]])]), device, 1, True)[:sampling_shape[0]]
                y1 = sampler._encode_prompt(list(text[torch.randint(0,len(text),[sampling_shape[0]])]), device, 1, True)[:sampling_shape[0]]
                            
                t = torch.rand(sampling_shape[0], device=device, dtype=sampler.text_encoder.dtype)
                t = add_dimensions(t, 3)

                zt0 = slerp(z0, z1, t)
                zt1 = slerp(z0, z1, t + epsilon)
                yt0 = torch.lerp(y0.unsqueeze(1), y1.unsqueeze(1), t).squeeze(1)
                yt1 = torch.lerp(y0.unsqueeze(1), y1.unsqueeze(1), t).squeeze(1)

                x0 = sampler(latents=zt0, prompt_embeds=yt0, num_inference_steps=50, generator=gen, output_type='latent', interrupt_step=interrupt_step).images
                x1 = sampler(latents=zt1, prompt_embeds=yt1, num_inference_steps=50, generator=gen, output_type='latent', interrupt_step=interrupt_step).images
                x0 = (x0 * 2 - 1).clip(-1., 1.)
                x1 = (x1 * 2 - 1).clip(-1., 1.)
                imgs = (x0, x1)

        return imgs

    def calculate_l2(x0, x1):
        dist = torch.sum((x0 - x1) ** 2, dim=(1,2,3))
        return dist

    dist_list_per_timesteps = []
    for j in range(1,51):
        print(f"Calculating {j}th timestep's distortion.")
        dist_list = []
        for i in range(0, n_samples, sampling_shape[0]):
            x0, x1 = generator(sampling_shape, interrupt_step=j)
            dist = calculate_l2(x0, x1)
            dist_list.append(dist.detach().cpu())

        dist_list = torch.cat(dist_list)[:n_samples].cpu().detach().numpy()
        print("dist_list=", dist_list)
        lo = np.percentile(dist_list, 10, interpolation='lower')
        hi = np.percentile(dist_list, 90, interpolation='higher')
        ppl = np.extract(np.logical_and(dist_list >= lo, dist_list <= hi), dist_list).mean()
        dist_list_per_timesteps.append(float(ppl))

    return dist_list_per_timesteps

#################################################################

def get_random_local_basis(model, random_state, noise = None, noise_dim = 512):
    '''
    noise_dim = 512 for StyleGAN, 128 for BigGAN
    
    ex)
    random_state = np.random.RandomState(seed)
    noise, z, z_local_basis, z_sv = get_random_local_basis(model, random_state)
    '''
    n_samples = 1
    if noise is not None:
        assert(list(noise.shape) == [n_samples, noise_dim])
        noise = noise.detach().float().to(model.device)
    else:
        noise = torch.from_numpy(
                random_state.standard_normal(noise_dim * n_samples)
                .reshape(n_samples, noise_dim)).float().to(model.device) #[N, noise_dim]
    noise.requires_grad = True
    
    if isinstance(model, StyleGAN2):
        mapping_network = model.model.style
    elif isinstance(model, StyleGAN):
        mapping_network = model.model._modules['g_mapping'].forward 
    elif isinstance(model, BigGAN):
        mapping_network = model.partial_forward_explicit
    else:
        raise NotImplemented   
    z = mapping_network(noise)

    ''' Compute Jacobian by batch '''
    noise_dim, z_dim = noise.shape[1], z.shape[1]
    noise_pad = noise.repeat(z_dim, 1).requires_grad_(True)
    z_pad = mapping_network(noise_pad)

    grad_output = torch.eye(z_dim).cuda()
    jacobian = torch.autograd.grad(z_pad, noise_pad, grad_outputs=grad_output, retain_graph=True)[0].cpu()
    
    ''' Get local basis'''
    # jacobian \approx torch.mm(torch.mm(z_basis, torch.diag(s)), noise_basis.t())
    z_basis, s, noise_basis = torch.svd(jacobian)
    return noise, z.detach(), z_basis.detach(), s.detach(), noise_basis.detach()


def compute_geodesic_metric(local_basis_1, local_basis_2, subspace_dim):
    subspace_1 = np.array(local_basis_1[:, :subspace_dim])
    subspace_2 = np.array(local_basis_2[:, :subspace_dim])
    
    u, s, v = np.linalg.svd(np.matmul(subspace_1.transpose(), subspace_2))
    s[s > 1] = 1
    s = np.arccos(s)
    return np.linalg.norm(s)

def forward(sample, unet, scheduler, n):
    image = sample
    noise = torch.randn_like(sample)
    if isinstance(unet, UNet2DModel_G):
        noisy_sample = scheduler.q_sample(sample, torch.tensor(n).to('cuda'), noise)
    else:
        noisy_sample = scheduler.add_noise(sample, noise, scheduler.timesteps[n])
    return noisy_sample

def compute_mcn(n_samples, sampling_shape, pooling_kernel, unet, scheduler, gen, device, accelerator=None, timestep_n=None):

    def generator(sampling_shape, gen, timestep_n):
        with torch.autocast("cuda"):
            with torch.no_grad():
                x = torch.randn(sampling_shape, device=device, generator=gen)
                if timestep_n is None:
                    print("timestep_n is not given!")
                    raise NotImplementedError
                sample = forward(x, unet, scheduler, timestep_n)
                # t = scheduler.timesteps[timestep_n]
                t = timestep_n
                # pooling_layer = torch.nn.AvgPool2d(pooling_kernel)
                # if hasattr(unet, "forward_H"):
                #     J = A.jacobian(lambda x: pooling_layer(unet.forward_H(x, t)), sample)
                # else:
                #     raise ("Unet does not have forward_H.")
                # J = torch.diagonal(J, 0, dim1=0, dim2=4)
                # s = torch.linalg.svdvals(rearrange(J, "hc hh hw xc xh xw b -> b (hc hh hw) (xc xh xw)"))

                _, _, s, _ = local_basis(unet, sample=sample, t=t, seed=None, pooling_kernel=pooling_kernel, shape=sampling_shape, revert_to_ori=False)
                cn = (s[:,0]/s[:,-1]) #**2
                # print('n=', timestep_n, 'cn=', cn)
        return cn.mean()

    cn_list = []
    pbar = tqdm(total = int(np.ceil(n_samples / sampling_shape[0])), disable=not accelerator.is_local_main_process )
    
    for i in range(0, n_samples, sampling_shape[0]):
        gen = torch.Generator(device='cuda')
        gen.manual_seed(i)
        cn = generator(sampling_shape, gen, timestep_n)
        cn_list.append(cn.detach().cpu())
        pbar.update(1)

    accelerator.wait_for_everyone()
    cn_list = np.array(cn_list)
    # lo = np.percentile(cn_list, 1, interpolation='lower') #1
    # hi = np.percentile(cn_list, 99, interpolation='higher') #99
    # mcn = np.extract(np.logical_and(cn_list >= lo, cn_list <= hi), cn_list).mean()
    mcn = cn_list.mean()

    return mcn

def compute_vor(n_samples, sampling_shape, pooling_kernel, unet, scheduler, gen, device, accelerator=None, timestep_n=None):

    def generator(sampling_shape, gen, timestep_n):
        with torch.autocast("cuda"):
            with torch.no_grad():
                x = torch.randn(sampling_shape, device=device, generator=gen)
                if timestep_n is None:
                    timestep_n = random.choice(range(0, len(scheduler.timesteps)//2))
                sample = forward(x, unet, scheduler, timestep_n)
                # t = scheduler.timesteps[timestep_n]
                t = timestep_n

                # pooling_layer = torch.nn.AvgPool2d(pooling_kernel)
                # if hasattr(unet, "forward_H"):
                #     J = A.jacobian(lambda x: pooling_layer(unet.forward_H(x, t)), sample)
                # else:
                #     raise ("Unet does not have forward_H.")
                # J = torch.diagonal(J, 0, dim1=0, dim2=4)
                # s = torch.linalg.svdvals(rearrange(J, "hc hh hw xc xh xw b -> b (hc hh hw) (xc xh xw)"))

                _, _, s, _ = local_basis(unet, sample=sample, t=t, seed=None, pooling_kernel=pooling_kernel, shape=sampling_shape, revert_to_ori=False)
        return s

    s_list = None
    pbar = tqdm(total = int(np.ceil(n_samples / sampling_shape[0])), disable=not accelerator.is_local_main_process )
    
    for i in range(0, n_samples, sampling_shape[0]):
        gen = torch.Generator(device='cuda')
        gen.manual_seed(i)
        s = generator(sampling_shape, gen, timestep_n)

        if s_list is None:
            s_list = s
        else:
            s_list = torch.cat((s_list, s), dim=0)
        pbar.update(1)

    vor = torch.log(s_list).var(dim=0)
    vor = vor.sum()
    # lo = np.percentile(cn_list, 1, interpolation='lower') #1
    # hi = np.percentile(cn_list, 99, interpolation='higher') #99
    # vor = np.extract(np.logical_and(cn_list >= lo, cn_list <= hi), cn_list).mean()

    return s_list, vor
