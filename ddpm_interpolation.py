from diffusers import DDPMScheduler, DDIMScheduler, VQModel, DDIMInverseScheduler
from h_unet import UNet2DModel_H
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
import torch
import PIL.Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import argparse, ast
import os

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
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

def image_process(image):
    image_processed = image.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).type(torch.uint8)

    return image_processed

def arg_as_list(s):
    return ast.literal_eval(s)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/default/",
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        default="google/ddpm-ema-celebahq-256",
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--seeds",
        type=arg_as_list,
        default=range(200),
        help="Random seeds",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Random seeds",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="# of inference steps for DDIM",
    )
    parser.add_argument("--interp_cond", default=False, action="store_true", help="interpolation")
    # parser.add_argument("--image_save", default=False, action="store_true", help="save image")
    # parser.add_argument("--find_seeds", default=False, action="store_true", help="find unstable seeds")
    parser.add_argument("--lpips", default=False, action="store_true", help="measure lpips distance btwn each frame")
    args = parser.parse_args()

    return args
##############################################################################

def main():
    args = parse_args()

    seeds = args.seeds
    # seeds = range(200)
    # noises = []
    # h_features = {seed:[] for seed in seeds}
    # interp_cond = args.interp_cond
    # image_save = args.image_save
    path = args.save_path
    os.makedirs(path, exist_ok=True)
    print(f"Saving path= {path}")

    unet = UNet2DModel_H.from_pretrained(args.unet_path)
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    unet.to(torch_device)

    for n, seed in enumerate(seeds):
        images_x_slerp = []

        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed)
        noise = torch.randn(
            (2, unet.config.in_channels, unet.sample_size, unet.sample_size),
            device = torch_device, generator=generator,
        )
        z0, z1 = noise[0].unsqueeze(0), noise[1].unsqueeze(0)

        step = args.step
        for w in torch.arange(1e-4, 1 + step, step, device = torch_device):
            image_x_slerp = slerp(w, z0, z1) 

            for i, t in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    residual_x_slerp = unet(image_x_slerp, t)[0]["sample"]

                image_x_slerp = scheduler.step(residual_x_slerp, t, image_x_slerp, eta=0.0)["prev_sample"]

            images_x_slerp.append(image_x_slerp/2 + 0.5)

        if args.lpips:
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(torch_device) # input detail is described in "torchmetric docs"
            lpips_dists = []
            for i in range(len(images_x_slerp)-1):
                image = images_x_slerp[i]
                image_next = images_x_slerp[i+1]
                lpips_dists.append(round(float(lpips(image, image_next)), 3))

        grid_x_slerp = make_grid(torch.cat(images_x_slerp, 0), nrow=int(1/step) + 1, padding=0)
        
        save_image(grid_x_slerp, path + f"/x_slerp_{seed}->{seed}.png")
        print(f'Saved {seed}->{seed}.png to {path}.')

        if args.lpips:
            print('lpips_dists=', lpips_dists, round(sum(lpips_dists)/ len(lpips_dists), 3))

        if n==len(seeds)-1:
            break

    print("Done!")

if __name__ == '__main__':
    main()