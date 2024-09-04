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
        default=range(10),
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
    parser.add_argument(
        "--inversion_steps",
        type=int,
        default=20,
        help="# of inversion steps for DDIM",
    )
    # parser.add_argument("--interp_cond", default=False, action="store_true", help="interpolation")
    parser.add_argument("--image_save", default=False, action="store_true", help="save image")
    # parser.add_argument("--find_seeds", default=False, action="store_true", help="find unstable seeds")
    parser.add_argument("--lpips", default=False, action="store_true", help="measure lpips distance btwn each frame")
    # parser.add_argument("--base", default=False, action="store_true", help="measure lpips distance btwn each frame")

    args = parser.parse_args()

    return args
##############################################################################

def main():
    args = parse_args()

    seeds = args.seeds
    image_save = args.image_save
    path = args.save_path
    os.makedirs(path, exist_ok=True)

    unet = UNet2DModel_H.from_pretrained(args.unet_path)
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    inverse_scheduler = DDIMInverseScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
    inverse_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    unet.to(torch_device)

    for seed in seeds:
        image = read_image(f'path_to_image/image_{seed}.png').to(torch_device).to(torch.float).unsqueeze(0) / 127.5 - 1
        
        for t in inverse_scheduler.timesteps[:args.inversion_steps]:
            with torch.no_grad():
                residual = unet(image, t)[0]["sample"]

            prev_image = inverse_scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
            image = prev_image

        for t in scheduler.timesteps[-args.inversion_steps:]:
            with torch.no_grad():
                residual = unet(image, t)[0]["sample"]

            prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
            image = prev_image

        if image_save:
            image_processed = image.cpu().permute(0, 2, 3, 1)
            image_processed = (image_processed + 1.0) * 127.5
            image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
            image_pil = PIL.Image.fromarray(image_processed[0])
            image_pil.save(path + f"/recovered_image_{seed}.png")

    print("Done!")

if __name__ == '__main__':
    main()