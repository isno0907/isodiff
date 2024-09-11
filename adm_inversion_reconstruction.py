from diffusers import DDPMScheduler, DDIMScheduler, VQModel, DDIMInverseScheduler
from h_unet import UNet2DModel_H, UNet2DModel_G, ADMP2Scheduler, ADMPipeline_H
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
        default="google/ddpm-ema-celebahq-256", #ADM/CelebA_HQ_G_32/checkpoint-1300
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--seeds",
        type=arg_as_list,
        default=[],
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
        "--inversion_portion",
        type=float,
        default=1.0,
        help="inversion portion",
    )
    parser.add_argument(
        "--inversion_steps",
        type=int,
        default=20,
        help="# of inversion steps for DDIM",
    )

    args = parser.parse_args()

    return args
##############################################################################

def main():
    args = parse_args()

    seeds = args.seeds
    # seeds = range(100)
    path = args.save_path
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    print(f"Saving path= {path}")

    # ADM
    unet = UNet2DModel_G(sample_size=256)

    if args.unet_path.endswith('pt'):
        print(f"Loading weight from {args.unet_path}")
        unet.load_weight(args.unet_path)
    else:
        print(f"Loading weight from {args.unet_path}")
        unet = unet.from_pretrained(args.unet_path)

    scheduler = ADMP2Scheduler(model=unet, timestep_respacing=f'ddim{args.num_inference_steps}')

    image_shape = (
                1,
                unet.config.in_channels,
                unet.config.sample_size,
                unet.config.sample_size,
            )

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    unet.to(torch_device)
    pipeline = ADMPipeline_H(
                            unet=unet,
                            scheduler=scheduler,
                            )
    pipeline.to(torch_device)
    generator = torch.Generator(device='cuda')
    # seeds = [x for x in range(300,700)]
    for seed in seeds:
        image = read_image(f'assets/examples/image_{seed}.png').to(torch_device).to(torch.float).unsqueeze(0) / 127.5 - 1
        inversed_image = scheduler.ddim_reverse_sample_loop(image, inv_p=args.inversion_portion)
        pil_image = pipeline(batch_size=1, num_inference_steps=args.num_inference_steps, latents=inversed_image, output_type='pil')["images"][0]
        pil_image.save(os.path.join(path,f"recon_{seed}.png"))

    print("Done!")

if __name__ == '__main__':
    main()