import argparse
import inspect
import logging
import math
import shutil
import os
import shutil
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import torch
import torch.nn.functional as F
import torch.autograd.functional as A
import numpy as np
import random
# import tensorflow as tf

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DiffusionPipeline, DDIMScheduler, VQModel
from h_unet import UNet2DModel_H, LDMPipeline_H
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available, randn_tensor
from diffusers.utils.import_utils import is_xformers_available

from eval import compute_fid, compute_ppl, add_dimensions, compute_distortion_per_timesteps
from utils import stereographic_proj, inv_stereographic_proj, riemmanian_metric, isometry_loss_t, isometry_loss_h
from accelerate import DistributedDataParallelKwargs

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=1, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=5, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=50)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="scaled_linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    ################################
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split of the dataset.",
    )
    parser.add_argument(
        "--lambda_iso", type=float, default=0, help="Hyperparameter, lambda value for pl_penalty"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of inference steps of reverse solver.",
    )
    parser.add_argument(
        "--fid_stats_path",
        type=str,
        default="assets/stats/celeba_hq_256.npz",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prompts_reps",
        type=int,
        default=1,
        help=("Number of generating repetitions for each validation prompt."),
    )

    parser.add_argument("--fid", default=False, action="store_true", help="Compute metrics.")
    parser.add_argument("--ppl", default=False, action="store_true", help="Compute metrics.")
    parser.add_argument("--noG", default=False, action="store_true", help="Apply naive G=I")
    parser.add_argument("--metric_step", type=int, default=250)
    parser.add_argument("--metric_batch", type=int, default=64)
    parser.add_argument("--normal_p", type=float, default=0.75)
    parser.add_argument("--timestep_weight", action="store_true", help="Apply linear decreasing iso loss")
    parser.add_argument("--lazy_reg_interval", type=int, default=2)


    parser.add_argument("--dists", action="store_true", help="Compute metrics.")
    parser.add_argument("--subfolder", action="store_true", help="Whether to use subfolder='unet' in from_pretrained method.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    ################################

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

#######################################################################
def compute_metrics(pipeline, args, accelerator, weight_dtype, step, dataset):
    logger.info("Calculating Metrics... ")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    batch_size = args.train_batch_size * (args.metric_batch if args.lambda_iso == 0 else args.metric_batch)
    sampling_shape = [batch_size, 3, args.resolution, args.resolution]
    # latent_sampling_shape = (batch_size, pipeline.unet.config.in_channels, pipeline.unet.config.sample_size, pipeline.unet.config.sample_size) # unet channel must be modified
    
    if args.fid:
        print("Calculating FID")
        N = 1000
        fid = compute_fid(N, 1, sampling_shape, pipeline, generator, args.fid_stats_path, accelerator.device)
        logging.info('FID at step %d: %.6f' % (step, fid))
        print(f"FID with {N} samples.")
    if args.ppl:
        print("Calculating PPL")
        N = 1000
        ppl = compute_ppl(n_samples=N, n_gpus=1, sampling_shape=sampling_shape, sampler=pipeline, gen=generator, device=accelerator.device)
        logging.info('PPL at step %d: %.6f' % (step, ppl))
        print(f"PPL with {N} samples.")
    if args.dists:
        print("Calculating Distortion per timesteps")
        dists = compute_distortion_per_timesteps(n_samples=10, n_gpus=1, sampling_shape=sampling_shape, sampler=pipeline, gen=generator, device=accelerator.device, text=text)
        print("Distortion per timesteps=")
        for dist in dists:
            print(dist)

    # for tracker in accelerator.trackers:
        # if tracker.name == "tensorboard":
    if args.fid:
        accelerator.log({"fid": fid}, step=step)
    if args.ppl:
        accelerator.log({"ppl": ppl}, step=step)
        # else:
        #     logger.warn(f"logging not implemented for {tracker.name}")
    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         if args.fid:
    #             accelerator.log({"fid": fid}, step=step)
    #         if args.ppl:
    #             accelerator.log({"ppl": ppl}, step=step)
    #     else:
    #         logger.warn(f"logging not implemented for {tracker.name}")
    
    del pipeline
    torch.cuda.empty_cache()

    return None

#######################################################################


def main(args):
    # shutil.rmtree(args.output_dir)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs],
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
    
    args_dict = vars(args)
    # hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in params.items()]
    # tensorboard_tracker = accelerator.get_tracker('tensorboard')
    # tf.summary.text('hyperparameters', tf.stack(hyperparameters))

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel_H)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel_H.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    if args.pretrained_model_name_or_path is not None:
        ########
        if args.subfolder:
            model = UNet2DModel_H.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet')
            vqvae = VQModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="vqvae")

        else:
            model = UNet2DModel_H.from_pretrained(args.pretrained_model_name_or_path)
            vqvae = VQModel.from_pretrained(args.pretrained_model_name_or_path)

    elif args.model_config_name_or_path is None:
        model = UNet2DModel_H(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(224, 448, 672, 896),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )
        # vqvae = VQModel.from_pretrained("stabilityai/stable-diffusion-2-1" , subfolder="vae")
        vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256" , subfolder="vqvae")

    else:
        config = UNet2DModel_H.load_config(args.model_config_name_or_path)
        model = UNet2DModel_H.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel_H,
            model_config=model.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    # accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    # if accepts_prediction_type:
    #     noise_scheduler = DDPMScheduler(
    #         num_train_timesteps=args.ddpm_num_steps,
    #         beta_schedule=args.ddpm_beta_schedule,
    #         prediction_type=args.prediction_type,
    #     )
    # else:
    
    # noise_scheduler = DDIMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    # noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    noise_scheduler = DDIMScheduler.from_pretrained("CompVis/ldm-celebahq-256", subfolder='scheduler')

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        try:
            images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        except:
            images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

            # resume_global_step = global_step
            # first_epoch = global_step // num_update_steps_per_epoch
            # resume_step = resume_global_step

    #######################################################################
    def visual_inspection(pipeline, args, accelerator, weight_dtype, epoch, global_step):
        logger.info("Visual Inspection ...")
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        unet = pipeline.unet

        if args.use_ema:
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            # if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
            generator = torch.Generator().manual_seed(0)
            # generator = torch.Generator(device=pipeline.device).manual_seed(0)
            # run pipeline in inference (sample random noise and denoise)
            
            images = pipeline(
                generator=generator,
                batch_size=args.eval_batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="numpy",
            ).images 
            
            # images = vqvae.decode(images).sample 

            if args.use_ema:
                ema_model.restore(unet.parameters())

            # denormalize the images and save to tensorboard
            images_processed = (images * 255).round().astype("uint8")

            if args.logger == "tensorboard":
                if is_accelerate_version(">=", "0.17.0.dev0"):
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                else:
                    tracker = accelerator.get_tracker("tensorboard")
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), global_step)

            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

        return None
    
    #######################################################################
    # Loss weight dict
    if args.timestep_weight:
        iso_loss_weight = np.linspace(args.lambda_iso * 1e-2, args.lambda_iso, noise_scheduler.config.num_train_timesteps)
    else:
        iso_loss_weight = np.array([args.lambda_iso] * noise_scheduler.config.num_train_timesteps)
    #######################################################################
    
    #######################################################################
    # Train!
    #######################################################################
    # if args.checkpointing_steps == 1000:
    #     raise ("Jaehoon, I changed code here so that args.checkpointing have 20 checkpoints. Check code below")
    # elif args.checkpointing_steps == 777:
    #     pass
    # else:
    #     args.checkpointing_steps = num_update_steps_per_epoch * args.num_epochs * args.gradient_accumulation_steps // 20
    # print("Checkpointing steps = ", args.checkpointing_steps)
    
    for epoch in range(first_epoch, args.num_epochs):
        unet = accelerator.unwrap_model(model)
        vqvae = vqvae.to(accelerator.device)
        ###
        pipeline = LDMPipeline_H(
            unet=unet,
            vqvae=vqvae,
            scheduler=noise_scheduler,
            )
        # visual_inspection(pipeline, args, accelerator, weight_dtype, epoch)

        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        visual_flag = True
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            if (args.ppl or args.fid or args.dists) and global_step % args.metric_step == 0:
                compute_metrics(pipeline, args, accelerator, weight_dtype, global_step, dataset)
            if global_step % args.checkpointing_steps == 0 and visual_flag:
                model.eval()
                visual_inspection(pipeline, args, accelerator, weight_dtype, epoch, global_step)
                model.train()
                visual_flag = False

            clean_images = vqvae.encode(batch["input"]).latents
            # logging.info(type(clean_images), clean_images.max(), clean_images.min())
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # for down_block in pipeline.unet.down_blocks:
            #         for p in down_block.parameters():
            #             p.requires_grad = False
            # for p in pipeline.unet.mid_block.parameters():
            #     p.requires_grad = False

            iso_cond = False
            r = random.random()
            if r < args.normal_p:
                timesteps = torch.randint(0, int(noise_scheduler.config.num_train_timesteps * args.normal_p), (bsz,), device=pipeline.device)
                timesteps = timesteps.long()
            else:
                timesteps = torch.randint(int(noise_scheduler.config.num_train_timesteps * args.normal_p), noise_scheduler.config.num_train_timesteps, (bsz,), device=pipeline.device)
                timesteps = timesteps.long()
                iso_cond = True

                # if global_step % args.lazy_reg_interval == 0:
                #     for down_block in pipeline.unet.down_blocks:
                #         for p in down_block.parameters():
                #             p.requires_grad = True
                #     for p in pipeline.unet.mid_block.parameters():
                #         p.requires_grad = True
                    

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps)[0].sample # (Output, h_features)
                # h_feature = model(noisy_images, timesteps)[1] # (B, 512, 8, 8)

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!

                    ##########################
                    if args.lambda_iso > 0 and iso_cond:
                        # iso_loss = args.lambda_iso * isometry_loss_t(model, noisy_images, timesteps, args, accelerator.device)
                        # assert bsz == 1
                        if args.noG:
                            # iso_loss = iso_loss_weight[timesteps] * isometry_loss_h(model, noisy_images, timesteps, args, accelerator.device)
                            iso_loss = args.lambda_iso * isometry_loss_h(model, noisy_images, timesteps, args, accelerator.device)
                        else:
                            # iso_loss = iso_loss_weight[timesteps] * isometry_loss_t(model, noisy_images, timesteps, args, accelerator.device)
                            iso_loss = args.lambda_iso * isometry_loss_t(model, noisy_images, timesteps, args, accelerator.device)

                        loss += iso_loss
                    ##########################

                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                visual_flag = True

                accelerator.log({"train_loss": loss}, step=global_step)
                if args.lambda_iso > 0 and iso_cond:
                    # accelerator.log({"iso_loss": iso_loss / iso_loss_weight[timesteps]}, step=global_step)
                    accelerator.log({"iso_loss": iso_loss / args.lambda_iso}, step=global_step)
                
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if args.lambda_iso > 0 and iso_cond:
                # logs = {"t": timesteps.detach(), "loss": loss.detach().item(), "loss_iso": iso_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                logs = {"train_loss": loss.detach().item(), "iso_loss": iso_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            else:
                logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if args.use_ema:
                # logs["ema_decay"] = ema_model.cur_decay_value
                pass

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
