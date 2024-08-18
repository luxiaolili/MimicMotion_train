#!/usr/bin/env python
# coding=utf-8


"""Script to train mimicmotion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import MimicMotionModel
from diffusers.models.lora import LoRALinearLayer
# from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from dataset import TikTokDataset
from mimicmotion.modules.pose_net import PoseNet

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()



def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def collate_fn(data):
    tgt_vid = torch.stack([example["tgt_vid"] for example in data])
    tgt_lmk = torch.stack([example["tgt_lmk"] for example in data])
    clip_img = torch.stack([torch.Tensor(example["clip_img"]) for example in data])
    ref_img = torch.stack([example["ref_img"] for example in data])

    return {
        "tgt_vid": tgt_vid,
        "tgt_lmk": tgt_lmk,
        "clip_img": clip_img,
        "ref_img": ref_img,
    }  


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--video_folder",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=127,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # # default to using the same revision for the non-ema model if not specified
    # if args.non_ema_revision is None:
    #     args.non_ema_revision = args.revision

    return args



def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load mimicmotion model
    mimicmotion_model = MimicMotionModel(args.pretrained_model_name_or_path)

    vae = mimicmotion_model.vae
    image_encoder = mimicmotion_model.image_encoder
    unet = mimicmotion_model.unet
    pose_net = mimicmotion_model.pose_net

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    
    pose_net.requires_grad_(True)
    
    del mimicmotion_model
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)


    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [{"params":unet.parameters()},{"params":pose_net.parameters()}],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = TikTokDataset(args.video_folder, width=args.width, height=args.height, sample_frames=args.num_frames, sample_rate=args.sample_rate)
    # sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, pose_net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, pose_net, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)
        
    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = unet.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

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
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    unet.train()
    pose_net.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(pose_net):
                # first, convert images to latent space.
                tgt_vid = batch["tgt_vid"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                ref_img = batch["ref_img"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )

                latents = tensor_to_vae_latent(tgt_vid, vae)
                ref_latent = vae.encode(ref_img).latent_dist.sample()
                ref_latent = ref_latent.repeat(1, 1, 1, 1)

                image_embedding = image_encoder(batch['clip_img']).image_embeds
                image_embedding = image_embedding.unsqueeze(1)
                bs_embed, seq_len, _ = image_embedding.shape
                image_embedding = image_embedding.repeat(1, 1, 1)
                image_embedding = image_embedding.view(bs_embed * 1, seq_len, -1)
                pose_latents = pose_net(batch['tgt_lmk'])

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                noise_aug_strength = 0.02
                

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                latent_model_input = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning.
                encoder_hidden_states = image_embedding

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    args.fps, # fixed
                    args.motion_bucket_id, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                ref_latents = ref_latent.unsqueeze(
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                input_latents = torch.cat(
                    [latent_model_input, ref_latents], dim=2)

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents
               
                model_pred = unet(
                    input_latents, timesteps, encoder_hidden_states=encoder_hidden_states, 
                    added_time_ids=added_time_ids, pose_latents=pose_latents).sample

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        unet_save_dir = os.path.join(save_path, 'unet')
                        posenet_save_dir = os.path.join(save_path, 'pose_net')
                        if not os.path.exists(unet_save_dir):
                            os.makedirs(unet_save_dir)
                        if not os.path.exists(posenet_save_dir):
                            os.makedirs(posenet_save_dir)
                        #unet_dict = unet.state_dict()
                        pose_dict = pose_net.state_dict()
                        unet.save_pretrained(unet_save_dir)
                        torch.save(pose_dict, os.path.join(posenet_save_dir, 'pose_net.pth'))
                        logger.info(f"Saved state to {save_path}")
                    

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                        # del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone() 
    accelerator.end_training()


if __name__ == "__main__":
    main()
