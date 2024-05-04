import collections
import copy
import datetime
import functools
import math
import os
import sys
import time
from typing import Callable, Optional

if "PyTorch_VAE" not in sys.path:
    sys.path.append("PyTorch_VAE")

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm, trange

import wandb
from diffusion_policy.common.pytorch_util import compute_conv_output_shape

# from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.model.diffusion import conditional_unet1d
from ema import EMAHelper
from lsdp_utils.Diffusion import Diffusion, DiffusionMLP
from lsdp_utils.EpisodeDataset import EpisodeDataloaders, EpisodeDataset
from lsdp_utils.utils import (
    LatentsToStateMLP,
    bcolors,
    denormalize_pn1,
    normalize_pn1,
    plot_losses,
    plot_samples,
)
from lsdp_utils.VanillaVAE import VanillaVAE

# Custom imports
from PyTorch_VAE import models

# cfg_latent = SimpleNamespace(batch_size=1024, )

# "/nas/ucb/ebronstein/lsdp/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
# "/home/tsadja/data_diffusion/pusht/pusht_cchi_v7_replay.zarr"

cfg = SimpleNamespace(
    dataset_path="/home/matteogu/ssd_data/data_diffusion/pusht/pusht_cchi_v7_replay.zarr",
    # vae_model_path='/nas/ucb/ebronstein/lsdp/models/pusht_vae/vae_32_20240403.pt',
    # vae_model_path='/home/matteogu/Desktop/prj_deepul/repo_online/lsdp/models/pusht_vae/vae_32_20240403.pt',
    vae_model_path="/home/matteogu/ssd_data/diffusion_models/models/vae/"
    "pusht_vae_klw_1.00e-07_ldim_128_bs_512_epochs_100_lr_0.0005_hdim_"
    "32_64_128_256_512/vae_99.pt",
    save_dir="/home/matteogu/ssd_data/diffusion_models/models/diffusion/",
    batch_size=8000,  # 3.8 Giga for state, better 512 for latents
    # batch_size=64,  # 3.8 Giga for state, better 512 for latents
    n_obs_history=0,  # if it is 0, it means unconditional generation
    n_pred_horizon=4,
    down_dims=[512, 1024],  # 512, 1024, 1024, 2048
    diffusion_step_embed_dim=256,  # in the original paper was 256
    lr=1e-3,  # optimization params
    epochs=300,
    n_warmup_steps=200,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    obs_key="img",
    latents_to_state=True,
)

assert cfg.obs_key == "img" or cfg.obs_key == "state"
print(f"{bcolors.OKGREEN}Hyperparamters of the current run:{bcolors.ENDC}")
print(cfg.__dict__)
print(f"{bcolors.OKGREEN} ---------------------- {bcolors.ENDC}")

# Make the episode dataset and create a DataLoader.
# this works
# # batch_size, n_obs_history, n_pred_horizon = 1024, 8, 8
# batch_size = 4096  # 3.8G
# n_obs_history = 8
# n_pred_horizon = 8


def train_diffusion():
    dataset = PushTImageDataset(cfg.dataset_path)
    full_dataset = torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2)
    N, C, H, W = full_dataset.shape

    # Make the state normalizer.
    max_state = dataset.replay_buffer["state"].max(axis=0)
    min_state = dataset.replay_buffer["state"].min(axis=0)

    if cfg.obs_key == "img":
        # Load VAE.
        # latent_dim = 32
        latent_dim = 128
        vae_model = VanillaVAE(
            in_channels=C,
            in_height=H,
            in_width=W,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128, 256, 512],
        ).to(cfg.device)
        vae_model.load_state_dict(torch.load(cfg.vae_model_path))
        cfg.STATE_DIM = latent_dim

        def get_latent(x, vae_model, device):
            x = x / 255.0
            x = 2.0 * x - 1.0
            return vae_model.encode(torch.from_numpy(x).to(device))[0].detach()

        normalize_encoder_input = functools.partial(
            get_latent, vae_model=vae_model, device=cfg.device
        )

        if cfg.latents_to_state is not None:
            # these two go together.
            model_LatentsToStateMLP = LatentsToStateMLP(
                in_dim=128, out_dim=5, hidden_dims=[256, 256, 128, 16]  # state
            ).to(cfg.device)
            mlp_path = "/home/matteogu/ssd_data/diffusion_models/models/latent_to_state/mlp_128to5.pt"
            model_LatentsToStateMLP.load_state_dict(torch.load(mlp_path))
            cfg.STATE_DIM = 5
        else:
            model_LatentsToStateMLP = None
    else:
        cfg.STATE_DIM = 5
        normalize_encoder_input = None
        model_LatentsToStateMLP = None

    state_normalizer = functools.partial(
        normalize_pn1, min_val=min_state, max_val=max_state
    )

    process_fns = {"state": state_normalizer, "img": normalize_encoder_input}

    print("Making datasets and dataloaders.")
    train_loader, val_loader = EpisodeDataloaders(
        dataset=dataset,
        include_keys=[cfg.obs_key],  # one key only
        process_fns=process_fns,
        cfg=cfg,
        val_ratio=0.9,
    )  # configuration params

    global_cond_dim = cfg.STATE_DIM * cfg.n_obs_history

    diff_model = conditional_unet1d.ConditionalUnet1D(
        input_dim=cfg.STATE_DIM,
        down_dims=cfg.down_dims,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        global_cond_dim=global_cond_dim,
    ).to(cfg.device)

    # diff_model = None  # load MLP baseline
    # diff_model = DiffusionMLP()

    optim_kwargs = dict(lr=cfg.lr)
    diffusion = Diffusion(
        train_data=train_loader,
        test_data=val_loader,
        model=diff_model,
        n_epochs=cfg.epochs,
        n_warmup_steps=cfg.n_warmup_steps,
        optim_kwargs=optim_kwargs,
        device=cfg.device,
        mlp_nograd_latents_to_state=model_LatentsToStateMLP,
    )

    wandb_run = None
    # wandb_run = wandb.init(project="state_1dconv_latent", name=name, reinit=True)

    # Load the model.
    load_dir = None
    # load_dir = "models/diffusion/pusht-1dconv_state_128_256_512_1024-obs_8-pred_8/2024-04-27_22-07-27"
    if load_dir is not None:
        diffusion.load(os.path.join(load_dir, "diffusion_model_final.pt"))
        train_losses = np.load(os.path.join(load_dir, "train_losses.npy"))
        test_losses = np.load(os.path.join(load_dir, "test_losses.npy"))
        save_dir = load_dir
    else:
        # Save directory.
        # name = f"pusht-1dconv_latent_128_256_512_1024-obs_8-pred_8"
        # name = f"pusht-1dconv_state_128_256_512_1024-obs_8-pred_8"
        name = (
            f'pusht_unet1d_{cfg.obs_key}_{str(cfg.down_dims)[1:-1].replace(", ", "_")}_edim_{cfg.diffusion_step_embed_dim}'
            f"obs_{cfg.n_obs_history}_pred_{cfg.n_pred_horizon}_bs_{cfg.batch_size}_lr_{cfg.lr}_e_{cfg.epochs}"
        )
        if diff_model is None:
            name = "MLP_Baseline"

        save_dir = f"{cfg.save_dir}{name}"
        if save_dir is not None:
            # Get the current timestamp and save it as a new directory.
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(save_dir)

        train_losses, test_losses = diffusion.train(
            wandb_run=wandb_run, save_freq=30, save_dir=save_dir, obs_key=cfg.obs_key
        )


if __name__ == "__main__":
    # train_diffusion_vae_latent()
    # train_diffusion_state()
    train_diffusion()
