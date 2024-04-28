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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm, trange

import wandb
from diffusion_policy.common.pytorch_util import compute_conv_output_shape
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.model.diffusion import conditional_unet1d
from ema import EMAHelper

# Custom imports
from PyTorch_VAE import models
from lsdp_utils.Diffusion import Diffusion
from lsdp_utils.VanillaVAE import VanillaVAE
from lsdp_utils.EpisodeDataset import EpisodeDataset, EpisodeDataloaders
from lsdp_utils.utils import plot_losses, plot_samples

from types import SimpleNamespace

# cfg_latent = SimpleNamespace(batch_size=1024, )

# "/nas/ucb/ebronstein/lsdp/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
# "/home/tsadja/data_diffusion/pusht/pusht_cchi_v7_replay.zarr"

cfg = SimpleNamespace(dataset_path='/home/matteogu/ssd_data/data_diffusion/pusht/pusht_cchi_v7_replay.zarr',
                      # vae_model_path='/nas/ucb/ebronstein/lsdp/models/pusht_vae/vae_32_20240403.pt',
                      vae_model_path='/home/matteogu/Desktop/prj_deepul/repo_online/lsdp/models/pusht_vae/vae_32_20240403.pt',
                      batch_size=4096,  # 3.8 Giga for state, better 512 for latents
                      n_obs_history=8,
                      n_pred_horizon=8,
                      diffusion_step_embed_dim=128,  # in the original paper was 256
                      lr=3e-4,  # optimization params
                      epochs=200,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),

                      VAE_ENABLED=False,
                      )


# Make the episode dataset and create a DataLoader.
# this works
# # batch_size, n_obs_history, n_pred_horizon = 1024, 8, 8
# batch_size = 4096  # 3.8G
# n_obs_history = 8
# n_pred_horizon = 8
#

def normalize_pn1(x, min_val, max_val):
    # Normalize to [0, 1]
    nx = (x - min_val) / (max_val - min_val)
    # Normalize to [-1, 1]
    return nx * 2 - 1


def denormalize_pn1(nx, min_val, max_val):
    # Denormalize from [-1, 1]
    x = (nx + 1) / 2
    # Denormalize from [0, 1]
    return x * (max_val - min_val) + min_val


def train_diffusion():
    dataset = PushTImageDataset(cfg.dataset_path)
    full_dataset = torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2)
    N, C, H, W = full_dataset.shape

    # Make the state normalizer.
    max_state = dataset.replay_buffer["state"].max(axis=0)
    min_state = dataset.replay_buffer["state"].min(axis=0)

    if cfg.VAE_ENABLED:
        # Load VAE.
        latent_dim = 32
        vae_model = VanillaVAE(
            in_channels=3, in_height=H, in_width=W, latent_dim=latent_dim
        ).to(cfg.device)
        vae_model.load_state_dict(torch.load(cfg.vae_model_path))
        include_keys = ["img"]
        cfg.STATE_DIM = latent_dim
        obs_key = "img"

        normalize_encoder_input = functools.partial(
            get_latent, vae_model=vae_model, device=cfg.device
        )
    else:
        include_keys = ["state"]
        cfg.STATE_DIM = 5
        obs_key = "state"
        normalize_encoder_input = None

    # Make train and val loaders
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, 0.1)
    val_idxs = np.where(val_mask)[0]
    train_idxs = np.where(~val_mask)[0]

    def get_latent(x, vae_model, device):
        x = x / 255.0
        x = 2 * x - 1
        return vae_model.encode(torch.from_numpy(x).to(device))[0].detach()



    state_normalizer = functools.partial(
        normalize_pn1, min_val=min_state, max_val=max_state
    )

    process_fns = {"state": state_normalizer, "img": normalize_encoder_input}

    print("Making datasets and dataloaders.")
    train_loader, val_loader = EpisodeDataloaders(dataset=dataset,
                                                  episode_train_idxs=train_idxs,
                                                  episode_val_idxs=val_idxs,
                                                  include_keys=include_keys,
                                                  process_fns=process_fns,
                                                  cfg=cfg)  # configuration params

    global_cond_dim = cfg.STATE_DIM * cfg.n_obs_history

    diff_model = conditional_unet1d.ConditionalUnet1D(
        input_dim=cfg.STATE_DIM,
        down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        global_cond_dim=global_cond_dim,
    ).to(cfg.device)

    optim_kwargs = dict(lr=cfg.lr)
    diffusion = Diffusion(
        train_data=train_loader,
        test_data=val_loader,
        model=diff_model,
        n_epochs=cfg.epochs,
        optim_kwargs=optim_kwargs,
        device=cfg.device,
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
        name = 'pusht'

        save_dir = f"models/diffusion/{name}"
        if save_dir is not None:
            # Get the current timestamp and save it as a new directory.
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(save_dir)

        train_losses, test_losses = diffusion.train(
            wandb_run=wandb_run, save_freq=30, save_dir=save_dir, obs_key=obs_key
        )


def train_diffusion_vae_latent():
    dataset = PushTImageDataset(cfg.dataset_path)
    full_dataset = torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2)
    N, C, H, W = full_dataset.shape
    # Make the state normalizer.
    max_state = dataset.replay_buffer["state"].max(axis=0)
    # min_state = np.zeros_like(max_state)
    min_state = dataset.replay_buffer["state"].min(axis=0)

    # Load VAE.
    latent_dim = 32
    vae_model = VanillaVAE(
        in_channels=3, in_height=H, in_width=W, latent_dim=latent_dim
    ).to(cfg.device)
    vae_model.load_state_dict(torch.load(cfg.vae_model_path))

    # Make train and val loaders
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, 0.1)
    val_idxs = np.where(val_mask)[0]
    train_idxs = np.where(~val_mask)[0]

    # Make the episode dataset and create a DataLoader.
    cfg.batch_size = 512

    def get_latent(x, vae_model, device):
        x = x / 255.0
        x = 2 * x - 1
        return vae_model.encode(torch.from_numpy(x).to(device))[0].detach()

    normalize_encoder_input = functools.partial(
        get_latent, vae_model=vae_model, device=device
    )

    # process_fns = {"state": state_normalizer, "img": normalize_encoder_input}
    process_fns = {"img": normalize_encoder_input}
    include_keys = ["img"]

    print("Making datasets and dataloaders.")
    train_loader, val_loader = prepareDatasets(dataset=dataset,
                                               episode_train_idxs=train_idxs,
                                               episode_val_idxs=val_idxs,
                                               include_keys=include_keys,
                                               process_fns=process_fns,
                                               cfg=cfg)  # configuration params

    # Set to latent_dim if diffusing in the VAE latent space and to 5 if diffusing in the state space.
    STATE_DIM = latent_dim

    global_cond_dim = STATE_DIM * n_obs_history
    diff_model = conditional_unet1d.ConditionalUnet1D(
        input_dim=STATE_DIM,
        down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=128,
        global_cond_dim=global_cond_dim,
    ).to(device)

    optim_kwargs = dict(lr=cfg.lr)
    diffusion = Diffusion(
        train_data=train_loader,
        test_data=val_loader,
        model=diff_model,
        n_epochs=cfg.epochs,
        optim_kwargs=optim_kwargs,
        device=device,
    )

    obs_key = "img"

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
        name = f"pusht-1dconv_latent_128_256_512_1024-obs_8-pred_8"
        save_dir = f"models/diffusion/{name}"
        if save_dir is not None:
            # Get the current timestamp and save it as a new directory.
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(save_dir)

        train_losses, test_losses = diffusion.train(
            wandb_run=wandb_run, save_freq=30, save_dir=save_dir, obs_key=obs_key
        )

    return_steps = [512]
    num_samples = 1000
    print("Sampling.")
    normalized_samples = sample(
        diffusion,
        num_samples=num_samples,
        return_steps=return_steps,
        data_shape=(n_obs_history, STATE_DIM),
        data_loader=train_loader,
        clip=None,
        clip_noise=(-3, 3),
        device=device,
    )

    in_range_normalized_samples = []
    for num_steps in range(normalized_samples.shape[0]):
        mask = ((normalized_samples >= -1) & (normalized_samples <= 1)).all(
            axis=(-2, -1)
        )
        in_range_normalized_samples.append(normalized_samples[mask])

    # [len(return_steps), num_in_range_samples, n_obs_history, dim=5]
    in_range_samples = np.array(
        [denormalize_pn1(s, min_state, max_state) for s in in_range_normalized_samples]
    )
    samples = np.array(
        [denormalize_pn1(s, min_state, max_state) for s in normalized_samples]
    )

    # Save samples.
    np.save(os.path.join(save_dir, f"train_samples_{num_samples}.npy"), samples)

    # Plot samples histogram.
    # n_data, state_dim = dataset.replay_buffer[obs_key].shape
    # plot_samples(
    #     in_range_samples,
    #     np.broadcast_to(
    #         dataset.replay_buffer[obs_key][:, None], [n_data, n_obs_history, state_dim]
    #     ),
    #     return_steps,
    #     save_dir=save_dir,
    # )

    # diff_states = []
    # for obs_history, pred_horizon in train_loader:
    #     norm_state = (
    #         obs_history[obs_key].detach().cpu().numpy()
    #     )  # [batch_size, n_history, dim]
    #     state = denormalize_pn1(
    #         norm_state, min_state, max_state
    #     )  # [batch_size, n_history, dim]
    #     diff_states.append(np.diff(state, axis=1))  # [batch_size, n_history - 1, dim]

    # diff_states = np.concatenate(diff_states, axis=0)  # [n_samples, n_history - 1, dim]
    # diff_samples = np.diff(
    #     in_range_samples, axis=2
    # )  # [len(return_steps), num_in_range_samples, n_history - 1, dim]

    # plot_samples(diff_samples, diff_states, return_steps, save_dir=save_dir)


def train_diffusion_state():
    print("Loading dataset.")

    dataset = PushTImageDataset(cfg.dataset_path)
    full_dataset = torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2)
    # Make the state normalizer.
    max_state = dataset.replay_buffer["state"].max(axis=0)
    # min_state = np.zeros_like(max_state)
    min_state = dataset.replay_buffer["state"].min(axis=0)

    # Make train and val loaders
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, 0.1)
    val_idxs = np.where(val_mask)[0]
    train_idxs = np.where(~val_mask)[0]

    state_normalizer = functools.partial(
        normalize_pn1, min_val=min_state, max_val=max_state
    )

    process_fns = {"state": state_normalizer}
    include_keys = ["state"]

    print("Making datasets and dataloaders.")

    train_loader, val_loader = prepareDatasets(dataset=dataset,
                                               episode_train_idxs=train_idxs,
                                               episode_val_idxs=val_idxs,
                                               include_keys=include_keys,
                                               process_fns=process_fns,
                                               cfg=cfg)  # configuration params

    global_cond_dim = cfg.STATE_DIM * cfg.n_obs_history
    diff_model = conditional_unet1d.ConditionalUnet1D(
        input_dim=cfg.STATE_DIM,
        down_dims=[128, 256, 512, 1024],
        global_cond_dim=global_cond_dim,
    ).to(device)

    optim_kwargs = dict(lr=cfg.lr)
    diffusion = Diffusion(
        train_data=train_loader,
        test_data=val_loader,
        model=diff_model,
        n_epochs=cfg.epochs,
        optim_kwargs=optim_kwargs,
        device=device,
    )

    wandb_run = None
    # wandb_run = wandb.init(project="state_1dconv_latent", name=name, reinit=True)

    obs_key = "state"

    # Load the model.
    # load_dir = "models/diffusion/pusht-1dconv_state_128_256_512_1024-obs_8-pred_8/2024-04-27_22-07-27"
    load_dir = None
    if load_dir is not None:
        diffusion.load(os.path.join(load_dir, "diffusion_model_final.pt"))
        train_losses = np.load(os.path.join(load_dir, "train_losses.npy"))
        test_losses = np.load(os.path.join(load_dir, "test_losses.npy"))
        save_dir = load_dir
    else:
        # Save directory.
        name = f"pusht-1dconv_state_128_256_512_1024-obs_8-pred_8"
        save_dir = f"models/diffusion/{name}"
        if save_dir is not None:
            # Get the current timestamp and save it as a new directory.
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(save_dir)

        train_losses, test_losses = diffusion.train(
            wandb_run=wandb_run, save_freq=30, save_dir=save_dir, obs_key=obs_key
        )

    return_steps = [512]
    num_samples = 1000
    print("Sampling.")
    normalized_samples = sample(
        diffusion,
        num_samples=num_samples,
        return_steps=return_steps,
        data_shape=(cfg.n_obs_history, cfg.STATE_DIM),
        data_loader=train_loader,
        clip=None,
        clip_noise=(-3, 3),
        device=cfg.device,
    )

    in_range_normalized_samples = []
    for num_steps in range(normalized_samples.shape[0]):
        mask = ((normalized_samples >= -1) & (normalized_samples <= 1)).all(
            axis=(-2, -1)
        )
        in_range_normalized_samples.append(normalized_samples[mask])

    # [len(return_steps), num_in_range_samples, n_obs_history, dim=5]
    in_range_samples = np.array(
        [denormalize_pn1(s, min_state, max_state) for s in in_range_normalized_samples]
    )
    samples = np.array(
        [denormalize_pn1(s, min_state, max_state) for s in normalized_samples]
    )

    # Save samples.
    np.save(os.path.join(save_dir, f"train_samples_{num_samples}.npy"), samples)

    # Plot samples histogram.
    n_data, state_dim = dataset.replay_buffer[obs_key].shape
    plot_samples(
        in_range_samples,
        np.broadcast_to(
            dataset.replay_buffer[obs_key][:, None], [n_data, n_obs_history, state_dim]
        ),
        return_steps,
        save_dir=save_dir,
    )

    diff_states = []
    for obs_history, pred_horizon in train_loader:
        norm_state = (
            obs_history[obs_key].detach().cpu().numpy()
        )  # [batch_size, n_history, dim]
        state = denormalize_pn1(
            norm_state, min_state, max_state
        )  # [batch_size, n_history, dim]
        diff_states.append(np.diff(state, axis=1))  # [batch_size, n_history - 1, dim]

    diff_states = np.concatenate(diff_states, axis=0)  # [n_samples, n_history - 1, dim]
    diff_samples = np.diff(
        in_range_samples, axis=2
    )  # [len(return_steps), num_in_range_samples, n_history - 1, dim]

    plot_samples(diff_samples, diff_states, return_steps, save_dir=save_dir)


if __name__ == "__main__":
    # train_diffusion_vae_latent()
    # train_diffusion_state()
    train_diffusion()
