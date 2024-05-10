import collections
import functools
import os
import pickle
import sys
from datetime import datetime

import cv2
import imageio

if "PyTorch_VAE" not in sys.path:
    sys.path.append("PyTorch_VAE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.utils.data import Dataset
from tqdm.notebook import tqdm, trange

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.model.diffusion import conditional_unet1d
from inverse_dynamics import InverseDynamicsCNN, InverseDynamicsMLP
from state_diffusion import Diffusion, sample
from utils import (
    denormalize_img,
    denormalize_pn1,
    denormalize_standard_normal,
    normalize_pn1,
    normalize_standard_normal,
)
from vae import VanillaVAE

ex = Experiment("pusht_state_rollout")


def create_gif(image_arrays, filename, duration=0.1):
    """
    Create a GIF from a list of NumPy array images.

    Parameters:
        image_arrays (list of np.ndarray): List of NumPy array images.
        filename (str): Filename for the output GIF.
        duration (float): Duration (in seconds) of each frame in the GIF.

    Returns:
        None
    """
    images = []
    for image_array in image_arrays:
        # Ensure that the image array is in uint8 format
        image_array = np.uint8(image_array)
        images.append(image_array)

    # Write the images to a GIF file
    with imageio.get_writer(filename, mode="I", duration=duration) as writer:
        for image in images:
            writer.append_data(image)


def get_exp_dir(save_dir, obs_key, max_steps, inv_dyn_mode, action_horizon, seed):
    tag = f"obs_{obs_key}-steps_{max_steps}-id_mode_{inv_dyn_mode}-action_horizon_{action_horizon}-seed_{seed}"
    return os.path.join(save_dir, tag)


@ex.config
def sacred_config():
    seed = 0

    obs_key = "state"  # state or image

    device = "cuda"
    data_path = (
        "/nas/ucb/ebronstein/lsdp/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )

    # VAE
    vae_path = "models/pusht_vae/vae_32_20240403.pt"
    vae_latent_dim = 32

    # Diffusion
    # Latent image based
    # diffusion_load_dir = "models/diffusion/pusht_unet1d_img_128_256_512_1024_edim_256_obs_8_pred_8_bs_256_lr_0.0003_e_250_ema_norm_latent_uniform/2"
    # State based
    diffusion_load_dir = "models/diffusion/pusht_unet1d_state_128_256_512_1024_edim_256_obs_8_pred_8_bs_256_lr_0.0003_e_100_ema_norm_latent_uniform/1"
    n_obs_history = 8
    n_pred_horizon = 8
    diffusion_step_embed_dim = 256
    normalize_latent = "uniform"
    use_ema_helper = True

    # Inverse dynamics
    # Options: set_state, action_is_diffusion_state, learned
    inv_dyn_mode = "action_is_diffusion_state"
    # CNN-based ID model on images
    # inv_dyn_path = "models/inverse_dynamics/pusht_cnn-img-obs_5-bs_256-lr_0.0001-epochs_10-train_on_recon-False-latent_dim_32/2024-05-06_17-04-08/inverse_dynamics_final.pt"
    # MLP-based ID model on states
    inv_dyn_path = "models/inverse_dynamics/pusht_mlp-state-obs_5-bs_256-lr_0.001-epochs_10-train_on_recon-False/2024-05-08_20-22-18/inverse_dynamics_final.pt"
    inv_dyn_type = "mlp"  # MLP or CNN
    inv_dyn_n_obs_history = 5

    # Inference
    # Limit environment interaction to 200 steps before termination
    max_steps = 200
    action_horizon = 8
    # Number of times to repeat the action at each step.
    n_repeat_action = 2

    # Saving
    root_save_dir = "rollouts/pusht"
    exp_dir = get_exp_dir(
        root_save_dir, obs_key, max_steps, inv_dyn_mode, action_horizon, seed
    )

    # NOTE: set the observers to this single FileStorageObserver instead of
    # appending to ex.observers. This allows running the experiment multiple
    # times without creating multiple observers.
    ex.observers = [FileStorageObserver(exp_dir, copy_sources=False)]


def rollout(
    STATE_DIM,
    max_steps,
    n_pred_horizon,
    n_obs_history,
    action_horizon,
    diffusion,
    obs_key,
    n_repeat_action,
    seed,
    inv_dyn_mode,
    inv_dyn_n_obs_history=None,
    inv_dyn_model=None,
    device="cuda",
):
    # Number of actions predicted.
    pred_horizon = n_pred_horizon
    # Number of observations in the history.
    obs_horizon = n_obs_history

    action_dim = 2

    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    if inv_dyn_mode == "learned" and action_horizon != 1:
        raise ValueError("Only action_horizon=1 is supported.")

    # 0. create env object
    env = PushTImageEnv() if obs_key == "image" else PushTEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(seed)

    # 2. must reset before use
    obs = env.reset()

    # Number of diffusion samples per step for rejection sampling.
    num_samples_per_step = 1

    # keep a queue of last obs_horizon steps of observations
    obs_deque = collections.deque([obs] * n_obs_history, maxlen=obs_horizon)
    # obs_deque = collections.deque([obs], maxlen=obs_horizon)
    # save states and rewards
    states_list = [obs for obs in obs_deque]
    # Visualizations of states.
    imgs_list = [env.render(mode="rgb_array")] * n_obs_history
    # for state in states_list:
    #     env._set_state(state)
    #     env.latest_action = None
    #     imgs_list.append(env.render(mode="rgb_array"))
    rewards = list()
    done = False
    step_idx = 0

    # Future states predicted by the diffusion model at each step.
    pred_states_list = []
    # Observation (state) history at each step.
    obs_history_list = []
    # Actions predicted by the inverse dynamics model at each step.
    actions_list = []

    # Normalization functions.
    denormalize_action = functools.partial(
        denormalize_pn1,
        min_val=env.action_space.low,
        max_val=env.action_space.high,
    )
    normalize_state = functools.partial(
        normalize_pn1,
        min_val=env.observation_space.low,
        max_val=env.observation_space.high,
    )
    denormalize_state = functools.partial(
        denormalize_pn1,
        min_val=env.observation_space.low,
        max_val=env.observation_space.high,
    )

    with tqdm(total=max_steps, desc="Eval PushTEnv") as pbar:
        iter = 0
        while not done:
            # stack the last obs_horizon number of observations (states)
            # Range: [0, 1]
            states = np.stack([obs for obs in obs_deque])  # [obs_horizon, 5]
            assert (states.min(axis=0) >= env.observation_space.low).all() and (
                states.max(axis=0) <= env.observation_space.high
            ).all()
            obs_history_list.append(states)

            # Normalize statest to [-1, 1]
            nstates = normalize_state(states)
            assert nstates.min() >= -1 and nstates.max() <= 1

            # device transfer
            nstates = torch.from_numpy(nstates).to(device, dtype=torch.float32)

            if states.shape[0] < n_obs_history:
                # Sample a random action if we don't have enough observations.
                # action = env.action_space.sample()  # [2,]

                # Move toward the center.
                action = (env.action_space.low + env.action_space.high) / 2
                # Unsqueeze to apply just one action (action_horizon=1).
                action = action[None]  # [1, 2]
            else:
                nstates_repeat = nstates.repeat(num_samples_per_step, 1, 1)
                # Sample from the diffusion model.
                with torch.no_grad():
                    rejection_sample_count = 0
                    while True:
                        # [len(return_steps), num_samples, n_pred_horizon, latent_dim] and
                        # [len(return_steps), num_samples, n_obs_history, latent_dim)]
                        npred_states, _ = sample(
                            diffusion,
                            num_samples=num_samples_per_step,
                            return_steps=[512],
                            data_shape=(n_pred_horizon, STATE_DIM),
                            obs_data=nstates_repeat,
                            obs_normalizer=None,
                            clip=None,
                            clip_noise=(-3, 3),
                            device=device,
                            obs_key=None,  # Unused because obs_data is not a dataloader.
                        )
                        # [num_samples, n_pred_horizon, latent_dim]
                        npred_states = npred_states.squeeze(0)
                        # print("npred_latents:", npred_latents.shape)

                        # Remove samples that are out of the range [-1, 1].
                        mask = ((npred_states >= -1) & (npred_states <= 1)).all(
                            axis=(-1, -2)
                        )
                        # Range: [-1, 1] (for real now)
                        if mask.sum() > 0:
                            # [n_in_range_samples, n_pred_horizon, latent_dim]
                            in_range_npred_states = npred_states[mask]
                            # print("in_range_npred_latents:", in_range_npred_latents.shape)
                            break
                        print(
                            f"Attempt {rejection_sample_count + 1}: no samples in range."
                        )

                        rejection_sample_count += 1

                    assert (
                        in_range_npred_states.min() >= -1
                        and in_range_npred_states.max() <= 1
                    )

                    in_range_pred_states = denormalize_state(in_range_npred_states)
                    pred_states_list.append(in_range_pred_states)

                    # Take the first sample prediction arbitrarily.
                    # [n_pred_horizon, 5]
                    in_range_npred_states = in_range_npred_states[0]
                    # print("in_range_npred_states:", in_range_npred_states.shape)

                    # Run the inverse dynamics model.
                    if inv_dyn_mode == "set_state":
                        # Set the env state directly.
                        next_pred_states = denormalize_state(in_range_npred_states)
                        action = None
                    elif inv_dyn_mode == "action_is_diffusion_state":
                        # [n_pred_horizon, 2]
                        nactions = in_range_npred_states[:, :action_dim]
                        # Denormalize the actions.
                        actions = denormalize_action(nactions)
                        assert (
                            actions.min() >= env.action_space.low[0]
                            and actions.max() <= env.action_space.high[0]
                        )
                    elif inv_dyn_mode == "learned":
                        in_range_npred_states = torch.tensor(
                            in_range_npred_states, dtype=torch.float32, device=device
                        )
                        # Concatenate the last inv_dyn_n_obs_history - 1 observations with
                        # the first predicted observation in order to get the action.
                        # NOTE: this currently only works for action_horizon = 1.
                        nobs_and_npred_states = torch.cat(
                            [
                                nstates[-(inv_dyn_n_obs_history - 1) :],
                                in_range_npred_states[:1],
                            ],
                            dim=0,
                        )
                        # nobs_and_npred = torch.cat([nimages[-(1):], in_range_npred_states[:(4)]], dim=0)
                        # print("nobs_and_npred:", nobs_and_npred.shape)

                        # NOTE: no need to denormalize the image because the inverse dynamics
                        # model takes in a normalized image.
                        # [1, action_dim]
                        nactions = (
                            inv_dyn_model(nobs_and_npred_states.unsqueeze(0))
                            .cpu()
                            .numpy()
                        )
                        assert nactions.min() >= -1 and nactions.max() <= 1
                        # print("nactions:", nactions.shape)
                        # Denormalize the actions.
                        # [1, action_dim]
                        actions = denormalize_action(nactions)
                        assert (
                            actions.min() >= env.action_space.low[0]
                            and actions.max() <= env.action_space.high[0]
                        )
                        # print("actions:", actions.shape)

            action = actions[:action_horizon, :]  # (action_horizon, action_dim)
            actions_list.append(action)

            # Execute action_horizon number of steps without replanning
            if action is not None:
                for i in range(len(action)):
                    # Repeat the action n_repeat_action times, and only record the
                    # observation and reward from the last step.
                    for _ in range(n_repeat_action):
                        # stepping env
                        obs, reward, done, _ = env.step(action[i])

                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs_list.append(env.render(mode="rgb_array"))
                    states_list.append(obs)

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx >= max_steps:
                        done = True
                    if done:
                        break
            elif inv_dyn_mode == "set_state":
                # Set the states directly for each predicted future state.
                for i in range(action_horizon):
                    state = next_pred_states[i]
                    env._set_state(state)
                    env.latest_action = None

                    obs = env._get_obs()
                    # save observations
                    obs_deque.append(obs)
                    imgs_list.append(env.render(mode="rgb_array"))
                    states_list.append(obs)

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    if step_idx >= max_steps:
                        done = True
                    if done:
                        break
            else:
                raise NotImplementedError()

            iter += 1

    # Maximum target coverage.
    score = max(rewards) if rewards else None

    return (
        score,
        states_list,
        imgs_list,
        rewards,
        pred_states_list,
        obs_history_list,
        actions_list,
    )


@ex.automain
def main(_config, _run, _log):
    n_obs_history = _config["n_obs_history"]
    n_pred_horizon = _config["n_pred_horizon"]
    normalize_latent = _config["normalize_latent"]
    diffusion_load_dir = _config["diffusion_load_dir"]
    obs_key = _config["obs_key"]
    device = _config["device"]

    if obs_key == "state":
        STATE_DIM = 5
    elif obs_key == "image":
        STATE_DIM = _config["vae_latent_dim"]
    else:
        raise NotImplementedError()

    # Load the dataset.
    dataset = PushTImageDataset(_config["data_path"])

    # Load the diffusion model.
    if n_pred_horizon == 1:
        down_dims = [128, 256]
    elif n_pred_horizon == 4:
        down_dims = [128, 256, 512]
    elif n_pred_horizon == 8:
        down_dims = [128, 256, 512, 1024]
    else:
        raise NotImplementedError()

    global_cond_dim = STATE_DIM * n_obs_history
    diff_model = conditional_unet1d.ConditionalUnet1D(
        input_dim=STATE_DIM,
        down_dims=down_dims,
        diffusion_step_embed_dim=_config["diffusion_step_embed_dim"],
        global_cond_dim=global_cond_dim,
    ).to(device)

    if obs_key == "image":
        # Make the observation normalizer.
        if normalize_latent == "uniform":
            latent_min = np.load(os.path.join(diffusion_load_dir, "latent_min.npy"))
            latent_max = np.load(os.path.join(diffusion_load_dir, "latent_max.npy"))
            obs_normalizer = functools.partial(
                normalize_pn1,
                min_val=torch.tensor(latent_min, dtype=torch.float32, device=device),
                max_val=torch.tensor(latent_max, dtype=torch.float32, device=device),
            )
        elif normalize_latent == "standard_normal":
            latent_mean = np.load(os.path.join(diffusion_load_dir, "latent_mean.npy"))
            latent_std = np.load(os.path.join(diffusion_load_dir, "latent_std.npy"))
            obs_normalizer = functools.partial(
                normalize_standard_normal,
                mean=torch.tensor(latent_mean, dtype=torch.float32, device=device),
                std=torch.tensor(latent_std, dtype=torch.float32, device=device),
            )
        else:
            raise NotImplementedError()
    else:
        obs_normalizer = None

    diffusion = Diffusion(
        train_data=None,
        test_data=None,
        obs_normalizer=obs_normalizer,
        model=diff_model,
        n_epochs=0,
        device=device,
        use_ema_helper=_config["use_ema_helper"],
    )
    diffusion.load(os.path.join(diffusion_load_dir, "diffusion_model_final.pt"))

    # Load the VAE
    # img_data = (
    #     torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2).float()
    # )
    # N, C, H, W = img_data.shape
    # vae = VanillaVAE(in_channels=C, in_height=H, in_width=W, latent_dim=vae_latent_dim).to(
    #     device
    # )
    # vae.load_state_dict(torch.load(vae_path))

    # Load the inverse dynamics model.
    # hidden_dims = None
    # if obs_key == "image":
    #     assert inv_dyn_type == "cnn"
    #     N, H, W, C = dataset.replay_buffer["img"].shape
    #     N, action_dim = dataset.replay_buffer["action"].shape
    #     inv_dyn_model = InverseDynamicsCNN(
    #         C, H, W, action_dim, inv_dyn_n_obs_history, hidden_dims=hidden_dims
    #     ).to(device)
    # else:
    #     inv_dyn_model = InverseDynamicsMLP(
    #         n_obs=inv_dyn_n_obs_history,
    #         obs_dim=STATE_DIM,
    #         action_dim=2,
    #         hidden_dims=[256, 256, 256],
    #     ).to(device)

    # inv_dyn_model.load_state_dict(torch.load(inv_dyn_path))

    (
        score,
        states_list,
        imgs_list,
        rewards,
        pred_states_list,
        obs_history_list,
        actions_list,
    ) = rollout(
        STATE_DIM,
        _config["max_steps"],
        n_pred_horizon,
        n_obs_history,
        _config["action_horizon"],
        diffusion,
        obs_key,
        _config["n_repeat_action"],
        _config["seed"],
        _config["inv_dyn_mode"],
        inv_dyn_n_obs_history=_config["inv_dyn_n_obs_history"],
        inv_dyn_model=None,
        device=device,
    )
    _log.info("Score: %.4f" % score)

    # Run directory.
    run_id = _run._id
    run_dir = os.path.join(_config["exp_dir"], run_id)
    save_dir = run_dir
    _log.info(f"Saving to {save_dir}")

    # Save GIF.
    gif_filepath = os.path.join(save_dir, "rollout.gif")
    create_gif(imgs_list, gif_filepath, duration=0.5)

    # Save rollout information to pickle file.
    rollout_data = {
        "score": score,
        "states": states_list,
        "rewards": rewards,
        "pred_states": pred_states_list,
        "obs_history": obs_history_list,
        "actions": actions_list,
    }
    info_filepath = os.path.join(save_dir, "data.pkl")
    with open(info_filepath, "wb") as f:
        pickle.dump(rollout_data, f)
