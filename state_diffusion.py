import collections
import copy
import datetime
import functools
import math
import os
import sys
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
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.model.diffusion import conditional_unet1d
from ema import EMAHelper
from PyTorch_VAE import models


def plot_losses(train_losses, test_losses, save_dir=None):
    # Plot train and test losses.
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(
        np.linspace(0, len(train_losses), len(test_losses)),
        test_losses,
        label="Test Loss",
    )
    # Remove outliers for better visualization
    # plt.ylim(0, 0.01)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "train_test_losses.png"))
    else:
        plt.show()


def plot_samples(samples, data, save_dir=None):
    # Plot histogram of each dimension of the samples
    for i, steps in enumerate(return_steps):
        step_samples = samples[i]  # [num_samples, n_history, dim]
        n_history = step_samples.shape[1]
        dim = step_samples.shape[2]

        fig, ax = plt.subplots(
            n_history, dim, figsize=(12, n_history * 4), squeeze=False
        )

        for i in range(n_history):
            for j in range(dim):

                ax[i, j].hist(
                    step_samples[:, i, j],
                    density=True,
                    bins=50,
                    color="blue",
                    alpha=0.5,
                )
                ax[i, j].hist(
                    data[:, i, j],
                    density=True,
                    bins=50,
                    color="green",
                    alpha=0.5,
                )
                ax[i, j].set_title(f"History {i}, Dim {j}")

        fig.suptitle(f"{steps} steps")
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"histogram_{steps}.png"))
        else:
            plt.show()


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


class EpisodeDataset(Dataset):
    def __init__(
        self,
        dataset,
        n_obs_history=1,
        n_pred_horizon=1,
        episode_idxs=None,
        include_keys: Optional[list[str]] = None,
        process_fns: Optional[dict[str, Callable]] = None,
        device: str = "cpu",
    ):
        """
        Initialize the dataset with the main dataset object that contains
        the replay_buffer. Also, specify the lengths of observation history
        and prediction horizon.
        """
        self.dataset = dataset
        self.n_obs_history = n_obs_history
        self.n_pred_horizon = n_pred_horizon
        self.episode_idxs = list(episode_idxs)
        self.include_keys = set(include_keys) if include_keys is not None else None
        if not self.include_keys:
            raise ValueError("At least one key must be included in the dataset.")
        self.process_fns = process_fns
        self.device = device
        self.prepare_data()

    def prepare_data(self):
        """
        Preprocess the episodes to create a flat list of samples.
        Each sample is a tuple of dictionaries: (obs_history, pred_horizon).
        """
        self.samples = []

        if self.episode_idxs is None:
            self.episode_idxs = range(self.dataset.replay_buffer.n_episodes)

        for episode_idx in tqdm(self.episode_idxs, desc="Preparing data"):
            episode = self.dataset.replay_buffer.get_episode(episode_idx)

            obs = {}

            if self.include_keys is None or "img" in self.include_keys:
                img = episode["img"].transpose(0, 3, 1, 2)  # CHW format
                if "img" in self.process_fns:
                    img = self.process_fns["img"](img)
                obs["img"] = torch.tensor(img, dtype=torch.float32).to(self.device)

            if self.include_keys is None or "action" in self.include_keys:
                action = episode["action"]
                if "action" in self.process_fns:
                    action = self.process_fns["action"](action)
                obs["action"] = torch.tensor(action, dtype=torch.float32).to(
                    self.device
                )

            if self.include_keys is None or "state" in self.include_keys:
                state = episode["state"]
                if "state" in self.process_fns:
                    state = self.process_fns["state"](state)
                obs["state"] = torch.tensor(state, dtype=torch.float32).to(self.device)

            # Iterate through the episode to create samples with observation history and prediction horizon
            n_obs = len(list(obs.values())[0])
            for i in range(n_obs - self.n_obs_history - self.n_pred_horizon + 1):
                obs_history = {}
                pred_horizon = {}

                for key, value in obs.items():
                    obs_history[key] = value[i : i + self.n_obs_history]
                    pred_horizon[key] = value[
                        i
                        + self.n_obs_history : i
                        + self.n_obs_history
                        + self.n_pred_horizon
                    ]

                self.samples.append((obs_history, pred_horizon))

    def __len__(self):
        """
        Return the total number of samples across all episodes.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return the idx-th sample from the dataset.
        """
        obs_history, pred_horizon = self.samples[idx]

        # Convert data to PyTorch tensors and ensure the data type is correct
        # for key, value in obs_history.items():
        #     obs_history[key] = torch.tensor(value, dtype=torch.float32)
        # for key, value in pred_horizon.items():
        #     pred_horizon[key] = torch.tensor(value, dtype=torch.float32)

        return obs_history, pred_horizon


class DiffusionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]):
        super(DiffusionMLP, self).__init__()

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """Run a forward pass.

        Args:
            x (torch.Tensor): Data with shape [batch_size, n_history, input_dim].
            t (torch.Tensor): Time embedding with shape [batch_size].
        """
        flat_x = x.flatten(1)  # [batch_size, n_history * input_dim]
        t = t.reshape(-1, 1)
        xt = torch.cat([flat_x, t], dim=1)
        out = self.net(xt)
        out = out.reshape(x.shape)
        return out


def warmup_cosine_decay_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a scheduler with warmup followed by cosine decay.

    Args:
        optimizer: Optimizer linked to the model parameters.
        warmup_steps: Number of steps for the warmup phase.
        total_steps: Total number of steps in the training.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def dropout_classes(y, null_class, dropout_prob=0.1):
    """Randomly dropout classes with a given probability."""
    dropout_mask = torch.rand(y.shape) < dropout_prob
    y[dropout_mask] = null_class
    return y


class Diffusion(object):
    def __init__(
        self,
        train_data=None,
        test_data=None,
        model=None,
        batch_size=1024,
        n_epochs=100,
        n_warmup_steps=100,
        has_labels=False,
        obs_normalizer=None,
        optim_kwargs: dict = None,
        device: str = "cuda",
        use_ema_helper=False,
    ):
        if has_labels:
            raise NotImplementedError()

        self.n_epochs = n_epochs
        self.has_labels = has_labels
        self.device = device

        self.obs_normalizer = (
            obs_normalizer if obs_normalizer is not None else lambda x: x
        )

        # Data loaders
        if isinstance(train_data, torch.utils.data.DataLoader):
            assert isinstance(test_data, torch.utils.data.DataLoader)
            self.train_loader = train_data
            self.test_loader = test_data
            train_data_shape = None
        elif train_data is not None:
            assert test_data is not None
            train_data_shape = train_data.shape
            self.train_loader, self.test_loader = self.create_loaders(
                train_data, test_data, batch_size
            )
        else:
            self.train_loader = None
            self.test_loader = None

        if model is None:
            assert train_data_shape is not None and len(train_data_shape) == 2
            input_dim = train_data_shape[1]
            self.model = DiffusionMLP(input_dim, input_dim)
        else:
            self.model = model
        self.model = self.model.to(device)

        def model_with_labels(x, labels, t, **kwargs):
            return self.model(x, labels, t, **kwargs)

        def model_without_labels(x, labels, t):
            return self.model(x, t)

        if has_labels:
            self.model_fn = model_with_labels
        else:
            self.model_fn = model_without_labels

        # Optimizer
        optim_kwargs = optim_kwargs or {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optim_kwargs)

        # LR scheduler
        if self.train_loader is not None:
            n_iters_per_epoch = len(self.train_loader)
            n_iters = n_epochs * n_iters_per_epoch
            self.scheduler = warmup_cosine_decay_scheduler(
                self.optimizer, n_warmup_steps, n_iters
            )
        else:
            self.scheduler = None

        self.use_ema_helper = use_ema_helper
        if self.use_ema_helper:
            self.ema_helper = EMAHelper(model=copy.deepcopy(self.model), power=3 / 4)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def create_loaders(self, train_data, test_data, batch_size):
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def get_alpha(self, t):
        return torch.cos(np.pi / 2 * t).to(self.device)

    def get_sigma(self, t):
        return torch.sin(np.pi / 2 * t).to(self.device)

    def compute_loss(self, x, labels=None):
        batch_size = x.shape[0]

        # Step 1: Sample diffusion timestep uniformly in [0, 1]
        t = torch.rand(batch_size, device=self.device)  # [batch_size]

        # Step 2: Compute noise-strength
        alpha_t = self.get_alpha(t)
        sigma_t = self.get_sigma(t)

        # Step 3: Apply forward process
        epsilon = torch.randn_like(x, device=self.device)
        exp_shape = [batch_size] + [1] * (len(x.shape) - 1)
        alpha_t = alpha_t.view(exp_shape)
        sigma_t = sigma_t.view(exp_shape)
        # Print shapes
        # print("x:", x.shape)
        # print("alpha_t:", alpha_t.shape)
        # print("sigma_t:", sigma_t.shape)
        # print("epsilon:", epsilon.shape)
        x_t = alpha_t * x + sigma_t * epsilon  # x.shape

        # Step 4: Estimate epsilon
        eps_hat = self.model_fn(x_t, labels, t)
        # print("eps_hat:", eps_hat.shape)

        # Step 5: Optimize the loss
        loss = (epsilon - eps_hat).pow(2).mean()
        return loss

    def eval(self, test_loader, obs_key: str = "state"):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x in test_loader:
                if self.has_labels:
                    raise NotImplementedError()
                    x, labels = x
                    labels = labels.to(self.device)
                else:
                    labels = None
                obs_history, pred_horizon = x
                obs = obs_history[obs_key].to(self.device)
                obs = self.obs_normalizer(obs)

                loss = self.compute_loss(obs, labels)
                total_loss += loss.item() * obs.shape[0]

        return total_loss / len(test_loader.dataset)

    def train(
        self,
        log_freq=100,
        save_freq: int = 10,
        obs_key: str = "state",
        process_labels_fn=None,
        save_dir=None,
        wandb_run=None,
    ):
        if wandb_run is not None:
            log_freq = None
        if save_dir is not None:
            # Get the current timestamp and save it as a new directory.
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, timestamp)
            os.makedirs(save_dir)

        train_losses = []
        test_losses = [self.eval(self.test_loader, obs_key=obs_key)]

        iter = 0
        for epoch in range(self.n_epochs):
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch + 1})
            epoch_train_losses = []
            # grad_norms = []
            self.model.train()

            for x in self.train_loader:
                if self.has_labels:
                    raise NotImplementedError()
                    x, labels = x
                    labels = labels.to(self.device)
                    if process_labels_fn is not None:
                        labels = process_labels_fn(labels)
                else:
                    labels = None

                obs_history, pred_horizon = x
                obs = obs_history[obs_key].to(self.device)
                # print("obs:", obs.shape)
                obs = self.obs_normalizer(obs)
                # print("obs:", obs.shape)

                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, labels)
                loss.backward()

                # Compute the norm of gradients
                total_norm = 0
                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                if wandb_run is not None:
                    wandb_run.log({"grad_norm": total_norm})
                    wandb_run.log({"batch_loss": loss.item()})
                # grad_norms.append(total_norm)

                self.optimizer.step()
                self.scheduler.step()

                if self.use_ema_helper:
                    self.ema_helper.step(self.model)

                epoch_train_losses.append(loss.item())

                if log_freq is not None and iter % log_freq == 0:
                    print(f"Epoch {epoch+1}, iter {iter}, Loss: {loss.item()}")

                iter += 1

            train_losses.extend(epoch_train_losses)
            test_losses.append(self.eval(self.test_loader, obs_key=obs_key))

            if save_dir is not None and epoch % save_freq == 0:
                self.save(os.path.join(save_dir, f"diffusion_model_epoch_{epoch}.pt"))

        if save_dir is not None:
            self.save(os.path.join(save_dir, "diffusion_model_final.pt"))
            np.save(os.path.join(save_dir, "train_losses.npy"), train_losses)
            np.save(os.path.join(save_dir, "test_losses.npy"), test_losses)

        return train_losses, test_losses


def ddpm_update(
    x,
    eps_hat,
    alpha_t,
    alpha_tm1,
    sigma_t,
    sigma_tm1,
    clip=None,
    clip_noise=None,
    device: str = "cuda",
):
    # assert not torch.isnan(eps_hat).any()
    # assert not torch.isnan(sigma_t).any()
    # assert alpha_t.abs().min() > 1e-6
    if torch.isnan(eps_hat).any():
        print("nan eps_hat")
    if torch.isnan(sigma_t).any():
        print("nan sigma_t")
    if alpha_t.abs().min() < 1e-6:
        print("alpha_t is too small")
    eta_t = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t.pow(2) / alpha_tm1.pow(2))
    # assert not torch.isnan(eta_t).any()
    if torch.isnan(eta_t).any():
        print("nan eta_t")
    x_tm1_mean = (x - sigma_t * eps_hat) / alpha_t
    # assert not torch.isnan(x_tm1_mean).any()
    if torch.isnan(x_tm1_mean).any():
        print("nan x_tm1_mean")
    if clip is not None:
        min, max = clip
        x_tm1_mean = torch.clamp(x_tm1_mean, min, max)
    update_term = alpha_tm1 * x_tm1_mean
    # assert not torch.isnan(update_term).any()
    if torch.isnan(update_term).any():
        print("nan update_term")
    noise_term = (
        torch.sqrt(torch.clamp(sigma_tm1.pow(2) - eta_t.pow(2), min=0)) * eps_hat
    )
    # assert not torch.isnan(noise_term).any()
    if torch.isnan(noise_term).any():
        print("nan noise_term")
    random_noise = torch.randn_like(x, device=device)
    if clip_noise:
        random_noise = torch.clamp(random_noise, clip_noise[0], clip_noise[1])
    random_noise *= eta_t
    x_tm1 = update_term + noise_term + random_noise
    return x_tm1


def sample(
    model,
    num_samples,
    return_steps,
    data_shape,
    labels=None,
    clip=None,
    clip_noise=None,
    cfg_w=None,
    null_class=None,
    device: str = "cuda",
):
    model.model.eval()
    if not isinstance(data_shape, (list, tuple)):
        data_shape = (data_shape,)
    x_shape = (num_samples,) + tuple(data_shape)
    exp_shape = [num_samples] + [1] * len(data_shape)
    samples = []  # [num_labels, len(return_steps), num_samples, *data_shape]

    if cfg_w is not None:
        assert labels is not None
        assert null_class is not None
        with torch.no_grad():
            null_class = torch.tensor(
                null_class, dtype=torch.int32, device=device
            ).expand(num_samples)

    if labels is None:
        labels = [None]
        model_kwargs = {}
    else:
        model_kwargs = {"training": False}

    for label in labels:
        label_samples = []
        with torch.no_grad():
            if label is not None:
                label = torch.tensor(label, dtype=torch.int32, device=device)
                label = label.expand(num_samples)
            for num_steps in return_steps:
                ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)

                x = torch.randn(x_shape, device=device)
                for i in range(num_steps):
                    t = torch.tensor([ts[i]], dtype=torch.float32, device=device)
                    tm1 = torch.tensor([ts[i + 1]], dtype=torch.float32, device=device)

                    alpha_t = model.get_alpha(t).expand(exp_shape)
                    alpha_tm1 = model.get_alpha(tm1).expand(exp_shape)
                    sigma_t = model.get_sigma(t).expand(exp_shape)
                    sigma_tm1 = model.get_sigma(tm1).expand(exp_shape)

                    # assert not torch.isnan(x).any(), f"step: {i}"
                    if torch.isnan(x).any():
                        print("nan x at step = ", i)
                    eps_hat = model.model_fn(
                        x, label, t.expand(num_samples), **model_kwargs
                    )
                    # assert not torch.isnan(eps_hat).any(), f"step: [{i}/{num_steps}], t: {t}"
                    if torch.isnan(eps_hat).any():
                        print("nan eps_hat step = ", i)
                        breakpoint()
                    if cfg_w is not None:
                        eps_hat_null = model.model_fn(
                            x, null_class, t.expand(num_samples), **model_kwargs
                        )
                        eps_hat = eps_hat_null + cfg_w * (eps_hat - eps_hat_null)

                    x = ddpm_update(
                        x,
                        eps_hat,
                        alpha_t,
                        alpha_tm1,
                        sigma_t,
                        sigma_tm1,
                        clip=clip,
                        clip_noise=clip_noise,
                    )

                label_samples.append(x.cpu().detach().numpy())
            samples.append(label_samples)

    # Squeeze out the label and return_steps dimensions if there's only.
    samples = np.array(samples)
    if len(labels) == 1:
        samples = samples.squeeze(0)
    return samples


def _get_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def _expand(t, data_shape):
    for _ in range(len(data_shape)):
        t = t[..., None]
    return t


def _x_hat(x_t, eps_hat, t, data_shape):
    alpha_t, sigma_t = _get_alpha_sigma(_expand(t, data_shape))
    return (x_t - sigma_t * eps_hat) / alpha_t


@torch.no_grad()
def sample_pieter(
    model_fn, n, num_steps, data_shape, clip_denoised=False, cfg_val=None
):

    ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1, dtype=np.float32)
    x = torch.randn(n, *data_shape, dtype=torch.float32).cuda()
    for i in range(num_steps):
        t_cur = torch.full((n,), ts[i], dtype=torch.float32).cuda()
        t_next = torch.full((n,), ts[i + 1], dtype=torch.float32).cuda()

        alpha_cur, sigma_cur = _get_alpha_sigma(_expand(t_cur, data_shape))
        alpha_next, sigma_next = _get_alpha_sigma(_expand(t_next, data_shape))
        ddim_sigma = (sigma_next / sigma_cur) * torch.sqrt(
            1 - alpha_cur**2 / alpha_next**2
        )

        if cfg_val is None:
            eps_hat = model_fn(x, None, t_cur)  # Labels are None.
        else:
            raise NotImplementedError()
            # eps_hat_cond = model_fn(x, t_cur)
            # eps_hat_uncond = model_fn(x, t_cur, dropout_cond=True)
            # eps_hat = eps_hat_uncond + cfg_val * (eps_hat_cond - eps_hat_uncond)

        x_hat = _x_hat(x, eps_hat, t_cur, data_shape)
        if clip_denoised:
            x_hat = torch.clamp(x_hat, -1, 1)
        x = (
            alpha_next * x_hat
            + torch.sqrt((sigma_next**2 - ddim_sigma**2).clamp(min=0)) * eps_hat
            + ddim_sigma * torch.randn_like(eps_hat)
        )
    # if self.decode_fn is not None:
    #     x = self.decode_fn(x)
    return x


if __name__ == "__main__":
    path = "/nas/ucb/ebronstein/lsdp/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"

    dataset = PushTImageDataset(path)
    full_dataset = torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2)
    # Make the state normalizer.
    max_state = dataset.replay_buffer["state"].max(axis=0)
    # min_state = np.zeros_like(max_state)
    min_state = dataset.replay_buffer["state"].min(axis=0)

    # Make train and val loaders
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, 0.1)
    val_idxs = np.where(val_mask)[0]
    train_idxs = np.where(~val_mask)[0]
    # train_idxs = [0]
    # val_idxs = [1]

    # Make the episode dataset and create a DataLoader.
    batch_size = 256
    n_obs_history = 8
    n_pred_horizon = 0

    def get_normalized_T_xy(state):
        """Get the normalized x, y coordinates of the T.

        Args:
            state (np.ndarray): State with shape (n, 5).
        """
        xy = state[:, 2:4]
        return normalize_pn1(xy, min_val=min_state[2:4], max_val=max_state[2:4])

    state_normalizer = functools.partial(
        normalize_pn1, min_val=min_state, max_val=max_state
    )
    process_fns = {"state": state_normalizer}
    # process_fns = {"state": get_normalized_T_xy}

    include_keys = ["state"]

    train_episode_dataset = EpisodeDataset(
        dataset,
        n_obs_history=n_obs_history,
        n_pred_horizon=n_pred_horizon,
        episode_idxs=train_idxs,
        include_keys=include_keys,
        process_fns=process_fns,
        device="cuda",
    )
    val_episode_dataset = EpisodeDataset(
        dataset,
        n_obs_history=n_obs_history,
        n_pred_horizon=n_pred_horizon,
        episode_idxs=val_idxs,
        include_keys=include_keys,
        process_fns=process_fns,
        device="cuda",
    )
    train_loader = torch.utils.data.DataLoader(
        train_episode_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_episode_dataset, batch_size=batch_size, shuffle=False
    )

    STATE_DIM = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input_dim + 1 because the time step is concatenated to the state.
    # input_dim = n_obs_history * STATE_DIM + 1
    # hidden_dims = [256] * 4
    # diff_model = DiffusionMLP(input_dim, input_dim - 1, hidden_dims).to(device)

    diff_model = conditional_unet1d.ConditionalUnet1D(
        input_dim=STATE_DIM, down_dims=[128, 256, 512, 1024]
    ).to(device)

    optim_kwargs = dict(lr=3e-4)
    diffusion = Diffusion(
        train_data=train_loader,
        test_data=val_loader,
        model=diff_model,
        n_epochs=200,
        optim_kwargs=optim_kwargs,
        device=device,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"pusht_1dconv_128_256_512_1024/{timestamp}"
    save_dir = f"models/diffusion/{name}"
    os.makedirs(save_dir)
    wandb_run = wandb.init(project='state_1dconv', name=name, reinit=True)
    wandb_run = None
    train_losses, test_losses = diffusion.train(
        wandb_run=wandb_run, save_freq=30, save_dir=save_dir
    )

    return_steps = [512]
    num_samples = 1000
    normalized_samples = sample(
        diffusion,
        num_samples=num_samples,
        return_steps=return_steps,
        data_shape=(n_obs_history, STATE_DIM),
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

    # Plot samples histogram.
    n_data, state_dim = dataset.replay_buffer["state"].shape
    plot_samples(
        in_range_samples,
        np.broadcast_to(
            dataset.replay_buffer["state"][:, None], [n_data, n_obs_history, state_dim]
        ),
        save_dir=save_dir,
    )

    diff_states = []
    for obs_history, pred_horizon in train_loader:
        norm_state = (
            obs_history["state"].detach().cpu().numpy()
        )  # [batch_size, n_history, dim]
        state = denormalize_pn1(
            norm_state, min_state, max_state
        )  # [batch_size, n_history, dim]
        diff_states.append(np.diff(state, axis=1))  # [batch_size, n_history - 1, dim]

    diff_states = np.concatenate(diff_states, axis=0)  # [n_samples, n_history - 1, dim]
    diff_samples = np.diff(
        in_range_samples, axis=2
    )  # [len(return_steps), num_in_range_samples, n_history - 1, dim]

    plot_samples(diff_samples, diff_states, save_dir=save_dir)
