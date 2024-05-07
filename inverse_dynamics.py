import argparse
import datetime
import functools
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from data_utils import EpisodeDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import compute_conv_output_shape
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from utils import (
    denormalize_img,
    denormalize_pn1,
    normalize_img,
    normalize_pn1,
    plot_losses,
)
from vae import VanillaVAE


class InverseDynamicsCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        action_dim: int,
        n_obs_history: int,
        hidden_dims: list[int] = None,
    ):
        super(InverseDynamicsCNN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        modules = []
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        # Define the initial part of the CNN that processes individual images
        self.conv_branch = nn.Sequential(
            *modules,
            # Flatten the output for the dense layers
            nn.Flatten(),
        )

        # Compute the shape of the output of the convolutional branch before it
        # is flattened and passed through the dense layers.
        conv_out_shape = compute_conv_output_shape(
            H=in_height,
            W=in_width,
            padding=padding,
            stride=stride,
            kernel_size=kernel_size,
            dilation=dilation,
            num_layers=len(hidden_dims),
            last_hidden_dim=hidden_dims[-1],
        )
        conv_out_size = np.prod(conv_out_shape)

        # Define the part of the network that combines features and predicts the action
        self.action_predictor = nn.Sequential(
            nn.Linear(n_obs_history * conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs_history: torch.Tensor):
        """Forward pass.

        Args:
            obs_history (torch.Tensor): Observation history of shape (batch_size, n_obs, C, H, W).
        """
        # List to hold the features extracted from each image in the observation history
        features_list = []

        # Iterate over the second dimension (n_obs) of the obs_history tensor
        for i in range(obs_history.size(1)):
            # Extract the i-th image from the observation history
            img_i = obs_history[:, i]

            # Process the image through the convolutional branch
            img_i_features = self.conv_branch(img_i)

            # Append the features to the list
            features_list.append(img_i_features)

        # Concatenate the features from all images along the feature dimension (dim=1)
        combined_features = torch.cat(features_list, dim=1)

        # Predict the action from the combined features
        action_pred = self.action_predictor(combined_features)
        return action_pred


@torch.no_grad()
def get_recon(x, vae_model, device):
    x = normalize_img(x)  # Range: [-1, 1]
    # Output is in [-1, 1] range because the VAE has a tanh activation at the end.
    return vae_model(torch.from_numpy(x).to(device))[0].detach()


def train_epochs(
    model,
    train_loader,
    val_loader,
    obs_key: str = "img",
    opt_kwargs: Optional[dict] = None,
    num_epochs=10,
    log_freq: Optional[int] = None,
    save_freq=2,
    save_dir: Optional[str] = None,
    device="cuda",
):
    criterion = nn.MSELoss()
    opt_kwargs = opt_kwargs or {}
    optimizer = torch.optim.Adam(model.parameters(), **opt_kwargs)

    train_losses = []
    test_losses = [
        eval(
            model,
            val_loader,
            criterion,
            device,
            obs_key=obs_key,
        )
    ]
    with trange(num_epochs, desc="Epoch") as tepoch:
        for epoch in tepoch:
            model.train()
            with tqdm(train_loader, desc="Batch") as tbatch:
                # Prediction horizon is unused.
                for i, (obs_history, _) in enumerate(tbatch):
                    obs = obs_history[obs_key]
                    # The second-to-last action is the target action because it was
                    # applied to get the last image.
                    action = obs_history["action"][:, -2]

                    obs = obs.to(device)
                    action = action.to(device)

                    optimizer.zero_grad()
                    action_pred = model(obs)
                    loss = criterion(action_pred, action)
                    loss.backward()
                    optimizer.step()

                    loss_cpu = loss.item()
                    train_losses.append(loss_cpu)

                    tbatch.set_postfix(loss=loss_cpu)
                    if log_freq is not None and (i % log_freq == 0):
                        print(f"Epoch {epoch}, Batch {i}, Train Loss: {loss_cpu}")

            # Eval
            test_loss = eval(
                model,
                val_loader,
                criterion,
                device,
                obs_key=obs_key,
            )
            test_losses.append(test_loss)
            tepoch.set_postfix(test_loss=test_loss)

            # Save
            if save_dir is not None and (
                epoch % save_freq == 0 or epoch == num_epochs - 1
            ):
                epoch_str = "final" if epoch == num_epochs - 1 else str(epoch)
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"inverse_dynamics_{epoch_str}.pt"),
                )

    return train_losses, test_losses


def eval(
    model,
    val_loader,
    criterion,
    device,
    obs_key: str = "img",
):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for obs_history, _ in val_loader:
            obs = obs_history[obs_key]
            # The second-to-last action is the target action because it was
            # applied to get the last image.
            action = obs_history["action"][:, -2]

            obs = obs.to(device)
            action = action.to(device)

            action_pred = model(obs)
            loss = criterion(action_pred, action)
            # Multiply the loss by the number of samples in the batch.
            test_losses.append(loss.item() * obs.shape[0])

    # Compute the average loss across all batches.
    test_loss = np.sum(test_losses) / len(val_loader.dataset)
    return test_loss


def train_inverse_dynamics(args):
    dataset = PushTImageDataset(args.data)
    device = args.device
    latent_dim = args.latent_dim
    vae_path = args.vae_path
    root_save_dir = args.root_save_dir
    lr = args.lr
    train_on_recon = args.train_on_recon

    # Load the VAE
    img_data = (
        torch.from_numpy(dataset.replay_buffer["img"]).permute(0, 3, 1, 2).float()
    )
    N, C, H, W = img_data.shape
    vae = VanillaVAE(in_channels=C, in_height=H, in_width=W, latent_dim=latent_dim).to(
        device
    )
    vae.load_state_dict(torch.load(vae_path))

    # Make train and val loaders
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, 0.1)
    val_idxs = np.where(val_mask)[0]
    train_idxs = np.where(~val_mask)[0]

    # Make the episode dataset and create a DataLoader.
    batch_size = 256
    n_obs_history = 5
    n_pred_horizon = 0
    include_keys = ["img", "action"]

    # Normalization functions.
    max_action = dataset.replay_buffer["action"].max(axis=0)
    min_action = np.zeros_like(max_action)
    normalize_action = functools.partial(
        normalize_pn1, min_val=min_action, max_val=max_action
    )
    if train_on_recon:
        img_normalizer = functools.partial(get_recon, vae_model=vae, device=device)
    else:
        img_normalizer = normalize_img

    process_fns = {"img": img_normalizer, "action": normalize_action}

    train_episode_dataset = EpisodeDataset(
        dataset,
        n_obs_history=n_obs_history,
        n_pred_horizon=n_pred_horizon,
        episode_idxs=train_idxs,
        include_keys=include_keys,
        process_fns=process_fns,
        device=device,
    )
    val_episode_dataset = EpisodeDataset(
        dataset,
        n_obs_history=n_obs_history,
        n_pred_horizon=n_pred_horizon,
        episode_idxs=val_idxs,
        include_keys=include_keys,
        process_fns=process_fns,
        device=device,
    )
    train_loader = torch.utils.data.DataLoader(
        train_episode_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_episode_dataset, batch_size=batch_size, shuffle=False
    )

    hidden_dims = None
    N, H, W, C = dataset.replay_buffer["img"].shape
    N, action_dim = dataset.replay_buffer["action"].shape
    cnn_id_model = InverseDynamicsCNN(
        C, H, W, action_dim, n_obs_history, hidden_dims=hidden_dims
    ).to(device)

    obs_key = "img"
    if root_save_dir is not None:
        name = (
            f'pusht_cnn-{obs_key}-'
            f"obs_{n_obs_history}-bs_{batch_size}-lr_{lr}-epochs_{args.n_epochs}-"
            f"train_on_recon-{train_on_recon}-latent_dim_{latent_dim}"
        )
        save_dir = os.path.join(root_save_dir, name)
        # Get the current timestamp and save it as a new directory.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(save_dir)
    else:
        save_dir = None

    train_losses, test_losses = train_epochs(
        cnn_id_model,
        train_loader,
        val_loader,
        obs_key=obs_key,
        opt_kwargs={"lr": lr},
        num_epochs=args.n_epochs,
        log_freq=2,
        save_freq=10,
        save_dir=save_dir,
        device=device,
    )

    # Plot losses.
    plot_losses(train_losses, test_losses, save_dir=save_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vae_path", type=str, default="models/pusht_vae/vae_32_20240403.pt"
    )
    parser.add_argument("--root_save_dir", type=str, default="models/inverse_dynamics")
    parser.add_argument("--device", type=str, default="cuda")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="/nas/ucb/ebronstein/lsdp/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr",
    )
    parser.add_argument("--train_on_recon", action="store_true")

    # Model
    parser.add_argument("--latent_dim", type=int, default=32)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=10)

    args = parser.parse_args()
    train_inverse_dynamics(args)
