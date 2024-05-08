import datetime
import os
import sys
from typing import Optional

if "PyTorch_VAE" not in sys.path:
    sys.path.append("PyTorch_VAE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torchvision.transforms import v2
from tqdm import tqdm, trange

from diffusion_policy.common.pytorch_util import compute_conv_output_shape
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from PyTorch_VAE import models
from utils import denormalize_img, normalize_img, plot_losses
from vae import VanillaVAE

ex = Experiment()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample


def get_exp_dir(
    save_dir,
    use_transforms,
    latent_dim,
    lr,
    epochs,
    kl_loss_coeff,
    batch_size,
):
    data_aug_tag = "data_aug" if use_transforms else "no_data_aug"
    tag = f"pusht_vae_dim_{latent_dim}_lr_{lr}_epochs_{epochs}_kl_loss_{kl_loss_coeff}_bs_{batch_size}_{data_aug_tag}"
    return os.path.join(save_dir, tag)


@ex.config
def sacred_config():
    # Data
    data_paths = [
        "diffusion_policy/data/pusht/images/random_100000/20240508-142447",
        "/nas/ucb/ebronstein/lsdp/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr",
    ]
    train_split = 0.8
    use_transforms = False

    # Parameters for the VAE
    batch_size = 256
    epochs = 20
    lr = 1e-3
    latent_dim = 32
    kl_loss_coeff = 1e-3  # Original: 1e-6

    device = None

    # Saving
    save_dir = "models/pusht_vae"
    exp_dir = get_exp_dir(
        save_dir,
        use_transforms,
        latent_dim,
        lr,
        epochs,
        kl_loss_coeff,
        batch_size,
    )

    # NOTE: set the observers to this single FileStorageObserver instead of
    # appending to ex.observers. This allows running the experiment multiple
    # times without creating multiple observers.
    ex.observers = [FileStorageObserver(exp_dir, copy_sources=False)]


def train(model, train_loader, val_loader, epochs, optimizer, kl_loss_coeff, device):
    train_losses, val_losses = [], []

    for epoch in trange(epochs):
        total_train_loss = 0
        model.train()
        for i, x in tqdm(enumerate(train_loader)):
            x = x.to(device)
            result = model(x)
            loss = model.loss_function(*result, M_N=kl_loss_coeff)["loss"]
            # loss = loss['loss']
            total_train_loss += loss.item()
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Train loss: {total_train_loss / len(train_loader):.4f}")

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, x in enumerate(val_loader):
                x = x.to(device)
                result = model(x)
                loss = model.loss_function(*result, M_N=kl_loss_coeff)["loss"]
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))
        print(f"Validation loss: {val_losses[-1]:.4f}")


    return train_losses, val_losses


def show_reconstructions(
    model: VanillaVAE,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    sample: bool = True,
    save_dir: Optional[str] = None,
):
    val_data = next(iter(val_loader))
    num_samples = 5
    val_data = val_data.to(device)
    if sample:
        result = model(val_data)
        recon = result[0]
    else:
        mu, _ = model.encode(val_data)
        recon = model.decode(mu)

    recon = denormalize_img(recon)
    val_data = denormalize_img(val_data)

    fig, ax = plt.subplots(2, num_samples, figsize=(num_samples * 2, 6))
    # fig.set_size_inches(10, 10)
    for ii in range(num_samples):
        ax[0, ii].imshow(
            val_data[ii].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        )
        ax[1, ii].imshow(
            recon[ii].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        )
        ax[0, ii].axis("off")
        ax[1, ii].axis("off")

    # plt.suptitle("Reconstructions")
    ax[0, 0].set_title("Ground Truth")
    ax[1, 0].set_title("Reconstruction")
    plt.tight_layout()
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sample_tag = "sample" if sample else "mean"
        plt.savefig(os.path.join(save_dir, f"recon_{sample_tag}.png"))
    else:
        plt.show()


@ex.automain
def main(_config, _run, _log):
    # Load the dataset.
    datasets = []
    data_paths = _config["data_paths"]
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    for path in tqdm(data_paths):
        if path.endswith(".zarr"):
            pusht_dataset = PushTImageDataset(path)
            dataset = pusht_dataset.replay_buffer["img"].transpose(0, 3, 1, 2)
            datasets.append(dataset)
        else:
            data_files = [file for file in os.listdir(path) if file.endswith(".npy")]
            for file in tqdm(data_files):
                file_path = os.path.join(path, file)
                datasets.append(np.load(file_path))

    full_dataset = np.concatenate(datasets, axis=0)

    # Data augmentation transforms.
    if _config["use_transforms"]:
        transforms = v2.Compose(
            [
                # v2.RandomResizedCrop(size=(96, 96), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomRotation(45),
            ]
        )
    else:
        transforms = None

    # Make the dataset.
    batch_size = _config["batch_size"]
    full_dataset = normalize_img(full_dataset)
    N, C, H, W = full_dataset.shape
    train_size = int(_config["train_split"] * N)
    val_size = N - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    train_dataset = CustomDataset(train_dataset, transform=transforms)
    val_dataset = CustomDataset(val_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        # num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Model and optimizer.
    device = _config["device"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaVAE(
        in_channels=3, in_height=H, in_width=W, latent_dim=_config["latent_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_config["lr"])

    train_losses, test_losses = train(
        model,
        train_loader,
        val_loader,
        _config["epochs"],
        optimizer,
        _config["kl_loss_coeff"],
        device,
    )

    # Save the model and the losses in run_dir.
    run_id = _run._id
    run_dir = os.path.join(_config["exp_dir"], run_id)
    _log.info(f"Saving to {run_dir}")
    model_path = os.path.join(run_dir, f"vae.pt")
    torch.save(model.state_dict(), model_path)
    np.save(os.path.join(run_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(run_dir, "test_losses.npy"), np.array(test_losses))

    # Plot the losses.
    plot_losses(train_losses, test_losses, save_dir=run_dir)

    # Plot the reconstructions.
    show_reconstructions(model, val_loader, device, sample=False, save_dir=run_dir)
