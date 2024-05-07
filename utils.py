import os

import matplotlib.pyplot as plt
import numpy as np


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


def normalize_standard_normal(x, mean, std):
    return (x - mean) / std


def denormalize_standard_normal(nx, mean, std):
    return nx * std + mean


def normalize_img(img):
    return normalize_pn1(img, 0, 255)


def denormalize_img(nimg):
    return denormalize_pn1(nimg, 0, 255)


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
