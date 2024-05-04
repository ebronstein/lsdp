import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
    plt.grid()
    plt.title(f"Min train: {np.min(train_losses):.2e} Last train: {train_losses[-1]:.2e} "
              f"\n Min test: {np.min(test_losses):.2e} Last test: {test_losses[-1]:.2e} ")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "train_test_losses.png"))
    else:
        plt.show()


def plot_samples(samples, data, return_steps, save_dir=None):
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


class LatentsToStateMLP(nn.Module):
    def __init__(
            self, in_dim, out_dim, hidden_dims: list[int]
    ):
        super().__init__()

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        # layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = obs_history.flatten(start_dim=1)
        x = self.model(x)
        return x
