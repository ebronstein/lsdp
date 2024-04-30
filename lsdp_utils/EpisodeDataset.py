import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
from tqdm.notebook import tqdm, trange
from diffusion_policy.common.sampler import get_val_mask


def EpisodeDataloaders(dataset, include_keys, process_fns, cfg, val_ratio=0.1) -> Tuple[Dataset, Dataset]:

    # Make train and val loaders
    val_mask = get_val_mask(dataset.replay_buffer.n_episodes, val_ratio=val_ratio, seed=0)
    episode_val_idxs = np.where(val_mask)[0]
    episode_train_idxs = np.where(~val_mask)[0]

    train_episode_dataset = EpisodeDataset(
        dataset,
        n_obs_history=cfg.n_obs_history,
        n_pred_horizon=cfg.n_pred_horizon,
        episode_idxs=episode_train_idxs,
        include_keys=include_keys,
        process_fns=process_fns,
        device=cfg.device,
    )

    val_episode_dataset = EpisodeDataset(
        dataset,
        n_obs_history=cfg.n_obs_history,
        n_pred_horizon=cfg.n_pred_horizon,
        episode_idxs=episode_val_idxs,
        include_keys=include_keys,
        process_fns=process_fns,
        device=cfg.device,
    )

    train_loader = torch.utils.data.DataLoader(
        train_episode_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_episode_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    return train_loader, val_loader


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
                if type(img) == torch.Tensor:
                    obs["img"] = img.to(torch.float32).to(self.device)
                else:
                    obs["img"] = torch.tensor(img, dtype=torch.float32, device=self.device)

            if self.include_keys is None or "action" in self.include_keys:
                action = episode["action"]
                if "action" in self.process_fns:
                    action = self.process_fns["action"](action)
                obs["action"] = torch.tensor(action, dtype=torch.float32, device=self.device)

            if self.include_keys is None or "state" in self.include_keys:
                state = episode["state"]
                if "state" in self.process_fns:
                    state = self.process_fns["state"](state)
                obs["state"] = torch.tensor(state, dtype=torch.float32, device=self.device)

            # Iterate through the episode to create samples with observation history and prediction horizon
            n_obs = len(list(obs.values())[0])
            for i in range(n_obs - self.n_obs_history - self.n_pred_horizon + 1):
                obs_history = {}
                pred_horizon = {}

                for key, value in obs.items():
                    obs_history[key] = value[i: i + self.n_obs_history]
                    pred_horizon[key] = value[
                                        i
                                        + self.n_obs_history: i
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
