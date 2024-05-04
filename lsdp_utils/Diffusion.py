import torch
import torch.nn as nn
import numpy as np
import time
import os
import datetime
from typing import Optional
import warnings


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
        mlp_nograd_latents_to_state=None,
    ):
        if has_labels:
            raise NotImplementedError()

        self.n_epochs = n_epochs
        self.has_labels = has_labels
        self.device = device
        self.mlp_nograd_latents_to_state = mlp_nograd_latents_to_state
        # Freeze the model
        if mlp_nograd_latents_to_state is not None:
            for param in self.mlp_nograd_latents_to_state.parameters():
                param.requires_grad = False

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

        def model_without_labels(x, labels, t, **kwargs):
            return self.model(x, t, **kwargs)

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

    @staticmethod
    @torch.jit.script
    def get_alpha(t):
        return torch.cos(np.pi / 2 * t)

    @staticmethod
    @torch.jit.script
    def get_sigma(t):
        return torch.sin(np.pi / 2 * t)

    def compute_loss(self, x, obs, labels=None):
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

        # Flatten obs
        obs = obs.flatten(1)

        # Step 4: Estimate epsilon
        eps_hat = self.model_fn(x_t, labels, t, global_cond=obs)
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
                pred = pred_horizon[obs_key].to(self.device)
                obs = self.obs_normalizer(obs)

                # print(pred.shape)
                if self.mlp_nograd_latents_to_state is not None:
                    pred = self.mlp_nograd_latents_to_state(pred)
                # print(pred.shape)
                loss = self.compute_loss(pred, obs, labels)
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

        if self.has_labels:
            raise NotImplementedError()
            x, labels = x
            labels = labels.to(self.device)
            if process_labels_fn is not None:
                labels = process_labels_fn(labels)
        else:
            warnings.warn('[fn: train] labels is always None')
            labels = None


        train_losses = []
        test_losses = [self.eval(self.test_loader, obs_key=obs_key)]

        iter = 0
        train_start_time = time.time()
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch + 1})
            epoch_train_losses = []
            # grad_norms = []
            self.model.train()

            for x in self.train_loader:
                obs_history, pred_horizon = x
                obs = obs_history[obs_key].to(self.device)
                # print("obs:", obs.shape)
                obs = self.obs_normalizer(obs)
                # print("obs:", obs.shape)
                pred = pred_horizon[obs_key].to(self.device)

                self.optimizer.zero_grad()
                # print(obs.shape, obs)

                if self.mlp_nograd_latents_to_state is not None:
                    pred = self.mlp_nograd_latents_to_state(pred)

                loss = self.compute_loss(pred, obs, labels)
                loss.backward()

                # Compute the norm only if we need to log it
                if wandb_run is not None:
                    # Compute the norm of gradients
                    total_norm = 0
                    for p in self.model.parameters():
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    wandb_run.log({"grad_norm": total_norm})
                    wandb_run.log({"batch_loss": loss.item()})
                # grad_norms.append(total_norm)

                self.optimizer.step()
                self.scheduler.step()

                if self.use_ema_helper:
                    self.ema_helper.step(self.model)

                epoch_train_losses.append(loss.item())

                if log_freq is not None and iter % log_freq == 0:
                    print(f"Epoch {epoch+1}, iter {iter}, Train loss: {loss.item():.4f}")

                iter += 1

            train_losses.extend(epoch_train_losses)
            test_loss = self.eval(self.test_loader, obs_key=obs_key)
            epoch_end_time = time.time()
            print(f"[{datetime.timedelta(seconds=epoch_end_time - epoch_start_time)}] "
                  f"Epoch {epoch+1}, iter {iter}, Train Loss:{test_loss:.4f} Test Loss: {test_loss:.4f}")
            test_losses.append(test_loss)


            if save_dir is not None and epoch % save_freq == 0:
                print(f'Saving.. {datetime.timedelta(seconds=time.time() - train_start_time)}')
                self.save(os.path.join(save_dir, f"diffusion_model_epoch_{epoch}.pt"))
                np.save(os.path.join(save_dir, "train_losses.npy"), train_losses)
                np.save(os.path.join(save_dir, "test_losses.npy"), test_losses)

        if save_dir is not None:
            print(f'Saving.. {datetime.timedelta(seconds=time.time() - train_start_time)}')
            self.save(os.path.join(save_dir, "diffusion_model_final.pt"))
            np.save(os.path.join(save_dir, "train_losses.npy"), train_losses)
            np.save(os.path.join(save_dir, "test_losses.npy"), test_losses)

        return train_losses, test_losses



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
    data_loader: Optional[torch.utils.data.DataLoader] = None,
    labels=None,
    clip=None,
    clip_noise=None,
    cfg_w=None,
    null_class=None,
    obs_key: str = "state",
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

                if data_loader is None:
                    x_shape = (num_samples,) + tuple(data_shape)
                    obs = None
                else:
                    n_obs = 0
                    obs = []
                    for obs_history, _ in data_loader:
                        # [batch_size, n_obs_history, state_dim]
                        obs_history = obs_history[obs_key]
                        # Concatenate observations along the time dimension.
                        obs_history = obs_history.flatten(1)
                        obs.append(obs_history)
                        n_obs += obs_history.shape[0]
                        if n_obs >= num_samples:
                            break
                    obs = torch.cat(obs, dim=0)[:num_samples].to(device)

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
                        x, label, t.expand(num_samples), global_cond=obs, **model_kwargs
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
                        device=device,
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
