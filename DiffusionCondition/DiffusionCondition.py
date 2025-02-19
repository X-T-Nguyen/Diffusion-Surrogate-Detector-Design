
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta=0.0, ddim_steps=50):
        """
        Args:
            model: Trained diffusion model (e.g., UNet)
            beta_1: The initial beta value of the noise schedule.
            beta_T: The final beta value of the noise schedule.
            T: Total number of diffusion timesteps used during training.
            eta: Controls the stochasticity (set eta=0.0 for deterministic sampling).
            ddim_steps: Number of timesteps to use during DDIM sampling.
        """
        super().__init__()
        self.model = model
        self.T = T
        self.eta = eta
        self.ddim_steps = ddim_steps

        # Create a linear beta schedule
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))
        
        # Create a mapping from DDIM steps to the original T timesteps
        self.ddim_timesteps = self.get_ddim_timesteps()

    def get_ddim_timesteps(self):
        # Evenly spaced timesteps from 0 to T-1
        return torch.linspace(0, self.T - 1, self.ddim_steps).long()

    def forward(self, x, labels):
        """
        Args:
            x: Input noise tensor of shape [batch, channels, height, width].
            labels: Corresponding labels tensor.
        Returns:
            x: Generated sample image.
        """
        for i in reversed(range(len(self.ddim_timesteps))):
            # Get current timestep (as an integer)
            t = int(self.ddim_timesteps[i].item())
            # Create a tensor for time conditioning for all batch items
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            
            # Predict the noise epsilon at time t using your model
            eps = self.model(x, t_tensor, labels)
            
            # Compute predicted x0 from the model output
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

            # Determine the previous timestep's alpha value
            if i > 0:
                t_prev = int(self.ddim_timesteps[i - 1].item())
                alpha_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_prev = torch.tensor(1.0, device=x.device)

            # Compute sigma for the DDIM update
            sigma = self.eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            # Optional noise term (only if eta > 0)
            noise = torch.randn_like(x) if self.eta > 0 else 0

            # Update x for the next (previous in time) step
            x = (
                torch.sqrt(alpha_prev) * x0_pred +
                torch.sqrt(1 - alpha_prev - sigma**2) * eps +
                sigma * noise
            )
        return x

