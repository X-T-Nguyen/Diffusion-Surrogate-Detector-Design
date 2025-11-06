# DiffusionCondition.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, energy_labels, xy_values, z_values, material_labels):
        t = torch.randint(self.T, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, energy_labels, xy_values, z_values, material_labels), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones((x_T.shape[0],), dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t, t, labels)
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
        super().__init__()
        self.model = model
        self.T = T
        self.eta = eta
        self.ddim_steps = ddim_steps
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))
        self.ddim_timesteps = self.get_ddim_timesteps()

    def get_ddim_timesteps(self):
        return torch.linspace(0, self.T - 1, self.ddim_steps).long()

    def forward(self, x, energy_labels, xy_values, z_values, material_labels):
        for i in reversed(range(len(self.ddim_timesteps))):
            t = int(self.ddim_timesteps[i].item())
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = self.model(x, t_tensor, energy_labels, xy_values, z_values, material_labels)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
            if i > 0:
                t_prev = int(self.ddim_timesteps[i - 1].item())
                alpha_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_prev = torch.tensor(1.0, device=x.device)
            sigma = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            noise = torch.randn_like(x) if self.eta > 0 else 0
            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev - sigma**2) * eps + sigma * noise
        return x


