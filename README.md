# Conditional Denoising Diffusion Probabilistic Model (DDPM) with DDIM Sampling

This repository implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) for high-energy physics simulations. The model is conditioned on different initial energy values and learns to simulate the process.

## 1. Conditional DDPM Formulation

A conditional DDPM models the data distribution $( p(x|y) )$, where $( y )$ is the conditioning variable (e.g., initial energy in our case). The forward diffusion process gradually adds Gaussian noise to the input:

$[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) I)
]$

where $( \alpha_t = 1 - \beta_t )$ is the noise schedule.

The reverse process is parameterized by a neural network $( \epsilon_\theta(x_t, t, y) )$ that estimates the noise added at each step:

$[
p_\theta(x_{t-1} | x_t, y) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, y), \Sigma_\theta(x_t, t, y))
]$

where

$[
\mu_\theta(x_t, t, y) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \alpha_t \prod_{s=1}^{t-1} \alpha_s}} \epsilon_\theta(x_t, t, y) \right)
]$

and $( \Sigma_\theta(x_t, t, y) )$ can be learned or fixed.

## 2. DDIM Sampling

Instead of using the Gaussian sampling approach in DDPM, DDIM (Denoising Diffusion Implicit Models) reformulates the sampling process to directly update \( x_t \) using:

$[
x_{t-1} = \sqrt{\alpha_{t-1}} \hat{x}_0 + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t, y) + \sigma_t \epsilon
]$

where $( \sigma_t \) is computed as:

$[
\sigma_t^2 = \eta^2 \frac{(1 - \alpha_{t-1})}{(1 - \alpha_t)} (1 - \alpha_t / \alpha_{t-1})
]$

- When $( \eta = 0 )$, DDIM is **deterministic**.
- When $( \eta > 0 )$, DDIM introduces **stochasticity**, similar to DDPM.

## 3. Implementation

The DDIM sampling method is implemented as follows:

```python
import torch
import torch.nn as nn

class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta=0.0, ddim_steps=50):
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
        return torch.linspace(0, self.T - 1, self.ddim_steps).long()

    def forward(self, x, labels):
        for i in reversed(range(len(self.ddim_timesteps))):
            t = int(self.ddim_timesteps[i].item())
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = self.model(x, t_tensor, labels)

            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

            if i > 0:
                t_prev = int(self.ddim_timesteps[i - 1].item())
                alpha_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_prev = torch.tensor(1.0, device=x.device)

            sigma = self.eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            noise = torch.randn_like(x) if self.eta > 0 else 0

            x = (
                torch.sqrt(alpha_prev) * x0_pred +
                torch.sqrt(1 - alpha_prev - sigma**2) * eps +
                sigma * noise
            )
        return x
```
---

## Results  
DDPM + Classifier Free Guidance:

| Ground Truth | Generated Sample |
|:------------:|:----------------:|
| <img src="https://github.com/Tungcg1906/DDPMs/blob/main/SampleImgs/ground_truth.png" alt="Ground Truth" width="300"> | <img src="https://github.com/Tungcg1906/DDPMs/blob/main/SampleImgs/SampledGuidenceImgs_300.png" alt="Generated Sample" width="300"> |

---

## References / Related Work  

(1) Kita, M., Dubiński, J., Rokita, P., & Deja, K. (2024). Generative Diffusion Models for Fast Simulations of Particle Collisions at CERN. arXiv. [https://doi.org/10.48550/arXiv.2406.03233](https://doi.org/10.48550/arXiv.2406.03233)  

(2) Phillips, A., Dau, H.-D., Hutchinson, M. J., De Bortoli, V., Deligiannidis, G., & Doucet, A. (2024). Particle Denoising Diffusion Sampler. arXiv. [https://doi.org/10.48550/arXiv.2402.06320](https://doi.org/10.48550/arXiv.2402.06320)  

(3) Kansal, R., Li, A., Duarte, J., Chernyavskaya, N., Pierini, M., Orzari, B., & Tomei, T. (2023). Evaluating generative models in high energy physics. *Physical Review D, 107*(7), 076017. [https://doi.org/10.1103/PhysRevD.107.076017](https://doi.org/10.1103/PhysRevD.107.076017)  

(4) Amram, O., & Pedro, K. (2023). Denoising diffusion models with geometry adaptation for high fidelity calorimeter simulation [arXiv:2308.03876v3]. arXiv. [https://doi.org/10.48550/arXiv.2308.03876](https://doi.org/10.48550/arXiv.2308.03876)  

---


