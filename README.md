# Conditional Denoising Diffusion Probabilistic Model in High Energy Physics

This repository implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) for high-energy physics simulations. The model is conditioned on different initial energy values and learns to simulate the process.

## 1. Conditional DDPM Formulation

A conditional DDPM models the data distribution $( p(x|y) )$, where $( y )$ is the conditioning variable (e.g., initial energy in our case). The forward diffusion process gradually adds Gaussian noise to the input:

 <img src="https://github.com/Tungcg1906/DDPMs/blob/main/SampleImgs/ddpm-model.png" alt="ddpm" width="800">

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

## 3. Model Configuration and Parameters

```python
from DiffusionFreeGuidence.TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        #"state": "train", 
        "state": "eval", 
        "epoch": 300,
        "batch_size": 25, 
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2], 
        "num_res_blocks": 2, 
        "dropout": 0.15,
        "lr": 1e-4, 
        "multiplier": 2,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 128,
        "grad_clip": 3., 
        "device": "cuda:0",
        "w": 1.8, 
        "save_weight_dir": "./CheckpointsCondition_large_batch/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_300.pt",
        "sampled_dir": "./Test_img_condition/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 5
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
```
---

## Results  
Classifier Free Guidance DDPM:

| Ground Truth | Generated Sample |
|:------------:|:----------------:|
| <img src="https://github.com/Tungcg1906/DDPMs/blob/main/SampleImgs/ground_truth.png" alt="Ground Truth" width="300"> | <img src="https://github.com/Tungcg1906/DDPMs/blob/main/SampleImgs/SampledGuidenceImgs_300.png" alt="Generated Sample" width="300"> |

Metrics evvaluation:

<img src="https://github.com/Tungcg1906/DDPMs/blob/main/evaluation_results/metrics_vs_epoch.png" alt="metric" width="800">

---

## References / Related Work  

(1) Kita, M., Dubiński, J., Rokita, P., & Deja, K. (2024). Generative Diffusion Models for Fast Simulations of Particle Collisions at CERN. arXiv. [https://doi.org/10.48550/arXiv.2406.03233](https://doi.org/10.48550/arXiv.2406.03233)  

(2) Phillips, A., Dau, H.-D., Hutchinson, M. J., De Bortoli, V., Deligiannidis, G., & Doucet, A. (2024). Particle Denoising Diffusion Sampler. arXiv. [https://doi.org/10.48550/arXiv.2402.06320](https://doi.org/10.48550/arXiv.2402.06320)  

(3) Kansal, R., Li, A., Duarte, J., Chernyavskaya, N., Pierini, M., Orzari, B., & Tomei, T. (2023). Evaluating generative models in high energy physics. *Physical Review D, 107*(7), 076017. [https://doi.org/10.1103/PhysRevD.107.076017](https://doi.org/10.1103/PhysRevD.107.076017)  

(4) Amram, O., & Pedro, K. (2023). Denoising diffusion models with geometry adaptation for high fidelity calorimeter simulation [arXiv:2308.03876v3]. arXiv. [https://doi.org/10.48550/arXiv.2308.03876](https://doi.org/10.48550/arXiv.2308.03876)  

---


