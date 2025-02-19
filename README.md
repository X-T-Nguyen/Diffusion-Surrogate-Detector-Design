# Denoising Diffusion Probability Model
Application of Denoising Diffusion Probability Model on High Energy Physics. The model is trained with a Geant4 simulation dataset. <br>
<br>


# Conditional DDPM and DDIM Sampler

## Conditional DDPM

During training, the conditional DDPM is optimized with the following objective:

$$
\mathcal{L} = \mathbb{E}_{x_0,\, y,\, t,\, \epsilon \sim \mathcal{N}(0,I)} \left[ \left\| \epsilon - \epsilon_\theta\Bigl(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\; t,\; y\Bigr) \right\|^2 \right]
$$

where:
- \( x_0 \) is the ground truth image,
- \( y \) is the conditioning label,
- \( t \) is the diffusion timestep,
- \( \epsilon \sim \mathcal{N}(0,I) \) is the Gaussian noise,
- \( \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \) with \( \alpha_t = 1 - \beta_t \).

The reverse (denoising) process is given by:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, x_0^{\text{pred}} + \sqrt{1-\bar{\alpha}_{t-1}}\, \epsilon,
$$

with the predicted \( x_0 \) computed as:

$$
x_0^{\text{pred}} = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t, y) \right).
$$

## DDIM Sampler

In our evaluation, we use a DDIM sampler, which modifies the reverse update. First, we compute the noise scale:

$$
\sigma_t = \eta\, \sqrt{ \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \left( 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} \right) }
$$

Then, the DDIM update rule is given by:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, x_0^{\text{pred}} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\, \epsilon + \sigma_t\, z,
$$

where:
- \( z \sim \mathcal{N}(0,I) \) is an optional noise term (used when \( \eta > 0 \)); setting \( \eta = 0 \) makes the process deterministic.

---

**Results**  
DDPM + Classifier Free Guidance:

| Ground Truth | Generated Sample |
|:------------:|:----------------:|
| <img src="https://raw.githubusercontent.com/Tungcg1906/DDPMs/SampledImgs/ground_truth.png" alt="Ground Truth" width="300"> | <img src="https://raw.githubusercontent.com/Tungcg1906/DDPMs/main/SampledImgs/SampledGuidenceImgs_300.png" alt="Generated Sample" width="300"> |

***Reference/Related Work***  
<br>
<br>
(1) Kita, M., Dubiński, J., Rokita, P., & Deja, K. (2024). Generative Diffusion Models for Fast Simulations of Particle Collisions at CERN. arXiv. [https://doi.org/10.48550/arXiv.2406.03233](https://doi.org/10.48550/arXiv.2406.03233)

(2) Phillips, A., Dau, H.-D., Hutchinson, M. J., De Bortoli, V., Deligiannidis, G., & Doucet, A. (2024). Particle Denoising Diffusion Sampler. arXiv. [https://doi.org/10.48550/arXiv.2402.06320](https://doi.org/10.48550/arXiv.2402.06320)

(3) Kansal, R., Li, A., Duarte, J., Chernyavskaya, N., Pierini, M., Orzari, B., & Tomei, T. (2023). Evaluating generative models in high energy physics. *Physical Review D, 107*(7), 076017. [https://doi.org/10.1103/PhysRevD.107.076017](https://doi.org/10.1103/PhysRevD.107.076017)

(4) Amram, O., & Pedro, K. (2023). Denoising diffusion models with geometry adaptation for high fidelity calorimeter simulation [arXiv:2308.03876v3]. arXiv. [https://doi.org/10.48550/arXiv.2308.03876](https://doi.org/10.48550/arXiv.2308.03876)
