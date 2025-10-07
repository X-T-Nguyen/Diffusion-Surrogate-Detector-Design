# Differentiable Surrogate for Detector Simulation and Design with Diffusion Models

This repository contains the code for the paper:

**"Differentiable Surrogate for Detector Simulation and Design with Diffusion Models"**  
by Xuan Tung Nguyen et al., [arXiv link TBD].

The project provides a conditional denoising-diffusion probabilistic model (DDPM) to simulate electromagnetic calorimeter showers. The model can generate high-fidelity, differentiable shower distributions conditioned on detector geometry, material, and incoming particle energy.

---

## Features

- **Pre-trained conditional DDPM**: Generates showers for various calorimeter geometries and energies.
- **Low-Rank Adaptation (LoRA) modules**: Enables efficient fine-tuning for new detector configurations.
- **Differentiable surrogate**: Supports gradient-based optimization of detector design.
- **Evaluation metrics**: Total energy, energy-weighted radius, and shower dispersion.


