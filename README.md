# Differentiable Surrogate for Detector Simulation and Design with Diffusion Models

This repository contains the code for the paper:

**"Differentiable Surrogate for Detector Simulation and Design with Diffusion Models"**  
by Xuan Tung Nguyen et al., [arXiv link TBD].

The project provides a conditional denoising-diffusion probabilistic model (DDPM) to simulate electromagnetic calorimeter showers. The model can generate high-fidelity, differentiable shower distributions conditioned on detector geometry, material, and incoming particle energy.

---

![Calo shower](https://github.com/X-T-Nguyen/Diffusion-Surrogate-Detector-Design/blob/main/images/compare_XY4_Z10_small.png)
![Energy pprofile](https://github.com/X-T-Nguyen/Diffusion-Surrogate-Detector-Design/blob/main/images/energy_profiles_PbF2_xy4_z10_small.png)


## Table of Contents
- [Installation](#installation)
- [Data](#data)

## Installation
```
git clone https://github.com/X-T-Nguyen/Diffusion-Surrogate-Detector-Design.git
cd Diffusion-Surrogate-Detector-Design
conda create -n diff-surrogate python=3.10
conda activate diff-surrogate
pip install -r requirements.txt
```

## Data
The dataset used in this work is publicly available on Zenodo: 📦 https://doi.org/10.5281/zenodo.17105137

