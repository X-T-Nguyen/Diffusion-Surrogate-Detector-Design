#!/usr/bin/env python
import os
import sys
import random

# ── Make your project root importable ────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
from torchvision.utils import save_image

from DiffusionFreeGuidence.ModelCondition import UNet
from DiffusionFreeGuidence.DiffusionCondition import DDIMSampler

# ▶️ Configuration ◀️
H5_PATH    = "/home/wek54nug/Denoising_ddpm/total_photon_shower.h5"
CKPT_PATH  = "../CheckpointsCondition/ckpt_1800.pt"
OUTDIR     = "./comparison_plots"
IMG_SIZE   = 32    # output pixels
XY_RAW     = 5     # cm
Z_RAW      = 15    # cm
MATERIAL   = 0.0   # 0→PbF2, 1→PbWO4
# ──────────────────────────────────────────────────────────────────────────────

energy_mapping = {
    0:"1 GeV",1:"10 GeV",2:"20 GeV",3:"30 GeV",4:"40 GeV",
    5:"50 GeV",6:"60 GeV",7:"70 GeV",8:"80 GeV",9:"90 GeV",10:"100 GeV"
}

# ensure randomness at every run
np.random.seed(None)
random.seed(None)
torch.manual_seed(torch.seed())

# 1) Load H5 and extract the (XY=1,Z=5) images per energy
with h5py.File(H5_PATH, "r") as f:
    imgs100 = f["images_xz"][:]        # (N,100,100)
    ens     = f["labels_energy"][:]    # (N,)
    xys     = f["labels_xy"][:]        # (N,)
    zs      = f["labels_z"][:]         # (N,)
    mats    = f["labels_material"][:]  # (N,)

# decode bytes→str if needed
mats = np.array([m.decode() if isinstance(m,bytes) else m for m in mats])

# pick the right material string
target_mat = np.unique(mats)[int(MATERIAL)]
mask = (mats==target_mat) & (xys==XY_RAW) & (zs==Z_RAW)

# Select desired energies
#desired = [0, 1, 5, 7, 10]   # corresponds to 1, 10, 50, 80, 100 GeV
# desired = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
desired = [0,1,2,3,4,5]
energies = [e for e in desired if e in ens[mask]]
nE = len(energies)

h5_images = []
for e in energies:
    idx = np.where(mask & (ens==e))[0]
    if idx.size:
        rand_idx = np.random.choice(idx)  # ✅ random image each run
        img32 = resize(imgs100[rand_idx], (IMG_SIZE, IMG_SIZE),
                       mode="reflect", anti_aliasing=True)
    else:
        img32 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    h5_images.append(img32)

# 2) Load DDPM sampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {
    "T":500, "channel":32, "channel_mult":[1,2,2,2],
    "num_res_blocks":2, "dropout":0.15,
    "beta_1":1e-4, "beta_T":0.028, "eta":0.0,
    "ddim_steps":50, "checkpoint": CKPT_PATH
}
def load_sampler(cfg, device):
    net = UNet(
        T=cfg["T"], num_energy_labels=11,
        ch=cfg["channel"], ch_mult=cfg["channel_mult"],
        num_res_blocks=cfg["num_res_blocks"], dropout=cfg["dropout"]
    ).to(device)
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    ckpt = {k.replace("module.",""):v for k,v in ckpt.items()}
    net.load_state_dict(ckpt); net.eval()
    return DDIMSampler(net, beta_1=cfg["beta_1"], beta_T=cfg["beta_T"],
                       T=cfg["T"], eta=cfg["eta"], ddim_steps=cfg["ddim_steps"]).to(device)

sampler = load_sampler(cfg, device)

# normalized XY,Z for conditioning
xy_norm = (XY_RAW - 1) / (5 - 1)
z_norm  = (Z_RAW  - 4) / (15 - 4)

ddpm_images = []
with torch.no_grad():
    for e in energies:
        x0 = torch.randn(1,1,IMG_SIZE,IMG_SIZE,device=device)  # ✅ random noise each run
        ce = torch.tensor([e], device=device, dtype=torch.long)
        cxy= torch.tensor([xy_norm], device=device)
        cz = torch.tensor([z_norm ], device=device)
        cm = torch.tensor([MATERIAL], device=device)
        out = sampler(x0, ce, cxy, cz, cm)
        img = out.clamp(-1,1).add(1).div(2).cpu().squeeze().numpy()
        ddpm_images.append(img)

# 3) Plot 2×nE comparison with row labels
os.makedirs(OUTDIR, exist_ok=True)
fig, axes = plt.subplots(2, nE, figsize=(2*nE, 4), squeeze=False)

for j, e in enumerate(energies):
    # Top row: Ground truth
    ax = axes[0, j]
    ax.imshow(h5_images[j], cmap="inferno", origin="lower")
    ax.set_title(energy_mapping[e], fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    
    # Bottom row: DDPM
    ax2 = axes[1, j]
    ax2.imshow(ddpm_images[j], cmap="inferno", origin="lower")
    ax2.set_xticks([]); ax2.set_yticks([])

# Row labels
axes[0, 0].set_ylabel("Ground truth",   fontsize=10, rotation="vertical", labelpad=12)
axes[1, 0].set_ylabel("DDPM generated", fontsize=10, rotation="vertical", labelpad=12)

plt.suptitle(f"Cell configuration {XY_RAW}×{XY_RAW}×{Z_RAW}cm³", fontsize=12)
plt.tight_layout(rect=[0.02, 0, 1, 0.95])

out_path = os.path.join(OUTDIR, f"compare_XY{XY_RAW}_Z{Z_RAW}_new.png")
plt.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved comparison plot to {out_path}")
