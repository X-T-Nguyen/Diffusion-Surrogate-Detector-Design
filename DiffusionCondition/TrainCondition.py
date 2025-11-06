# TrainCondition.py
import os
import h5py
from typing import Dict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer, DDIMSampler
from DiffusionFreeGuidence.ModelCondition import UNet
#from Scheduler import GradualWarmupScheduler

print("Number of CPU cores available:", os.cpu_count())
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# TrainCondition.py (updated parts)

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, transform=None, use_both_views=False):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.images_xz = self.h5_file['images_xz'][:]  
        self.images_yz = self.h5_file['images_yz'][:]  
        # Load other labels
        self.energy_labels = self.h5_file['labels_energy'][:]  # e.g. [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.xy_labels = self.h5_file['labels_xy'][:]           # e.g. [1, 2, 3, 4, 5]
        self.z_labels = self.h5_file['labels_z'][:]             # e.g. [4, 5, 8, 10, 15]
        self.material_labels = self.h5_file['labels_material'][:]  # e.g. ["PbF2", "PbWO4", ...]

        # Map energy labels to discrete indices (keep as before)
        unique_energy = np.sort(np.unique(self.energy_labels))
        self.energy_mapping = {energy: idx for idx, energy in enumerate(unique_energy)}
        self.energy_labels = np.array([self.energy_mapping[e] for e in self.energy_labels])
        print("Energy mapping:", self.energy_mapping)

        # For xy and z, use continuous values and normalize them to [0, 1]
        self.xy_labels = self.xy_labels.astype(np.float32)
        self.z_labels = self.z_labels.astype(np.float32)
        self.xy_labels = (self.xy_labels - self.xy_labels.min()) / (self.xy_labels.max() - self.xy_labels.min())
        self.z_labels = (self.z_labels - self.z_labels.min()) / (self.z_labels.max() - self.z_labels.min())

        # For material, define a mapping (here pre-training data are all "PbF2")
        self.material_mapping = {"PbF2": 0.0, "PbWO4": 1.0} 
        # Convert material strings to floats
        self.material_labels = np.array([
            self.material_mapping[(m.decode('utf-8').strip() if isinstance(m, bytes) else m.strip())]
            for m in self.material_labels
                ], dtype=np.float32)

        self.transform = transform
        self.use_both_views = use_both_views

    def __len__(self):
        return len(self.energy_labels)

    def __getitem__(self, idx):
        img_xz = self.images_xz[idx]
        img_yz = self.images_yz[idx]
        energy_label = self.energy_labels[idx]   # discrete index (int)
        xy_value = self.xy_labels[idx]           # continuous value in [0, 1]
        z_value = self.z_labels[idx]             # continuous value in [0, 1]
        material_value = self.material_labels[idx]  # now a float (e.g. 0.0)

        # Convert images to tensor and normalize to [-1, 1]
        img_xz = torch.tensor(img_xz, dtype=torch.float32).unsqueeze(0) / 0.5 - 1.0
        img_yz = torch.tensor(img_yz, dtype=torch.float32).unsqueeze(0) / 0.5 - 1.0

        if self.transform is not None:
            img_xz = self.transform(img_xz)
            img_yz = self.transform(img_yz)

        # Use both views if desired
        img = torch.cat([img_xz, img_yz], dim=0) if self.use_both_views else img_xz

        # Return energy as long and xy, z, material as floats
        return img, (torch.tensor(energy_label, dtype=torch.long),
                     torch.tensor(xy_value, dtype=torch.float32),
                     torch.tensor(z_value, dtype=torch.float32),
                     torch.tensor(material_value, dtype=torch.float32))



h5_file = 'total_photon_shower.h5'
DEBUG = False

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    dataset = HDF5Dataset(
        h5_file_path=h5_file,  
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
        ])
    )
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, 
        num_workers=10, drop_last=True, pin_memory=True, persistent_workers=True
    )

    image, label = dataset[0]
    print("Image shape:", image.shape)
    print("Total number of images in HDF5 dataset:", len(dataset))
    print("Start training Conditional DDPMS ...")


    net_model = UNet(
        T=modelConfig["T"], 
        num_energy_labels=11,  
        ch=modelConfig["channel"], 
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"], 
        dropout=modelConfig["dropout"]
    ).to(device)

    ###############################
    # If more than one GPU is available, wrap the model.
    ###############################
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net_model = torch.nn.DataParallel(net_model)
    ###############################
    ###############################

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
    ).to(device)

    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, (energy_labels, xy_values, z_values, material_values) in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = images.to(device)
                energy_labels = energy_labels.to(device)
                xy_values = xy_values.to(device)
                z_values = z_values.to(device)
                material_values = material_values.to(device)

                # Apply null conditioning
                mask_e = torch.rand(energy_labels.size(0), device=device) < 0.1
                energy_labels[mask_e] = 11  
                mask_xy = torch.rand(xy_values.size(0), device=device) < 0.1
                xy_values[mask_xy] = 0.5  
                mask_z = torch.rand(z_values.size(0), device=device) < 0.1
                z_values[mask_z] = 0.5  

               
                loss = trainer(x_0, energy_labels, xy_values, z_values, material_values).mean()  
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()


                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        
        if (e + 1) % 10 == 0:
            torch.save(net_model.state_dict(), os.path.join(modelConfig["save_weight_dir"], f'ckpt_{e + 1}.pt'))

def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    from DiffusionFreeGuidence.ModelCondition import UNet
    model = UNet(
        T=modelConfig["T"],
        num_energy_labels=11,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    ckpt_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"])
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}  # for DataParallel models
    model.load_state_dict(ckpt)
    print("Model weights loaded for evaluation.")
    model.eval()

    sampler = DDIMSampler(
        model,
        beta_1=modelConfig["beta_1"],
        beta_T=modelConfig["beta_T"],
        T=modelConfig["T"],
        eta=1.0,
        ddim_steps=50
    ).to(device)

    energy_conditions =  list(range(11)) #[0, 1, 2, 3, 4]
    # Select specific pairs of xy and z values.
    xy_z_pairs = [(0.0, 0.0), (0.25, 0.0909), (0.5, 0.3636), (0.75, 0.5455), (1.0, 1.0)]
    samples_per_condition = 5

    # Define material conditions to sample for.
    material_conditions = [0.0, 1.0]  # 0.0: PbF2, 1.0: PbWO4

    with torch.no_grad():
        for material_val in material_conditions:
            for xy, z in xy_z_pairs:
                grid_rows = []
                for e in energy_conditions:
                    row_samples = []
                    for i in range(samples_per_condition):
                        energy_label = torch.full((1,), e, device=device, dtype=torch.long)
                        xy_value = torch.tensor([xy], device=device, dtype=torch.float32)
                        z_value = torch.tensor([z], device=device, dtype=torch.float32)
                        # Set material label condition to the current material value.
                        material_label = torch.tensor([material_val], device=device, dtype=torch.float32)

                        noisyImage = torch.randn(1, 1, modelConfig["img_size"], modelConfig["img_size"], device=device)
                        sampledImg = sampler(noisyImage, energy_label, xy_value, z_value, material_label)
                        sampledImg = sampledImg * 0.5 + 0.5
                        row_samples.append(sampledImg.cpu())
                    row = torch.cat(row_samples, dim=3)
                    grid_rows.append(row)
                grid = torch.cat(grid_rows, dim=2)
                grid_filename = os.path.join(
                    modelConfig["sampled_dir"],
                    f"energy_grid_material{material_val}_xy{xy}_z{z}.png"
                )
                save_image(grid, grid_filename, nrow=samples_per_condition)
                print(f"Saved grid for material {material_val}, xy {xy} and z {z} at {grid_filename}")
