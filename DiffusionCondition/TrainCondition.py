

import os
import h5py
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, transform=None, use_both_views=False):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.images_xz = self.h5_file['images_xz'][:]  
        self.images_yz = self.h5_file['images_yz'][:]  
        self.labels = self.h5_file['labels'][:] 

        # Print original labels and their counts
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("Original unique labels and their counts:")
        for label, count in zip(unique_labels, counts):
            print(f"Label: {label}, Count: {count}")

       # Map the labels (e.g., 5, 10) to 0, 1
        unique_labels = np.unique(self.labels)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([self.label_mapping[label] for label in self.labels])

        self.transform = transform
        self.use_both_views = use_both_views

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_xz = self.images_xz[idx]  # Shape: (100, 100)
        img_yz = self.images_yz[idx]  # Shape: (100, 100)
        label = self.labels[idx]  # Integer label (0 or 1)

        # Convert to Tensor and Normalize [-1, 1]
        img_xz = torch.tensor(img_xz, dtype=torch.float32).unsqueeze(0) / 0.5 - 1.0  # Shape: (1, 100, 100)
        img_yz = torch.tensor(img_yz, dtype=torch.float32).unsqueeze(0) / 0.5 - 1.0  # Shape: (1, 100, 100)

        if self.transform is not None:
            img_xz = self.transform(img_xz)  # Apply transform
            img_yz = self.transform(img_yz)  # Apply transform

        if self.use_both_views:
            img = torch.cat([img_xz, img_yz], dim=0)  # Shape: (2, 100, 100) if using both views
        else:
            img = img_xz  # Only use XZ view

        return img, label



h5_file = 'total_photon_shower.h5'
DEBUG = False


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    dataset = HDF5Dataset(
        h5_file_path=h5_file,  
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Lambda(lambda t: t.repeat(1, 1, 1)),  # Convert 1 channel to 3 channels
            #transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize 
        ])
    )
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=1, drop_last=True, pin_memory=True
    )
    
    image, label = dataset[0]
    print("Image shape:", image.shape)
    print("Total number of images in HDF5 dataset:", len(dataset))
    print("Start training Conditional DDPMS ...")


    # model setup
    net_model = UNet(
        T=modelConfig["T"], 
        num_labels=6, 
        ch=modelConfig["channel"], 
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"], 
        dropout=modelConfig["dropout"]
        ).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), 
        lr=modelConfig["lr"], 
        weight_decay=1e-4)

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=modelConfig["epoch"], 
        eta_min=0, 
        last_epoch=-1)

    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, 
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 20, 
        after_scheduler=cosineScheduler)

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], 
        modelConfig["beta_T"], 
        modelConfig["T"]
        ).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) 
                # Set 10% of labels to null class (5)
                mask = torch.rand(labels.size(0), device=device) < 0.1
                labels[mask] = 5  # Use 5 as null class

                '''
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                '''
                    
                #loss = trainer(x_0, labels).sum() / b ** 2.
                loss = trainer(x_0, labels).mean()  # Directly averages over batch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        # Save model every 10 epochs
        if (e + 1) % 10 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{e + 1}.pt'
            ))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    
    # Load model and evaluate
    with torch.no_grad():
        # Generate balanced labels:
        num_labels = 5  # num_labels = modelConfig.get("num_labels", 5)
        batch_size = modelConfig["batch_size"]
        labels = torch.arange(batch_size) % num_labels
        labels = labels.to(device)

        print("labels: ", labels)
        
        # Model setup
        model = UNet(
            T=modelConfig["T"], 
            num_labels=6, 
            ch=modelConfig["channel"], 
            ch_mult=modelConfig["channel_mult"],
            num_res_blocks=modelConfig["num_res_blocks"], 
            dropout=modelConfig["dropout"]
            ).to(device)

        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)

        model.load_state_dict(ckpt)
        print("model load weight done.")

        model.eval()
        torch.manual_seed(1234) # add the seed
        '''
        #DDPMS sampler
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        '''

        # DDIM sampler
        sampler = DDIMSampler(
            model,
            beta_1=modelConfig["beta_1"],
            beta_T=modelConfig["beta_T"],
            T=modelConfig["T"],
            eta= 1.0,        
            ddim_steps= 50   
        ).to(device)
        
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 1, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        
        print(noisyImage[0][0][0][:5])

        # Generate the sampled images
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # Normalize to [0, 1]
        print(sampledImgs)
        
        # Save the sampled images
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

        #generated_images = sampler(noise, labels).cpu()
        #generated_images = torch.clamp(generated_images, -1, 1)
