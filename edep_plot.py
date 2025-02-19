import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from DiffusionFreeGuidence.ModelCondition import UNet
from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, DDIMSampler
import torch.nn.functional as F  # For resizing (interpolation)

torch.manual_seed(1234)

#############################################
# 1. Utility function to compute profiles
#############################################
def compute_energy_profiles(images):
    """
    Given a batch of images, compute the energy deposit profiles along
    x, y, and z directions.

    Assumes images are normalized in [-1, 1] and have shape 
      (batch, channels, H, W)
    where channels==2 (channel 0 = xz view, channel 1 = yz view).
    
    Returns:
       profile_x:   (W,) energy deposit along x (from xz view)
       profile_y:   (W,) energy deposit along y (from yz view)
       profile_z:   (H,) energy deposit along z (averaged from both views)
    """
    # Scale images back to [0, 1]
    images = (images + 1) / 2.0
    batch, channels, H, W = images.shape

    if channels == 2:
        # x profile from xz view
        profile_x = images[:, 0, :, :].sum(dim=1)  # (batch, W)
        # y profile from yz view
        profile_y = images[:, 1, :, :].sum(dim=1)  # (batch, W)
        # z profile from both views
        profile_z_xz = images[:, 0, :, :].sum(dim=2)  # (batch, H)
        profile_z_yz = images[:, 1, :, :].sum(dim=2)  # (batch, H)
        profile_z = 0.5 * (profile_z_xz + profile_z_yz)
    else:
        raise ValueError("Expected images with 2 channels (both views).")

    # Average over batch
    profile_x_mean = profile_x.mean(dim=0).cpu().numpy()  # shape (W,)
    profile_y_mean = profile_y.mean(dim=0).cpu().numpy()    # shape (W,)
    profile_z_mean = profile_z.mean(dim=0).cpu().numpy()      # shape (H,)
    return profile_x_mean, profile_y_mean, profile_z_mean

#############################################
# 2. Function to compute real data profiles by energy label
#############################################
def get_real_energy_profiles_by_label(h5_file, energy_val, img_size, device, batch_size=25):
    """
    Opens the HDF5 file, filters the images for which the stored label equals energy_val,
    and computes the energy deposit profiles for these images.
    
    Assumes that the raw labels in the file are the energy values (e.g. 1, 10, 50, 100, 200).
    The images are resized to (img_size, img_size) before computing the profiles.
    """
    with h5py.File(h5_file, 'r') as f:
        images_xz = f['images_xz'][:]   # shape: (N, H, W)
        images_yz = f['images_yz'][:]   # shape: (N, H, W)
        labels = f['labels'][:]         # shape: (N,)
    
    # Select indices with the desired energy
    idx = np.where(labels == energy_val)[0]
    if len(idx) == 0:
        raise ValueError(f"No images found for energy {energy_val} GeV")
    idx = idx[:batch_size]  # take at most batch_size images

    imgs_xz = images_xz[idx]
    imgs_yz = images_yz[idx]
    # Convert to tensor; note: original images are assumed in [0,1]
    imgs_xz = torch.tensor(imgs_xz, dtype=torch.float32).unsqueeze(1)  # shape: (B, 1, H, W)
    imgs_yz = torch.tensor(imgs_yz, dtype=torch.float32).unsqueeze(1)  # shape: (B, 1, H, W)
    # Normalize to [-1, 1]
    imgs_xz = imgs_xz / 0.5 - 1.0
    imgs_yz = imgs_yz / 0.5 - 1.0
    # Resize images to (img_size, img_size)
    imgs_xz = F.interpolate(imgs_xz, size=(img_size, img_size), mode='bilinear', align_corners=False)
    imgs_yz = F.interpolate(imgs_yz, size=(img_size, img_size), mode='bilinear', align_corners=False)
    # Stack to form 2-channel images
    imgs = torch.cat([imgs_xz, imgs_yz], dim=1)  # shape: (B, 2, img_size, img_size)
    imgs = imgs.to(device)
    return compute_energy_profiles(imgs)

#############################################
# 3. Functions to load model and generate samples for a given label
#############################################
def load_model_and_sampler(checkpoint_path, modelConfig, device):
    """
    Loads the UNet model and its diffusion sampler from the checkpoint.
    """
    model = UNet(
        T=modelConfig["T"],
        num_labels=6,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    '''
    sampler = GaussianDiffusionSampler(
        model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"],
        w=modelConfig["w"]
    ).to(device)
    '''
    sampler = DDIMSampler(
            model,
            beta_1=modelConfig["beta_1"],
            beta_T=modelConfig["beta_T"],
            T=modelConfig["T"],
            eta= 0.0,        
            ddim_steps= 50   
    ).to(device)
    return model, sampler

def get_generated_energy_profiles_for_label(sampler, modelConfig, device, label, batch_size=25):
    """
    Generates a batch of samples conditioned on the given label.
    
    The model is trained to generate 1-channel images; we duplicate the channel
    to obtain a 2-channel tensor (so it can be used by compute_energy_profiles).
    
    The parameter 'label' here should be in the model’s label space.
    (For example, if the original energies [1,10,50,100,200] were mapped to [0,1,2,3,4],
    then pass the corresponding integer.)
    """
    noise = torch.randn(batch_size, 1, modelConfig["img_size"], modelConfig["img_size"], device=device)
    labels_tensor = torch.full((batch_size,), label, dtype=torch.long, device=device)
    with torch.no_grad():
        gen_imgs = sampler(noise, labels_tensor)
    # Duplicate channel to get 2 channels
    gen_imgs = gen_imgs.repeat(1, 2, 1, 1)
    return compute_energy_profiles(gen_imgs)

#############################################
# 4. Plotting function: grid plot for each energy label
#############################################
def plot_profiles_grid(real_profiles_dict, gen_profiles_dict, energy_values):
    n_labels = len(energy_values)
    # Create a figure with 2 rows (x-profile and z-profile) and n_labels columns
    fig, axs = plt.subplots(2, n_labels, figsize=(4 * n_labels, 8))
    
    for j, energy in enumerate(energy_values):
        # Extract the profiles for the current energy label.
        # Note: real_profiles_dict and gen_profiles_dict both return (x, y, z) profiles.
        # We'll use the x-profile (first element) and the z-profile (third element).
        real_x, _, real_z = real_profiles_dict[energy]
        gen_x, _, gen_z = gen_profiles_dict[energy]
        
        # Create coordinate axes based on the actual profile lengths
        x_axis = np.arange(len(real_x))
        z_axis = np.arange(len(real_z))
        
        # First row: x-profile (longitudinal view)
        axs[0, j].plot(x_axis, real_x, label='Ground truth', linestyle='-', color='blue')
        axs[0, j].plot(x_axis, gen_x, label='Generated', linestyle='-', color='red')
        axs[0, j].set_title(f"{energy} GeV")
        if j == 0:
            axs[0, j].set_ylabel("Edep (GeV) along longitudinal axis")
        axs[0, j].grid(True)
        
        # Second row: z-profile (transverse view)
        axs[1, j].plot(z_axis, real_z, label='Ground truth', linestyle='-', color='blue')
        axs[1, j].plot(z_axis, gen_z, label='Generated', linestyle='-', color='red')
        if j == 0:
            axs[1, j].set_ylabel("Edep (GeV) along transverse axis")
        axs[1, j].set_xlabel("Coordinate")
        axs[1, j].grid(True)
        
        # Add legends only for the first column
        if j == 0:
            axs[0, j].legend()
            axs[1, j].legend()
    
    plt.tight_layout()
    plt.savefig("/home/wek54nug/Denoising_ddpm/evalution_results/energy_profiles_grid_200.png")
    plt.show()


#############################################
# 5. Main evaluation & plotting
#############################################
if __name__ == "__main__":
    # Model configuration (adjust as needed)
    modelConfig = {
        "epoch": 200,
        "batch_size": 25,         # Use 25 for evaluation 
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 2e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 3, # 1.8
        "save_weight_dir": "./CheckpointsCondition_large_batch/",
    }
    
    device = torch.device(modelConfig["device"])
    h5_file = "total_photon_shower.h5"
    checkpoint_path = os.path.join(modelConfig["save_weight_dir"], "ckpt_200.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Define the energy values (in GeV) present in your dataset.
    # We assume that the HDF5 file stores raw energy labels (1, 10, 50, 100, 200).
    energy_values = [1, 10, 50, 100, 200]
    
    # For the generated data, the model is conditioned on label indices.
    # If the original energies are sorted as above, then the mapping is:
    # 1  -> 0, 10 -> 1, 50 -> 2, 100 -> 3, 200 -> 4.
    # (Adjust this mapping if necessary.)
    
    # Compute real data profiles for each energy label.
    real_profiles_dict = {}
    for energy in energy_values:
        real_profiles_dict[energy] = get_real_energy_profiles_by_label(h5_file, energy, modelConfig["img_size"], device)
        print(f"Real profiles computed for energy {energy} GeV.")
    
    # Load the model and sampler once.
    model, sampler = load_model_and_sampler(checkpoint_path, modelConfig, device)
    
    # Compute generated data profiles for each energy label.
    gen_profiles_dict = {}
    for i, energy in enumerate(energy_values):
        # Mapping: energy 1 -> label 0, 10 -> label 1, etc.
        gen_label = i  
        gen_profiles_dict[energy] = get_generated_energy_profiles_for_label(sampler, modelConfig, device, gen_label)
        print(f"Generated profiles computed for energy {energy} GeV (conditioned on label {gen_label}).")
    
    # Plot grid: rows = coordinate (x, y, z), columns = energy values
    plot_profiles_grid(real_profiles_dict, gen_profiles_dict, energy_values)
