import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from DiffusionFreeGuidence.ModelCondition import UNet
from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler
from DiffusionFreeGuidence.DiffusionCondition import DDIMSampler

# Check if the HDF5 file exists and print its header.
print(os.path.exists("total_photon_shower.h5"))
with open("total_photon_shower.h5", "rb") as f:
    header = f.read(8)
    print(header)

# --------------------------
# Metric Functions
# --------------------------

def compute_total_energy(image_batch):
    """
    Compute total deposited energy by summing pixel values.
    Assumes images are in [-1, 1] so scales them back to [0,1].
    """
    image_batch = (image_batch + 1) / 2  # Scale back to [0,1]
    return image_batch.sum(dim=[1, 2, 3])  # Sum over (C, H, W)

def compute_shower_center_x(image_batch):
    """
    Compute the center of the shower along the x-axis.
    """
    image_batch = (image_batch + 1) / 2  # Scale back to [0,1]
    batch_size, _, height, width = image_batch.shape
    x_indices = torch.arange(width, device=image_batch.device).view(1, 1, 1, width)
    energy = image_batch.sum(dim=-2, keepdim=True)  # Sum over height (y-axis)
    shower_center_x = (x_indices * energy).sum(dim=[2, 3]) / (energy.sum(dim=[2, 3]) + 1e-8)
    return shower_center_x.view(batch_size)

def compute_sigma_y_bar(image_batch):
    """
    (Legacy function: uncentered second moment along y)
    Compute sigma_y_bar for an image batch using the second moment method.
    This function computes the uncentered second moment along y.
    """
    image_batch = (image_batch + 1) / 2  # Scale back to [0,1]
    batch_size, _, height, width = image_batch.shape
    y_coords = torch.arange(height, dtype=torch.float32, device=image_batch.device)
    I_y = image_batch.sum(dim=-1, keepdim=True)  # Sum over width (x-axis)
    sigma_y = (y_coords**2).view(1, 1, -1, 1) * I_y
    sigma_y_bar = sigma_y.sum(dim=2).squeeze(1)
    return sigma_y_bar

def compute_sigma_y(image_batch):
    """
    Compute the energy-weighted standard deviation (sigma_y) along the y-axis for an image batch.
    This measures the spread of the energy deposition along the y-axis.
    """
    # Scale images from [-1, 1] to [0, 1]
    image_batch = (image_batch + 1) / 2  
    batch_size, channels, height, width = image_batch.shape

    # Create a tensor of y coordinates with shape (1, 1, height, 1) for broadcasting.
    y_coords = torch.arange(height, dtype=torch.float32, device=image_batch.device).view(1, 1, height, 1)
    
    # Sum over the x-axis to obtain the energy profile along y.
    I_y = image_batch.sum(dim=-1, keepdim=True)  # shape: (batch_size, channels, height, 1)
    
    # Compute the total energy per image along y.
    total_energy = I_y.sum(dim=2, keepdim=True)  # shape: (batch_size, channels, 1, 1)
    
    # Compute the energy-weighted mean y position.
    mean_y = (y_coords * I_y).sum(dim=2, keepdim=True) / (total_energy + 1e-8)
    
    # Compute the variance: the energy-weighted second central moment.
    variance = ((y_coords - mean_y) ** 2 * I_y).sum(dim=2, keepdim=True) / (total_energy + 1e-8)
    
    # Compute standard deviation (spread) along y.
    sigma_y = torch.sqrt(variance)
    
    # Return a tensor of shape (batch_size,)
    return sigma_y.view(batch_size)

# --------------------------
# HDF5 Dataset Class
# --------------------------

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file_path, transform=None, use_both_views=False):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.images_xz = self.h5_file['images_xz'][:]
        self.images_yz = self.h5_file['images_yz'][:]
        self.labels = self.h5_file['labels'][:]

        # Map labels to new indices (if needed)
        unique_labels = np.unique(self.labels)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([self.label_mapping[label] for label in self.labels])
        self.transform = transform
        self.use_both_views = use_both_views

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_xz = self.images_xz[idx]  # shape: (100, 100)
        img_yz = self.images_yz[idx]  # shape: (100, 100)
        label = self.labels[idx]
        img_xz = torch.tensor(img_xz, dtype=torch.float32).unsqueeze(0) / 0.5 - 1.0
        img_yz = torch.tensor(img_yz, dtype=torch.float32).unsqueeze(0) / 0.5 - 1.0
        if self.transform is not None:
            img_xz = self.transform(img_xz)
            img_yz = self.transform(img_yz)
        img = torch.cat([img_xz, img_yz], dim=0) if self.use_both_views else img_xz
        return img, label

# --------------------------
# Evaluation and Plotting Functions
# --------------------------

def get_ground_truth_metrics(h5_file, batch_size, transform, device):
    """
    Loads a batch of real images from the HDF5 dataset and computes average metric values.
    These values serve as the "target" ground-truth for the MSE computation.
    """
    dataset = HDF5Dataset(h5_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images, _ = next(iter(dataloader))  # take the first batch
    images = images.to(device)
    gt_total_energy = compute_total_energy(images).mean().item()
    gt_shower_center_x = compute_shower_center_x(images).mean().item()
    # Use the improved sigma_y function to compute the energy spread along y.
    gt_sigma_y = compute_sigma_y(images).mean().item()
    return gt_total_energy, gt_shower_center_x, gt_sigma_y

def evaluate_checkpoint(checkpoint_path, modelConfig, device, gt_metrics):
    """
    Loads a given checkpoint, samples a batch of images, computes the metrics, and
    returns the MSE between the generated metrics and the provided ground-truth metrics.
    """
    gt_total_energy, gt_shower_center_x, gt_sigma_y = gt_metrics

    # Initialize the model (ensure the hyperparameters match your training configuration)
    model = UNet(
        T=modelConfig["T"],
        num_labels=6,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    
    # Load weights from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Set up the sampler
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
        

    # Use a fixed noise tensor and fixed labels for consistent evaluation.
    batch_size = modelConfig["batch_size"]
    noise = torch.randn(batch_size, 1, modelConfig["img_size"], modelConfig["img_size"], device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # or use a balanced set of labels

    with torch.no_grad():
        generated_imgs = sampler(noise, labels)

    # Compute the metrics on the generated images.
    gen_total_energy = compute_total_energy(generated_imgs).mean().item()
    gen_shower_center_x = compute_shower_center_x(generated_imgs).mean().item()
    # Compute the energy spread using the improved sigma_y function.
    gen_sigma_y = compute_sigma_y(generated_imgs).mean().item()

    # Compute the Mean Squared Error (MSE) for each metric.
    mse_total_energy = (gen_total_energy - gt_total_energy) ** 2
    mse_shower_center_x = (gen_shower_center_x - gt_shower_center_x) ** 2
    mse_sigma_y = (gen_sigma_y - gt_sigma_y) ** 2

    return mse_total_energy, mse_shower_center_x, mse_sigma_y

def evaluate_all_checkpoints(modelConfig, h5_file):
    """
    Loops over all checkpoint files, computes the MSE metrics for each, and returns a dictionary.
    """
    device = torch.device(modelConfig["device"])
    # Use the same transform as used in training (adjust if needed)
    transform = transforms.Compose([
        transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
        transforms.Lambda(lambda t: t.repeat(1, 1, 1)),  # expand channels if necessary
    ])

    # Get ground-truth metrics from real images.
    gt_metrics = get_ground_truth_metrics(h5_file, modelConfig["batch_size"], transform, device)
    print("Ground truth metrics (from dataset):", gt_metrics)

    checkpoints_dir = modelConfig["save_weight_dir"]
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoints_dir) if f.startswith("ckpt_") and f.endswith(".pt")],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    results = {}
    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(checkpoints_dir, ckpt)
        mse_total, mse_center, mse_sigma = evaluate_checkpoint(ckpt_path, modelConfig, device, gt_metrics)
        results[ckpt] = {
            "mse_total_energy": mse_total,
            "mse_shower_center_x": mse_center,
            "mse_sigma_y": mse_sigma
        }
        print(f"Checkpoint: {ckpt}")
        print(f"  MSE Total Energy:     {mse_total:.6f}")
        print(f"  MSE Shower Center X:  {mse_center:.6f}")
        print(f"  MSE Sigma Y:          {mse_sigma:.6f}\n")
    return results

def plot_metrics_vs_epoch(results):
    """
    Given a results dictionary with checkpoint names as keys and metric MSEs as values,
    plot the metrics versus epoch number in a single figure with three side-by-side plots.
    """
    epochs = []
    mse_total_energy = []
    mse_shower_center_x = []
    mse_sigma_y = []

    # Extract epoch number from the checkpoint filename, e.g. "ckpt_5.pt" -> epoch 5.
    for ckpt, metrics in results.items():
        try:
            epoch = int(ckpt.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            continue  # skip files that don't match the expected naming pattern
        epochs.append(epoch)
        mse_total_energy.append(metrics['mse_total_energy'])
        mse_shower_center_x.append(metrics['mse_shower_center_x'])
        mse_sigma_y.append(metrics['mse_sigma_y'])

    # Sort by epoch.
    sorted_indices = np.argsort(epochs)
    epochs = np.array(epochs)[sorted_indices]
    mse_total_energy = np.array(mse_total_energy)[sorted_indices]
    mse_shower_center_x = np.array(mse_shower_center_x)[sorted_indices]
    mse_sigma_y = np.array(mse_sigma_y)[sorted_indices]

    # Create the figure with three subplots side-by-side.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].plot(epochs, mse_total_energy, linestyle='-')
    axs[0].set_title('Total Energy MSE vs Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel(r'$MSE_{E}$')
    
    axs[1].plot(epochs, mse_shower_center_x, linestyle='-')
    axs[1].set_title('Shower Center X MSE vs Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel(r'$MSE_{\bar{X}}$')
    
    axs[2].plot(epochs, mse_sigma_y, linestyle='-')
    axs[2].set_title('Sigma Y MSE vs Epoch')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel(r'$MSE_{\sigma_y}$')
    
    # Add gridlines to all subplots.
    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("/home/wek54nug/Denoising_ddpm/evalution_results/metrics_vs_epoch.png")
    plt.show()

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    modelConfig = {
        "state": "eval", 
        "epoch": 200,
        "batch_size": 25,  # batch_size for eval = 25 (maximum for A100 GPU)
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],  # [1, 2, 2, 2]
        "num_res_blocks": 2, 
        "dropout": 0.15,
        "lr": 2e-4,  # 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 128,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8, 
        "save_weight_dir": "./CheckpointsCondition_large_batch/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_10.pt",
        "sampled_dir": "./Test_img_condition/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 5
    }

    # Path to your HDF5 dataset file (for computing ground-truth metrics)
    h5_file = "total_photon_shower.h5"

    # Evaluate all checkpoints and compute the MSE metrics for each checkpoint.
    results = evaluate_all_checkpoints(modelConfig, h5_file)
    print("Results:", results)

    # Plot the three metrics vs. epoch in one figure with three side-by-side plots.
    plot_metrics_vs_epoch(results)
