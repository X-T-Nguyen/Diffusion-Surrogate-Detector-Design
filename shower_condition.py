import uproot
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Define file paths for different energies

test_files = [
    "/scratch/wek54nug/data/total_photon_shower_1_1.root",
    "/scratch/wek54nug/data/total_photon_shower_1_2.root",

    "/scratch/wek54nug/data/total_photon_shower_10_1.root",
    "/scratch/wek54nug/data/total_photon_shower_10_2.root",

    "/scratch/wek54nug/data/total_photon_shower_50_1.root",
    "/scratch/wek54nug/data/total_photon_shower_50_2.root",
    "/scratch/wek54nug/data/total_photon_shower_50_3.root",
    "/scratch/wek54nug/data/total_photon_shower_50_4.root",
    "/scratch/wek54nug/data/total_photon_shower_50_5.root",
    "/scratch/wek54nug/data/total_photon_shower_50_6.root",
    "/scratch/wek54nug/data/total_photon_shower_50_7.root",
    "/scratch/wek54nug/data/total_photon_shower_50_8.root",
    "/scratch/wek54nug/data/total_photon_shower_50_9.root",
    "/scratch/wek54nug/data/total_photon_shower_50_10.root",

    "/scratch/wek54nug/data/total_photon_shower_100_1.root",
    "/scratch/wek54nug/data/total_photon_shower_100_2.root",
    "/scratch/wek54nug/data/total_photon_shower_100_3.root",
    "/scratch/wek54nug/data/total_photon_shower_100_4.root",
    "/scratch/wek54nug/data/total_photon_shower_100_5.root",
    "/scratch/wek54nug/data/total_photon_shower_100_6.root",
    "/scratch/wek54nug/data/total_photon_shower_100_7.root",
    "/scratch/wek54nug/data/total_photon_shower_100_8.root",
    "/scratch/wek54nug/data/total_photon_shower_100_9.root",
    "/scratch/wek54nug/data/total_photon_shower_100_10.root",
    
    "/scratch/wek54nug/data/total_photon_shower_200_1.root",
    "/scratch/wek54nug/data/total_photon_shower_200_2.root",
    "/scratch/wek54nug/data/total_photon_shower_200_3.root",
    "/scratch/wek54nug/data/total_photon_shower_200_4.root",
    "/scratch/wek54nug/data/total_photon_shower_200_5.root",
    "/scratch/wek54nug/data/total_photon_shower_200_6.root",
    "/scratch/wek54nug/data/total_photon_shower_200_7.root",
    "/scratch/wek54nug/data/total_photon_shower_200_8.root",
    "/scratch/wek54nug/data/total_photon_shower_200_9.root",
    "/scratch/wek54nug/data/total_photon_shower_200_10.root",
    "/scratch/wek54nug/data/total_photon_shower_200_11.root",
    "/scratch/wek54nug/data/total_photon_shower_200_12.root",
    "/scratch/wek54nug/data/total_photon_shower_200_13.root",
    "/scratch/wek54nug/data/total_photon_shower_200_14.root",
    "/scratch/wek54nug/data/total_photon_shower_200_15.root",
    "/scratch/wek54nug/data/total_photon_shower_200_16.root",
    "/scratch/wek54nug/data/total_photon_shower_200_17.root",
    "/scratch/wek54nug/data/total_photon_shower_200_18.root",
    "/scratch/wek54nug/data/total_photon_shower_200_19.root",
    "/scratch/wek54nug/data/total_photon_shower_200_20.root"
]




# Define binning for 2D histograms
num_bins = 100
x_range = (-10, 10)
y_range = (-10, 10)
z_range = (-250, 0)

# Create HDF5 file
output_file = "/home/wek54nug/Denoising_ddpm/total_photon_shower.h5"
h5f = h5py.File(output_file, "w")

# Prepare lists to store averaged data
avg_images_xz = []
avg_images_yz = []
avg_labels = []

# Process each file
for root_file in test_files:
    # Extract energy from filename
    energy = int(root_file.split("_")[-2].split(".")[0])

    # Open ROOT file
    with uproot.open(root_file) as file:
        tree = file["photon_sim"]
        data = tree.arrays(["EventID", "x", "y", "z", "dE"], library="np")

        event_ids = np.unique(data["EventID"])  # Unique event IDs
        num_events = len(event_ids)

        # Process events in batches of 5
        batch_size = 5
        num_batches = num_events // batch_size

        for i in range(num_batches):
            # Select event range
            batch_event_ids = event_ids[i * batch_size:(i + 1) * batch_size]

            # Initialize sum histograms
            H_xz_sum = np.zeros((num_bins, num_bins), dtype=np.float32)
            H_yz_sum = np.zeros((num_bins, num_bins), dtype=np.float32)

            for event_id in batch_event_ids:
                mask = data["EventID"] == event_id
                x, y, z, dE = data["x"][mask], data["y"][mask], data["z"][mask], data["dE"][mask]

                # Create 2D histograms
                H_xz, _, _ = np.histogram2d(x, z, bins=num_bins, range=[x_range, z_range], weights=dE)
                H_yz, _, _ = np.histogram2d(y, z, bins=num_bins, range=[y_range, z_range], weights=dE)

                # Accumulate sums
                H_xz_sum += H_xz
                H_yz_sum += H_yz

            # Compute average
            H_xz_avg = H_xz_sum / batch_size
            H_yz_avg = H_yz_sum / batch_size

            # Normalize (optional)
            H_xz_avg /= H_xz_avg.max() if H_xz_avg.max() > 0 else 1
            H_yz_avg /= H_yz_avg.max() if H_yz_avg.max() > 0 else 1

            # Store results
            avg_images_xz.append(H_xz_avg)
            avg_images_yz.append(H_yz_avg)
            avg_labels.append(energy)

# Convert lists to numpy arrays
avg_images_xz = np.array(avg_images_xz, dtype=np.float32)
avg_images_yz = np.array(avg_images_yz, dtype=np.float32)
avg_labels = np.array(avg_labels, dtype=np.int32)

# Save to HDF5
h5f.create_dataset("images_xz", data=avg_images_xz)
h5f.create_dataset("images_yz", data=avg_images_yz)
h5f.create_dataset("labels", data=avg_labels)
h5f.close()

print(f"Saved averaged test HDF5 file: {output_file}")

# ---- Save a 5x5 Grid Image for Testing ----
num_samples = min(25, len(avg_images_xz))  # Ensure we don't exceed dataset size
indices = np.random.choice(len(avg_images_xz), num_samples, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, idx in enumerate(indices):
    ax = axes[i // 5, i % 5]
    ax.imshow(avg_images_xz[idx], cmap="inferno", origin="lower")
    ax.axis("off")
    ax.set_title(f"{avg_labels[idx]} GeV")

plt.tight_layout()
plt.savefig("/home/wek54nug/Denoising_ddpm/avg_test_sample_images_xz.png", dpi=300)
plt.close()

# Repeat for (y, z) images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, idx in enumerate(indices):
    ax = axes[i // 5, i % 5]
    ax.imshow(avg_images_yz[idx], cmap="inferno", origin="lower")
    ax.axis("off")
    ax.set_title(f"{avg_labels[idx]} GeV")

plt.tight_layout()
plt.savefig("/home/wek54nug/Denoising_ddpm/avg_test_sample_images_yz.png", dpi=300)
plt.close()

print("Saved 5x5 test sample images for averaged visualization.")
