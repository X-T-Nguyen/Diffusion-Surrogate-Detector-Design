import uproot
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import glob

##############################
#   Data Processing Section  #
##############################

# Flag to use manual overrides for the histogram ranges
use_manual_overrides = True

# Dictionary of manual overrides for each (xy, z) pair.
manual_range_overrides = {
    (1, 5):  {"x_range": (-10, 10), "y_range": (-10, 10), "z_range": (-100, 50)},
    (2, 4):  {"x_range": (-10, 10), "y_range": (-10, 10), "z_range": (-80, 80)},
    (3, 8):  {"x_range": (-10, 10), "y_range": (-10, 10), "z_range": (-195, 10)},
    (4, 10): {"x_range": (-10, 10), "y_range": (-10, 10), "z_range": (-250, 10)},
    (5, 15): {"x_range": (-15, 15), "y_range": (-15, 15), "z_range": (-350, 10)},
}

# Scaling factors for automatic range.
scaleXY = 5.0   # For XY: originally, 4 cm -> (-10,10)
scaleZ  = 25.0  # For Z: originally, 10 cm -> (-250,0)

# Energy mapping dictionary (values in MeV to label index).
#energy_dict = {1000: 0, 10000: 1, 50000: 2, 100000: 3, 200000: 4}
energy_dict = {
    1000: 0, 10000: 1, 20000: 2, 30000: 3, 40000: 4,
    50000: 5, 60000: 6, 70000: 7, 80000: 8, 90000: 9, 100000: 10}

# List of ROOT files to process.
root_files = glob.glob("/scratch/wek54nug/large_files/*.root")

# Fixed number of histogram bins.
num_bins = 100

# Output HDF5 file.
output_file = "/home/wek54nug/Denoising_ddpm/total_photon_shower.h5"
h5f = h5py.File(output_file, "w")

# Lists to store averaged images and labels.
avg_images_xz = []
avg_images_yz = []
avg_energy_labels = []  # Energy labels (mapped: 0-4)
avg_xy_labels = []      # XY sizes (granularity, e.g. 2,3,4,5,6)
avg_z_labels = []       # Z depths (e.g. 6,10,15,20)
avg_material_labels = []  # Material labels (e.g. "PbF2" or "PbWO4")

# Process each ROOT file.
for root_file in root_files:
    filename = os.path.basename(root_file)
    parts = filename.split("_")
    
    # Extract material, xy, and z from the filename.
    try:
        # Check if the fourth token starts with "xy". If so, no material is specified,
        # so we default to "PbF2" (or any default you choose).
        if parts[3].startswith("xy"):
            material = "PbF2"
            xy = int(parts[3].lstrip("xy"))
            z  = int(parts[4].lstrip("z"))
        else:
            # Otherwise, the fourth token is the material name.
            material = parts[3]
            xy = int(parts[4].lstrip("xy"))
            z  = int(parts[5].lstrip("z"))
    except Exception as ex:
        print(f"Error parsing xy/z/material from {filename}: {ex}")
        continue

    # Set histogram ranges based on manual overrides or automatic scaling.
    if use_manual_overrides and (xy, z) in manual_range_overrides:
        current_x_range = manual_range_overrides[(xy, z)]["x_range"]
        current_y_range = manual_range_overrides[(xy, z)]["y_range"]
        current_z_range = manual_range_overrides[(xy, z)]["z_range"]
        print(f"Manual override for ({xy}, {z}) in file {filename}:")
    else:
        current_x_range = (-(xy * scaleXY) / 2.0, (xy * scaleXY) / 2.0)
        current_y_range = (-(xy * scaleXY) / 2.0, (xy * scaleXY) / 2.0)
        current_z_range = (-(z * scaleZ), 0)
        print(f"Automatic ranges for ({xy}, {z}) in file {filename}:")

    print(f"  File: {filename}")
    print(f"  Material: {material}")
    print(f"  x_range: {current_x_range}")
    print(f"  y_range: {current_y_range}")
    print(f"  z_range: {current_z_range}")

    # Open the ROOT file and read the arrays.
    with uproot.open(root_file) as file:
        tree = file["photon_sim"]
        data = tree.arrays(["EventID", "primaryE", "x", "y", "z", "dE"], library="np")
        event_ids = np.unique(data["EventID"])
        num_events = len(event_ids)

        # Process events in batches (here using batch_size = 1).
        batch_size = 1
        num_batches = num_events // batch_size

        for i in range(num_batches):
            batch_event_ids = event_ids[i * batch_size:(i + 1) * batch_size]

            H_xz_sum = np.zeros((num_bins, num_bins), dtype=np.float32)
            H_yz_sum = np.zeros((num_bins, num_bins), dtype=np.float32)
            batch_energies = []

            for event_id in batch_event_ids:
                mask = data["EventID"] == event_id
                x = data["x"][mask]
                y = data["y"][mask]
                z_data = data["z"][mask]
                dE = data["dE"][mask]

                primaryE_values = data["primaryE"][mask]
                if primaryE_values.size > 0:
                    batch_energies.append(np.median(primaryE_values))

                H_xz, _, _ = np.histogram2d(x, z_data, bins=num_bins, 
                                            range=[current_x_range, current_z_range], 
                                            weights=dE)
                H_yz, _, _ = np.histogram2d(y, z_data, bins=num_bins, 
                                            range=[current_y_range, current_z_range], 
                                            weights=dE)
                H_xz_sum += H_xz
                H_yz_sum += H_yz

            H_xz_avg = H_xz_sum / batch_size
            H_yz_avg = H_yz_sum / batch_size

            # Normalize histograms if needed.
            if H_xz_avg.max() > 0:
                H_xz_avg /= H_xz_avg.max()
            if H_yz_avg.max() > 0:
                H_yz_avg /= H_yz_avg.max()

            # Determine the energy label from primaryE.
            if len(batch_energies) > 0:
                avg_energy = np.median(batch_energies)
                avg_energy_int = int(round(avg_energy))
                if avg_energy_int in energy_dict:
                    energy_label = energy_dict[avg_energy_int]
                else:
                    possible = np.array(list(energy_dict.keys()))
                    closest = possible[np.argmin(np.abs(possible - avg_energy_int))]
                    energy_label = energy_dict[closest]
            else:
                energy_label = -1

            avg_images_xz.append(H_xz_avg)
            avg_images_yz.append(H_yz_avg)
            avg_energy_labels.append(energy_label)
            avg_xy_labels.append(xy)
            avg_z_labels.append(z)
            avg_material_labels.append(material)

    # Print progress after finishing each file.
    print(f"Finished processing file: {filename}")

# Convert lists to numpy arrays.
avg_images_xz = np.array(avg_images_xz, dtype=np.float32)
avg_images_yz = np.array(avg_images_yz, dtype=np.float32)
avg_energy_labels = np.array(avg_energy_labels, dtype=np.int32)
avg_xy_labels = np.array(avg_xy_labels, dtype=np.int32)
avg_z_labels = np.array(avg_z_labels, dtype=np.int32)

# When saving string data with h5py, define a special variable-length string dtype.
str_dt = h5py.special_dtype(vlen=str)
h5f.create_dataset("images_xz", data=avg_images_xz)
h5f.create_dataset("images_yz", data=avg_images_yz)
h5f.create_dataset("labels_energy", data=avg_energy_labels)
h5f.create_dataset("labels_xy", data=avg_xy_labels)
h5f.create_dataset("labels_z", data=avg_z_labels)
h5f.create_dataset("labels_material", data=np.array(avg_material_labels, dtype=object), dtype=str_dt)
h5f.close()

print(f"Saved averaged HDF5 file: {output_file}")

##############################
#      Plotting Section      #
##############################

# Open the HDF5 file and load the datasets.
with h5py.File("total_photon_shower.h5", "r") as f:
    images_xz       = f["images_xz"][:]        # shape: (n_samples, height, width)
    labels_energy   = f["labels_energy"][:]      # energy labels: integers 0-4 (0: 1 GeV, etc.)
    labels_xy       = f["labels_xy"][:]          # granularity values (e.g. 2,3,4,5,6)
    labels_material = f["labels_material"][:]    # material labels (e.g. b'PbF2', b'PbWO4')

# Decode material labels from bytes to strings if needed.
labels_material = np.array([lm.decode('utf-8') if isinstance(lm, bytes) else lm for lm in labels_material])

# Mapping for energy levels.
#energy_mapping = {0: "1 GeV", 1: "10 GeV", 2: "50 GeV", 3: "100 GeV", 4: "200 GeV"}
energy_mapping ={0: "1 GeV", 1: "10 GeV", 2: "20 GeV", 3: "30 GeV", 4: "40 GeV", 5: "50 GeV", 6: "60 GeV", 7: "70 GeV", 8: "80 GeV", 9: "90 GeV", 10: "100 GeV"}

# Get unique materials and plot for each.
unique_materials = np.unique(labels_material)
print("Unique materials found for plotting:", unique_materials)

for material in unique_materials:
    # Filter events for the current material.
    mask_material = labels_material == material
    if not np.any(mask_material):
        print(f"No data found for material: {material}")
        continue

    material_images = images_xz[mask_material]
    material_energy = labels_energy[mask_material]
    material_xy     = labels_xy[mask_material]

    # Get unique granularity (XY) and energy levels.
    unique_xy = np.sort(np.unique(material_xy))
    unique_energy = np.sort(np.unique(material_energy))
    
    # Create a grid of subplots (rows: granularity, columns: energy levels).
    fig, axes = plt.subplots(len(unique_xy), len(unique_energy), figsize=(15, 15))
    
    for i, xy_val in enumerate(unique_xy):
        for j, energy_val in enumerate(unique_energy):
            current_mask = (material_xy == xy_val) & (material_energy == energy_val)
            indices = np.where(current_mask)[0]
            ax = axes[i, j]
            
            if len(indices) > 0:
                # Use the first image found for this combination.
                img = material_images[indices[0]]
                ax.imshow(img, cmap="inferno", origin="lower")
            else:
                ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set column titles for energy levels (top row).
            if i == 0:
                ax.set_title(energy_mapping.get(energy_val, f"E{energy_val}"), fontsize=10)
            # Set row labels for granularity (first column).
            if j == 0:
                ax.set_ylabel(f"XY: {xy_val}", fontsize=10)
    
    plt.suptitle(f"Material: {material}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"/home/wek54nug/Denoising_ddpm/plot_material_{material}.png", dpi=300)
    plt.close()

print("Plots generated for each material (PbF2 and PbWO4).")
