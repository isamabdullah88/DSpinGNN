"""
loaddata.py

Parse PyTorch Geometric Data objects.
Author: Isam Balghari
"""

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import logging

import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_distances(dataloader):
    all_cr_cr = []
    all_cr_i = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 1. Extract the 4 raw Cr-I distances per bridge
            # This flattens the [E, 4] tensor into a 1D list of all individual bond legs
            all_cr_i.extend(batch.cr_i_bonds.view(-1).cpu().numpy())
            
            # 2. Calculate Cr-Cr distances (PBC aware)
            src, dst = batch.cr_edge_index
            graphidxs = batch.batch[src]
            
            cell = batch.cell.view(-1, 3, 3)
            bcell = cell[graphidxs]
            
            edge_shift = batch.cr_edge_shift.to(bcell.dtype)
            tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
            
            radvec = batch.pos[dst] - batch.pos[src] + tvec
            dist = radvec.norm(dim=1, keepdim=False)
            
            all_cr_cr.extend(dist.cpu().numpy())
    
    all_cr_cr = np.array(all_cr_cr)
    all_cr_i = np.array(all_cr_i)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Cr-Cr Distances (1NN range)
    ax1.hist(all_cr_cr, bins=50, color='purple', edgecolor='black', alpha=0.7)
    ax1.set_title("Distribution of Cr-Cr Distances")
    ax1.set_xlabel("Distance (Å)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual Cr-I Bond Legs (The 4 distances)
    ax2.hist(all_cr_i, bins=70, color='teal', edgecolor='black', alpha=0.7)
    ax2.set_title("Distribution of Individual Cr-I Legs\n(4 per Cr-Cr edge)")
    ax2.set_xlabel("Distance (Å)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()

    # Final physical bounds for your Radial layers
    print("-" * 30)
    print(f"Cr-Cr Range: {all_cr_cr.min():.4f} to {all_cr_cr.max():.4f} Å")
    print(f"Cr-I  Range: {all_cr_i.min():.4f} to {all_cr_i.max():.4f} Å")
    print("-" * 30)

def plot_cosines(dataloader):
    all_cosines = []
    
    with torch.no_grad():
        for batch in dataloader:
            all_cosines.extend(batch.cr_cr_angles.view(-1).cpu().numpy())
    
    all_cosines = np.array(all_cosines)
    
    plt.figure(figsize=(8, 5))
    # plt.hist(all_cosines, bins=50, color='skyblue', edgecolor='black')
    plt.plot(all_cosines, marker='o', linestyle='', markersize=2)
    plt.title("Distribution of Cosine Angles in Dataset")
    plt.xlabel("Cosine of Cr-Cr-Cr Angle")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    # plt.show()

def plot_j_vs_distance(dataloader):
    """
    Extracts all Cr-Cr distances and J values from a dataloader 
    and plots the spatial distribution of the exchange couplings.
    """
    all_distances = []
    all_j_values = []
    
    # We don't need gradients for visualization
    with torch.no_grad():
        for batch in dataloader:
            
            # 1. Recreate your exact periodic boundary distance math
            src, dst = batch.cr_edge_index
            graphidxs = batch.batch[src]
            
            cell = batch.cell.view(-1, 3, 3)
            bcell = cell[graphidxs]
            
            edge_shift = batch.cr_edge_shift.to(bcell.dtype)
            tvec = torch.einsum('ei, eij -> ej', edge_shift, bcell)
            
            radvec = batch.pos[dst] - batch.pos[src] + tvec
            dist = radvec.norm(dim=1, keepdim=False)

            # yexchange = batch.y_exchange.view(-1)[dist < 4.5]
            # dist = dist[dist < 4.5]
            yexchange = batch.y_exchange.view(-1)
            
            # 2. Extract distances and ground-truth J values
            # Move to CPU and convert to numpy for matplotlib
            all_distances.extend(dist.cpu().numpy())
            all_j_values.extend(yexchange.cpu().numpy())

    # Convert to numpy arrays for plotting
    all_distances = np.array(all_distances)
    all_j_values = np.array(all_j_values)

    # ==========================================
    # Create the Plot
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Use a small alpha (transparency) so you can see density in the clusters
    plt.scatter(all_distances, all_j_values, alpha=0.3, s=15, c='blue', edgecolors='none')
    
    # Draw a vertical line at your 4.0 A cutoff to separate 1NN from the rest
    plt.axvline(x=4.0, color='red', linestyle='--', linewidth=2, label='4.0 Å Cutoff (1NN)')
    
    # Draw a horizontal line at J=0 for reference
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    
    plt.title("Cr-Cr Exchange Coupling vs. Bond Distance", fontsize=16)
    plt.xlabel("Cr-Cr Bond Distance (Å)", fontsize=14)
    plt.ylabel("Exchange Coupling J Magnitude", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure so you can download it from RunPod
    plt.savefig("j_vs_distance_distribution.png", dpi=300, bbox_inches='tight')
    # plt.show()

def getdata(datasetpath, batch_size=32):
    logprefix = "[DATA] "
    logger = logging.getLogger(__name__)

    datalist = torch.load(datasetpath, weights_only=False)

    """
    jvalslist = []
    for data in datalist:
        # if data.exchange:
        jvalslist += data.y_exchange.view(-1).tolist()
    # print('javalslist: ', jvalslist)
    jvals = torch.tensor(jvalslist, dtype=torch.float64)
    mean_jval = jvals.mean()
    logger.info(f"{logprefix}Calculated Mean Exchange J (Baseline): {mean_jval:.4f} meV")

    for data in datalist:
        data.y_exchange = data.y_exchange.to(torch.float64) - mean_jval
        data.y_exchange = data.y_exchange.to(torch.float32)

    jvalslist = []
    for data in datalist:
        jvalslist += data.y_exchange.view(-1).tolist()
    
    jvals = torch.tensor(jvalslist, dtype=torch.float64)
    meanj = jvals.mean()
    logger.info(f"{logprefix}Shifted Dataset Mean Exchange J: {meanj:.6f} meV")
    """

    # trsize = int(1.0 * len(datalist))
    # vsize = len(datalist) - trsize
    # # ttsize = len(datalist) - trsize - vsize
    # ttsize = 0

    # generator = torch.Generator().manual_seed(42)

    # trainlist, valist, testlist = random_split(datalist, [trsize, vsize, ttsize],
    #                            generator=generator)
    trainlist, valist = datalist['train'], datalist['val']
    testlist = []  # No test set for now, but you can create one later

    # Assuming 'all_samples' is your list of PyG graphs
    # min_c = min([graph.cr_cr_angles.min().item() for graph in trainlist + valist])
    # max_c = max([graph.cr_cr_angles.max().item() for graph in trainlist + valist])
    # logger.info(f"Dataset Cosine Range: {min_c:.4f} to {max_c:.4f}")
    
    logger.info(f'{logprefix}Train Set: {len(trainlist)}')
    logger.info(f'{logprefix}Validation Set: {len(valist)}')
    logger.info(f'{logprefix}Test Set: {len(testlist)}')

    trainloader = DataLoader(trainlist, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True)
    valloader = DataLoader(valist, batch_size=batch_size, shuffle=False, num_workers=4,
                           pin_memory=True)
    testloader = DataLoader(testlist, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)

    return trainloader, valloader, testloader

if __name__ == "__main__":
    from logger import getlogger
    logger = getlogger()
    
    trainloader, valloader, testloader = getdata("./DataSets/GNN/Exchange-Relaxed-Striped_2-Rcut_4.5.pth", 32)

    plot_j_vs_distance(trainloader)
    plot_j_vs_distance(valloader)
    plot_cosines(trainloader)
    plot_cosines(valloader)
    plot_distances(trainloader)
    plot_distances(valloader)
    plt.show()

    """
    forces = []
    yexchange = []
    for k, batch in enumerate(trainloader):
        # print('batch y_exchange: ', batch.y_exchange.shape)
        # print('batch exchange: ', batch.exchange.shape)
        # forces.append(torch.norm(batch.y_forces, dim=1))
        logger.info(f"Batch {k}: y_exchange shape: {batch.y_exchange.view(-1).shape}.")
        yexchange += batch.y_exchange.view(-1).tolist()

    # forces = torch.stack(forces)
    yexchange = torch.tensor(yexchange, dtype=torch.float64)

    # print('All forces shape: ', forces.shape)
    import matplotlib.pyplot as plt
    # plt.hist(forces.cpu().numpy().flatten(), bins=100)
    plt.plot(yexchange.cpu().numpy().flatten(), marker='o', linestyle='', markersize=2)
    plt.title('Distribution of Exchange Magnitudes in Training Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Exchange Magnitude')
    plt.show()
    """
