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
    plt.show()

def getdata(datasetpath, batch_size=32):
    logprefix = "[DATA] "
    logger = logging.getLogger(__name__)

    datalist = torch.load(datasetpath, weights_only=False)

    # Assuming 'dataset' is your list of Data() objects
    logger.info(f"{logprefix}Pre-processing dataset to eliminate the float32 precision trap...")

    # 1. Extract all energies using float64 to ensure perfect precision during the sum
    # all_energies = torch.tensor([data.y_energy.item() for data in datalist], dtype=torch.float64)
    # mean_energy = all_energies.mean()

    jvalslist = []
    for data in datalist:
        # if data.exchange:
        jvalslist += data.y_exchange.view(-1).tolist()
    # print('javalslist: ', jvalslist)
    jvals = torch.tensor(jvalslist, dtype=torch.float64)
    mean_jval = jvals.mean()
    logger.info(f"{logprefix}Calculated Mean Exchange J (Baseline): {mean_jval:.4f} meV")

    # logger.info(f"{logprefix}Calculated Mean Energy (Baseline): {mean_energy:.4f} eV")

    # 2. Shift every graph individually and convert back to float32
    # (We subtract the float64 mean, then cast the result back to float32 for fast training)
    for data in datalist:
        data.y_exchange = data.y_exchange.to(torch.float64) - mean_jval
        data.y_exchange = data.y_exchange.to(torch.float32)

    # 3. Verify the shift worked (the new mean should be mathematically zero)
    # shifted_mean = torch.tensor([data.y_energy.item() for data in datalist]).mean()
    # logger.info(f"{logprefix}Shifted Dataset Mean: {shifted_mean:.6f} eV")

    jvalslist = []
    for data in datalist:
        jvalslist += data.y_exchange.view(-1).tolist()
    
    jvals = torch.tensor(jvalslist, dtype=torch.float64)
    meanj = jvals.mean()
    logger.info(f"{logprefix}Shifted Dataset Mean Exchange J: {meanj:.6f} meV")
    
    trsize = int(0.85 * len(datalist))
    vsize = len(datalist) - trsize
    # ttsize = len(datalist) - trsize - vsize
    ttsize = 0

    generator = torch.Generator().manual_seed(42)

    trainlist, valist, testlist = random_split(datalist, [trsize, vsize, ttsize],
                               generator=generator)
    
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
    
    trainloader, valloader, testloader = getdata("./DataSets/GNN/Rattled-Exchange-Full-Normal-Rcut_4.5.pth", 32)

    plot_j_vs_distance(trainloader)

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
