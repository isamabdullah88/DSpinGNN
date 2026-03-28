"""
loaddata.py

Parse PyTorch Geometric Data objects.
Author: Isam Balghari
"""

from turtle import pos

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def getdata(datasetpath, batch_size=32):
    logprefix = "[DATA] "

    datalist = torch.load(datasetpath)

        # Assuming 'dataset' is your list of Data() objects
    print("Pre-processing dataset to eliminate the float32 precision trap...")

    # 1. Extract all energies using float64 to ensure perfect precision during the sum
    all_energies = torch.tensor([data.y_energy.item() for data in datalist], dtype=torch.float64)
    mean_energy = all_energies.mean()

    print(f"Calculated Mean Energy (Baseline): {mean_energy:.4f} eV")

    # 2. Shift every graph individually and convert back to float32
    # (We subtract the float64 mean, then cast the result back to float32 for fast training)
    for data in datalist:
        shifted_val = data.y_energy.to(torch.float64) - mean_energy
        data.y_energy = shifted_val.to(torch.float32)

    # 3. Verify the shift worked (the new mean should be mathematically zero)
    shifted_mean = torch.tensor([data.y_energy.item() for data in datalist]).mean()
    print(f"Shifted Dataset Mean: {shifted_mean:.6f} eV")

    trsize = int(1.0 * len(datalist))
    vsize = int(0.0 * trsize)
    ttsize = len(datalist) - trsize - vsize

    generator = torch.Generator().manual_seed(42)

    trainlist, valist, testlist = random_split(datalist, [trsize, vsize, ttsize],
                               generator=generator)
    # logger.info(f'{logprefix}Train Set: {len(trainlist)}')
    # logger.info(f'{logprefix}Validation Set: {len(valist)}')
    # logger.info(f'{logprefix}Test Set: {len(testlist)}')

    trainloader = DataLoader(trainlist, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True)
    valloader = DataLoader(valist, batch_size=batch_size, shuffle=False, num_workers=4,
                           pin_memory=True)
    testloader = DataLoader(testlist, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)

    return trainloader, valloader, testloader
if __name__ == "__main__":
    
    trainloader, valloader, testloader = getdata("./DataSets/GNN/RattleGNN-Strain_0.0-Rcut_7.0.pth", 1)

    forces = []
    for k, batch in enumerate(trainloader):
        # print(f"Batch {k+1}:")
        # print('pos: ', batch.pos.shape)
        # print('z: ', batch.z.shape)
        # print('edge_index: ', batch.edge_index.shape)
        # print('y_energy: ', batch.y_energy.shape)
        # print('batch: ', batch.batch.shape)
        print('y_forces: ', torch.norm(batch.y_forces, dim=1).shape)
        forces.append(torch.norm(batch.y_forces, dim=1))

    forces = torch.stack(forces)
    print('All forces shape: ', forces.shape)
    import matplotlib.pyplot as plt
    # plt.hist(forces.cpu().numpy().flatten(), bins=100)
    plt.plot(forces.cpu().numpy().flatten(), marker='o', linestyle='', markersize=2)
    plt.title('Distribution of Force Magnitudes in Training Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Force Magnitude')
    plt.show()
