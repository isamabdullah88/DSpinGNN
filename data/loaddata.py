"""
loaddata.py

Parse PyTorch Geometric Data objects.
Author: Isam Balghari
"""

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import logging

def getdata(datasetpath, batch_size=32):
    logprefix = "[DATA] "
    logger = logging.getLogger(__name__)

    datalist = torch.load(datasetpath, weights_only=False)

    """
        # Assuming 'dataset' is your list of Data() objects
    logger.info(f"{logprefix}Pre-processing dataset to eliminate the float32 precision trap...")

    # 1. Extract all energies using float64 to ensure perfect precision during the sum
    all_energies = torch.tensor([data.y_energy.item() for data in datalist], dtype=torch.float64)
    mean_energy = all_energies.mean()

    jvalslist = []
    for data in datalist:
        if data.exchange:
            jvalslist += data.y_exchange.view(-1).tolist()
    # print('javalslist: ', jvalslist)
    jvals = torch.tensor(jvalslist, dtype=torch.float64)
    mean_jval = jvals.mean()
    logger.info(f"{logprefix}Calculated Mean Exchange J (Baseline): {mean_jval:.4f} meV")

    logger.info(f"{logprefix}Calculated Mean Energy (Baseline): {mean_energy:.4f} eV")

    # 2. Shift every graph individually and convert back to float32
    # (We subtract the float64 mean, then cast the result back to float32 for fast training)
    for data in datalist:
        shifted_val = data.y_energy.to(torch.float64) - mean_energy
        data.y_energy = shifted_val.to(torch.float32)

        if data.exchange:
            data.y_exchange = data.y_exchange.to(torch.float64) - mean_jval
            data.y_exchange = data.y_exchange.to(torch.float32)

    # 3. Verify the shift worked (the new mean should be mathematically zero)
    shifted_mean = torch.tensor([data.y_energy.item() for data in datalist]).mean()
    logger.info(f"{logprefix}Shifted Dataset Mean: {shifted_mean:.6f} eV")

    jvalslist = []
    for data in datalist:
        if data.exchange:
            jvalslist += data.y_exchange.view(-1).tolist()
    
    jvals = torch.tensor(jvalslist, dtype=torch.float64)
    meanj = jvals.mean()
    logger.info(f"{logprefix}Shifted Dataset Mean Exchange J: {meanj:.6f} meV")
    """
    
    trsize = int(0.9 * len(datalist))
    vsize = int(0.1 * trsize)
    ttsize = len(datalist) - trsize - vsize

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
    
    trainloader, valloader, testloader = getdata("./DataSets/GNN/MixedDataset_All-Rcut_7.0.pth", 32)

    forces = []
    for k, batch in enumerate(valloader):
        print('batch y_exchange: ', batch.y_exchange.shape)
        print('batch exchange: ', batch.exchange.shape)
        # forces.append(torch.norm(batch.y_forces, dim=1))

    # forces = torch.stack(forces)
    # print('All forces shape: ', forces.shape)
    # import matplotlib.pyplot as plt
    # # plt.hist(forces.cpu().numpy().flatten(), bins=100)
    # plt.plot(forces.cpu().numpy().flatten(), marker='o', linestyle='', markersize=2)
    # plt.title('Distribution of Force Magnitudes in Training Set')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Force Magnitude')
    # plt.show()
