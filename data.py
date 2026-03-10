"""
qm9_parser.py

Parse QM9 .xyz files into PyTorch Geometric Data objects.
Author: Isam Balghari
"""

import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import random_split

# Map from element symbol to atomic number
ATOM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


def getdata(data_dir, mini=True, batch_size=32):

    cache_path = os.path.join(data_dir, "md17_aspirin_processed.pt")
    dataset_path = os.path.join(data_dir, "md17_aspirin.npz")

    # if os.path.exists(cache_path):
    #     print("Loading cached dataset...")
    #     dataset = torch.load(cache_path)
    #     trainloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    #     valloader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
    #     testloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    #     return trainloader, valloader, testloader
    
    dataset = np.load(dataset_path)

    sizeidx = 211762
    
    z = torch.tensor(dataset['z'], dtype=torch.long)
    pos = torch.tensor(dataset['R'], dtype=torch.float)
    F = torch.tensor(dataset['F'], dtype=torch.float)
    E = torch.tensor(dataset['E'], dtype=torch.float)

    dataset_list = []
    for i in range(sizeidx):
    
        datapt = Data(z=z, pos=pos[i], y=E[i], forces=F[i])

        if i% 10000 == 0:
            print(f"Processed {i} data points")

        dataset_list.append(datapt)

    # Save processed dataset to cache
    # torch.save(dataset_list, cache_path)
    # print(f"Saved processed dataset to {cache_path}")

    trsize = 1000
    vsize = 1000
    ttsize = sizeidx - trsize - vsize

    generator = torch.Generator().manual_seed(42)

    train_list, val_list, test_list = random_split(dataset_list, [trsize, vsize, ttsize],
                               generator=generator)
    print('train_list: ', len(train_list))
    print('val_list: ', len(val_list))
    print('test_list: ', len(test_list))
    trainloader = DataLoader(train_list, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True)
    valloader = DataLoader(val_list, batch_size=batch_size, shuffle=False, num_workers=4,
                           pin_memory=True)
    testloader = DataLoader(test_list, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)

    return trainloader, valloader, testloader
if __name__ == "__main__":
    getdata()