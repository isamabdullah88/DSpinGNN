"""
loaddata.py

Parse PyTorch Geometric Data objects.
Author: Isam Balghari
"""

from turtle import pos

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from logger import getlogger

logger = getlogger()

def getdata(datasetpath, batch_size=32):
    logprefix = "[DATA] "

    datalist = torch.load(datasetpath)

    trsize = int(0.8 * len(datalist))
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
    
    trainloader, valloader, testloader = getdata("./DataSets/GNN/ExchangeGNN.pth", 1)

    batch = next(iter(trainloader))
    print('pos: ', batch.pos.shape)
    print('z: ', batch.z.shape)
    print('edge_index: ', batch.edge_index.shape)
    print('y_energy: ', batch.y_energy.shape)
    print('batch: ', batch.batch.shape)