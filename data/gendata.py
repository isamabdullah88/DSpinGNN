import os
import torch
import numpy as np
from torch_geometric.data import Data
from ase.db import connect

from .hubbard import EspressoHubbard
from .CrI3 import CrI3
from .exchange import ExchangeGraph
from .crystaltensor import CrystalGraphTensor

from logging import getLogger

class DataGenerator:
    def __init__(self, rcut, stntypes, strains, phase='FM'):
        self.rcut = rcut

        self.dataset = []

        self.phase = phase
        self.stntypes = stntypes
        self.strains = strains

        self.hubbard = EspressoHubbard()
        self.crI3 = CrI3()

        self.egraph = ExchangeGraph()
        self.cgraph = CrystalGraphTensor()

        self.lprefix = "[DataGenerator] "
        # self.logger = getLogger(__name__)
    


    def parse(self, wkdir, stndir, fthresh=5.0):
        pwipath = os.path.join(wkdir, stndir, "espresso.pwi")
        pwopath = os.path.join(wkdir, stndir, "espresso.pwo")

        if not os.path.exists(pwipath):
            logger.warning(f"{self.lprefix}Input file not found for {stndir} at {pwipath}. Skipping...")
            return None, None

        if not os.path.exists(pwopath):
            logger.warning(f"{self.lprefix}Output file not found for {stndir} at {pwopath}. Skipping...")
            return None, None

        # atoms = self.crI3.strain_atoms(stntype=stntype, stnvalue=stnvalue)
        # atoms = self.crI3.batoms.copy()
        atomsout = self.hubbard.parse(pwipath, pwopath)

        z = torch.tensor(atomsout.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atomsout.get_positions(), dtype=torch.float32)
        energy = atomsout.get_potential_energy()
        forces = atomsout.get_forces()
        cell = torch.tensor(np.array(atomsout.get_cell()), dtype=torch.float32)

        if forces.any() > fthresh:
            logger.warning(f"{self.lprefix}Warning: Large forces detected for {stndir}. Max force magnitude: {np.max(np.linalg.norm(forces, axis=1)):.4f} eV/Å. Skipping this sample.")
            return None, None
        
        edgeidxs, edgeshifts = self.cgraph.tensorgraph(self.rcut, atomsout)

        exchange = True
        #  Parse TB2J outputs for this strain
        tb2jpath = os.path.join(wkdir, stndir, "tmp/TB2J_results/Multibinit", "exchange.xml")
        if not os.path.exists(tb2jpath):
            logger.warning(f"{self.lprefix}TB2J output file not found for {stndir} at {tb2jpath}. Skipping TB2J parsing...")
            cr_edges, exchangejs = torch.zeros((2, 1), dtype=torch.long), torch.zeros((1, 1), dtype=torch.float32) 
            cr_shifts, cr_edgedists = torch.zeros((1, 3), dtype=torch.int64), torch.zeros((1,), dtype=torch.float32)
            exchange = False
        else:
            cr_edges, exchangejs, cr_shifts, cr_edgedists = self.egraph.graph(tb2jpath, self.rcut, atomsout)
            # self.logger.info(f"Cr Edges Shape: {cr_edges.shape}, Exchange J Shape: {exchangejs.shape}, Edge Shifts Shape: {cr_shifts.shape}, Edge Distances Shape: {cr_edgedists.shape}")

        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            y_energy=torch.tensor([energy], dtype=torch.float32),
            y_forces=torch.tensor(forces, dtype=torch.float32),
            edge_index=edgeidxs,
            edge_shift=edgeshifts,
            cr_edge_index=cr_edges,
            cr_edge_shift=cr_shifts,
            cr_edge_dist=cr_edgedists,
            y_exchange=exchangejs,
            exchange = torch.tensor([exchange], dtype=torch.bool).view(-1, 1)
        )

        return data, atomsout
    
    # TODO: Refactor this method to be cleaner and more modular. It's doing a lot of things at once right now.
    def generate(self, datasetdir, rattleidxs):
        """
        Main method to generate the dataset. It iterates over all strains, parses the DFT and TB2J outputs, and saves the PyG Data objects.
        """
        # db = connect('./DataSets/GNN/RattleGNN-Strain_0.0-Rcut_7.0.db')
        numsamples = 0
        for stnsdir, stnvalues, rattleidxlist in zip(self.stntypes, self.strains, rattleidxs):
            logger.info(f"-" * 50)
            logger.info(f"Processing work directory: {stnsdir}...")
            logger.info(f"-" * 50)
            for strain, rattleidx in zip(stnvalues, rattleidxlist):
                stntype = [f for f in ["Biaxial", "Uniaxial_X", "Shear_XY"] if f in stnsdir][0]
                stndir = f"{stnsdir}/FM/Strain_{stntype}_{strain:.4f}_{rattleidx}"

                logger.info(f"Processing strain (rattled): {stndir}...")

                data, atoms = self.parse(datasetdir, stndir)
                if data is not None:
                    self.dataset.append(data)
                    # db.write(atoms)
                    numsamples += 1
                else:
                    logger.warning(f"Data parsing failed for strain {strain:.4f} (Rattle idx: {rattleidx}). Skipping this sample.")

        logger.info(f"Dataset generation complete. Total samples: {numsamples}")

        return self.dataset



if __name__ == "__main__":
    from pathlib import Path
    from logger import getlogger
    logger = getlogger()

    RCUT = 7.0    
    datasetdir = "./DataSets/MixedDataset/"
    datasetpath = f"./DataSets/GNN/Mixed-Dataset-Rattled-Rcut_{RCUT:.1f}.pth"

    dirpath = Path(datasetdir)
    dirlist = [d.name for d in dirpath.iterdir() if not d.name.startswith(".") and d.is_dir()]
    logger.info(f"Found directories: {dirlist}")

    stntypes = []
    strains = []
    rattleidxs = []
    numsamples = 0
    for dirname in dirlist:
        stntypes.append(dirname)

        stnvalues = []
        rattleidx = []

        stndir = Path(os.path.join(datasetdir, dirname, "FM"))
        stnlist = [d.name for d in stndir.iterdir() if not d.name.startswith(".") and d.is_dir()]
        for stn in stnlist:
            stnvalues.append(float(stn.split('_')[-2]))
            rattleidx.append(int(stn.split('_')[-1]))
            numsamples += 1

        strains.append(stnvalues)
        rattleidxs.append(rattleidx)

    logger.info(f"Number of strain types: {len(stntypes)}")
    logger.info(f"Total samples: {numsamples}")
    logger.info(f"Strains: {strains[0]}")
    logger.info(f"Rattle indices: {rattleidxs[0]}")

    datagen = DataGenerator(RCUT, stntypes=stntypes, strains=strains, phase='FM')

    dataset = datagen.generate(datasetdir=datasetdir, rattleidxs=rattleidxs)

    logger.info(f"Saving to {datasetpath}...")
    torch.save(dataset, datasetpath)

