import logging
import torch

from .sample import SampleProcessor
from .directory import DirectoryExplorer


class DatasetBuilder:
    """Orchestrates the discovery and processing of the entire dataset."""
    def __init__(self, rcut):
        self.logger = logging.getLogger(__name__)
        self.processor = SampleProcessor(rcut=rcut)

    def generate(self, dataset_dir, phase='FM'):
        self.logger.info("-" * 50)
        self.logger.info(f"Scanning directory: {dataset_dir} ...")
        
        sample_dirs = DirectoryExplorer.find_samples(dataset_dir, phase)
        self.logger.info(f"Found {len(sample_dirs)} potential rattled strains.")
        self.logger.info("-" * 50)

        dataset = []
        for sdir in sample_dirs:
            data = self.processor.process(sdir)
            if data is not None:
                dataset.append(data)

        self.logger.info("-" * 50)
        self.logger.info(f"Dataset generation complete!")
        self.logger.info(f"Successfully processed: {len(dataset)} / {len(sample_dirs)} samples.")
        
        return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    RCUT = 4.5    
    datasetdir = "./DataSets/ExchangeDataset-Full-Mixed/"
    datasetpath = f"./DataSets/GNN/Exchange-Full-Mixed-Extreme-Stripped_-1.5_3-Pruned_-3.0_3.0-Rcut_{RCUT:.1f}.pth"

    # 1. Initialize Builder
    builder = DatasetBuilder(rcut=RCUT)
    
    # 2. Generate Data
    dataset = builder.generate(datasetdir)

    # 3. Save Data 
    # Note: Because you built the excellent `create_stratified_split` in loaddata.py, 
    # we package everything into 'train' here, and let loaddata handle the val splitting!
    logger.info(f"Saving {len(dataset)} graphs to {datasetpath}...")
    torch.save({'train': dataset, 'val': []}, datasetpath)
    logger.info("Save successful.")