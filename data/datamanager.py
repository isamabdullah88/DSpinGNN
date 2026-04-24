"""
loaddata.py

Parse PyTorch Geometric Data objects and visualize physical properties.
Author: Isam Balghari
"""

import logging
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


class DatasetManager:
    """Handles loading, splitting, and batching of PyG datasets."""
    
    def __init__(self, batch_size=32, num_workers=4, random_state=42):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

    def stratified_split(self, dataset, test_size=0.15, extreme_threshold=2.5):
        """
        Splits a graph dataset ensuring that crystals with extreme J values
        are proportionally distributed between train and validation sets.
        """
        graph_labels = []
        
        for data in dataset:
            j_values = data.y_exchange.view(-1)
            # Check if ANY bond in this crystal is outside the physical threshold
            has_extreme = torch.any(torch.abs(j_values) > extreme_threshold).item()
            graph_labels.append(1 if has_extreme else 0)

        train_dataset, val_dataset = train_test_split(
            dataset, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=graph_labels
        )
        
        # Validation checks
        train_extremes = sum(1 for data in train_dataset if torch.any(torch.abs(data.y_exchange) > extreme_threshold))
        val_extremes = sum(1 for data in val_dataset if torch.any(torch.abs(data.y_exchange) > extreme_threshold))
        
        self.logger.info(f"--- Split Complete ---")
        self.logger.info(f"Total Crystals: {len(dataset)}")
        self.logger.info(f"Train Set: {len(train_dataset)} (Extremes: {train_extremes})")
        self.logger.info(f"Val Set:   {len(val_dataset)} (Extremes: {val_extremes})")
        
        return train_dataset, val_dataset

    def dataloaders(self, datasetpath):
        """Loads the raw file and returns PyTorch DataLoaders."""
        datalist = torch.load(datasetpath, weights_only=False)
        
        # Combine if the loaded dict already had splits
        if isinstance(datalist, dict) and 'train' in datalist:
            dataset = datalist['train'] + datalist.get('val', [])
        else:
            dataset = datalist

        trainlist, valist = self.stratified_split(dataset, test_size=0.15)

        # Common kwargs for loaders
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True
        }

        trainloader = DataLoader(trainlist, shuffle=True, **loader_kwargs)
        valloader = DataLoader(valist, shuffle=False, **loader_kwargs)
        
        return trainloader, valloader



# ==========================================
# 3. Execution Script
# ==========================================
if __name__ == "__main__":
    from .inspector import GraphVisualizer
    # Setup basic logging to console if running standalone
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    
    DATA_PATH = "./DataSets/GNN/Exchange-Full-Mixed-Extreme-Stripped_-1.5_3-Pruned_-3.0_3.0-Rcut_4.5.pth"
    
    # 1. Initialize Managers
    data_manager = DatasetManager(batch_size=32)
    visualizer = GraphVisualizer()
    
    # 2. Get Data
    trainloader, valloader = data_manager.dataloaders(DATA_PATH)
    
    # 3. Run Visualizations
    visualizer.plot_j_vs_distance(trainloader, save_path="train_j_vs_dist.png")
    visualizer.plot_j_vs_distance(valloader, save_path="val_j_vs_dist.png")
    
    visualizer.plot_cosines(trainloader)
    visualizer.plot_distances(trainloader)
    
    # Show all active matplotlib windows
    plt.show()