import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    """
    Generic checkpoint loader. 
    Accepts an already-instantiated model so this utility remains architecture-agnostic.
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    start_epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    
    return model, optimizer, start_epoch, loss


def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def initialize_shift_scale(model, data_loader, z_map, device):
    """
    Solves for the isolated atom energies using a Linear Least Squares solver.
    
    Args:
        model: The initialized neural network.
        data_loader: The training dataloader.
        z_map (dict): Mapping of atomic numbers to indices, e.g., {24: 0, 53: 1} for CrI3.
        device: The PyTorch device.
    """
    logger.info("Auto-initializing Energy Shifts via Least Squares (NumPy Mode)...")
    
    num_species = len(z_map)
    A_list = []
    y_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cpu()
            z_values = batch.z
            
            # Map atomic numbers (Z) to indices (0, 1) safely
            try:
                mapped_z = torch.tensor([z_map.get(int(z), -1) for z in z_values])
            except Exception as e:
                logger.error(f"Failed to map atomic numbers: {e}")
                return model
            
            if (mapped_z == -1).any():
                invalid_z = z_values[mapped_z == -1][0].item()
                logger.error(f"Error: Atomic number '{invalid_z}' not found in provided z_map {z_map}.")
                return model

            # Build the occurrence matrix
            one_hot = torch.nn.functional.one_hot(mapped_z, num_species).float()
            A_batch = torch.zeros(batch.num_graphs, num_species)
            A_batch.index_add_(0, batch.batch, one_hot)
            
            A_list.append(A_batch)
            y_list.append(batch.y_energy)

    # Convert to NumPy for the highly stable lstsq solver
    A_full = torch.cat(A_list, dim=0).numpy()
    y_full = torch.cat(y_list, dim=0).numpy()
    
    logger.info(f"Solving base energies for {len(y_full)} structures...")
    
    try:
        # lstsq finds the best-fit base energy per atom type
        solution, _, _, _ = np.linalg.lstsq(A_full, y_full, rcond=None)
        solution = solution.flatten()
        logger.info(f"Success! Calculated Base Energy Shifts: {solution}")
        
        # Inject the solution into the model's output block
        solution_tensor = torch.from_numpy(solution).float().to(device)
        
        # NOTE: Ensure your model's output block has a 'shift' parameter defined
        model.output_block.shift.data = solution_tensor
        
    except Exception as e:
        logger.warning(f"NumPy Solver Failed: {e}. Falling back to Mean Energy.")
        mean_val = np.mean(y_full) / np.mean(np.sum(A_full, axis=1))
        fallback = torch.ones(num_species, device=device) * float(mean_val)
        model.output_block.shift.data = fallback
        logger.info(f"Used Fallback Mean Shift: {mean_val}")

    return model


def savecheckpoint(checkpoint_path, epoch, model, optimizer, loss):
    """Saves the model, optimizer state, and current loss to disk."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, checkpoint_path)