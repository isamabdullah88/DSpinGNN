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
    """
    logger.info("Auto-initializing Energy Shifts via Least Squares (NumPy Mode)...")
    
    num_species = len(z_map)
    A_list = []
    y_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cpu()
            z_values = batch.z
            
            try:
                mapped_z = torch.tensor([z_map.get(int(z), -1) for z in z_values])
            except Exception as e:
                logger.error(f"Failed to map atomic numbers: {e}")
                return model
            
            if (mapped_z == -1).any():
                invalid_z = z_values[mapped_z == -1][0].item()
                logger.error(f"Error: Atomic number '{invalid_z}' not found in provided z_map {z_map}.")
                return model

            one_hot = torch.nn.functional.one_hot(mapped_z, num_species).float()
            A_batch = torch.zeros(batch.num_graphs, num_species)
            A_batch.index_add_(0, batch.batch, one_hot)
            
            A_list.append(A_batch)
            y_list.append(batch.y_energy)

    A_full = torch.cat(A_list, dim=0).numpy()
    y_full = torch.cat(y_list, dim=0).numpy()
    
    logger.info(f"Solving base energies for {len(y_full)} structures...")
    
    try:
        solution, _, _, _ = np.linalg.lstsq(A_full, y_full, rcond=None)
        solution = solution.flatten()
        logger.info(f"Success! Calculated Base Energy Shifts: {solution}")
        
        # =========================================================
        # THE FIX: INJECT RATHER THAN OVERWRITE
        # =========================================================
        # Iterate through your map (e.g. 24: 0, 53: 1)
        for real_z, solver_idx in z_map.items():
            # Place the Cr energy (idx 0) into slot 24, and I (idx 1) into slot 53
            model.output_block.shift.data[real_z] = float(solution[solver_idx])
            
    except Exception as e:
        logger.warning(f"NumPy Solver Failed: {e}. Falling back to Mean Energy.")
        mean_val = np.mean(y_full) / np.mean(np.sum(A_full, axis=1))
        
        # Safely inject the fallback mean into the correct slots as well
        for real_z in z_map.keys():
            model.output_block.shift.data[real_z] = float(mean_val)
            
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