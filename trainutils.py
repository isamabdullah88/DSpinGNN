import os
import torch
import numpy as np
from model import NequIP

def loadmodel(checkpoint_path, mps):
    # Load model architecture
    model = NequIP(mps=mps)
    device = torch.device('mps' if mps and torch.backends.mps.is_available() else
                        'cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def count_parameters(model):
    # Assuming your model is named 'model'
# and your layers are stored in 'model.layers' or similar
    # first_layer = model.atomembeds # Adjust if your list is named differently
    
    # params = sum(p.numel() for p in first_layer.parameters())
    # print(f"--> Parameters in Layer 0: {params:,}")

    # 1. Total Parameters (Weights + Biases)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. Trainable Parameters (The ones optimizer actually updates)
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print(f"Total Parameters:     {total_params:,}")
    # print(f"Trainable Parameters: {trainable_params:,}")
    
    return total_params


def initialize_shift_scale(model, data_loader):
    print("--> Auto-initializing shift/scale (NumPy Mode)...")
    
    # Get the device where the model lives (cuda, mps, or cpu)
    model_device = next(model.parameters()).device
    num_species = model.output_block.shift.shape[0]
    
    # Define Map: Aspirin (H=1, C=6, O=8) -> (0, 1, 2)
    # This prevents the "All Zeros" bug
    z_map = {1: 0, 6: 1, 7: 2, 8: 2} # Adjust '8' to 2 or 3 based on your model!
    
    A_list = []
    y_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move to CPU for safe processing
            batch = batch.cpu()
            
            # --- 1. Map Atoms ---
            z_values = batch.z
            if z_values.max() < num_species:
                mapped_z = z_values.long() # Already indices
            else:
                try:
                    mapped_z = torch.tensor([z_map.get(int(z), -1) for z in z_values])
                except:
                    print("Error in mapping loop.")
                    return model
                
                if (mapped_z == -1).any():
                    invalid = z_values[mapped_z == -1][0].item()
                    print(f"!!! Error: Atom Type '{invalid}' not found in z_map.")
                    return model

            # --- 2. Build Matrix (PyTorch CPU) ---
            one_hot = torch.nn.functional.one_hot(mapped_z, num_species).float()
            A_batch = torch.zeros(batch.num_graphs, num_species)
            A_batch.index_add_(0, batch.batch, one_hot)
            
            A_list.append(A_batch)
            y_list.append(batch.y)

    # --- 3. Solve with NumPy (The Robust Part) ---
    # Convert large tensors to NumPy arrays
    A_full = torch.cat(A_list, dim=0).numpy()
    y_full = torch.cat(y_list, dim=0).numpy()
    
    print(f"--> Solving for {len(y_full)} structures using NumPy...")
    
    try:
        # NumPy's lstsq is extremely stable
        # rcond=None lets it determine the machine precision cut-off automatically
        solution, residuals, rank, s = np.linalg.lstsq(A_full, y_full, rcond=None)
        
        # Select just the solution (first item) and flatten
        solution = solution.flatten()
        print(f"--> Success! Shifts: {solution}")
        
        # --- 4. Move result back to PyTorch & GPU ---
        solution_tensor = torch.from_numpy(solution).float().to(model_device)
        model.output_block.shift.data = solution_tensor
        
    except Exception as e:
        print(f"--> NumPy Solver Failed: {e}")
        # Fallback: Mean Energy / Mean Atoms
        mean_val = np.mean(y_full) / np.mean(np.sum(A_full, axis=1))
        fallback = torch.ones(num_species, device=model_device) * float(mean_val)
        model.output_block.shift.data = fallback
        print(f"--> Used Fallback Mean: {mean_val}")

    return model



def savecheckpoint(checkpoint_path, epoch, model, optimizer, loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item()
    }, checkpoint_path)
    print(f"✅ Saved checkpoint: {checkpoint_path}")
