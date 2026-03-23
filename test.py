import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model import DSpinGNN, force
import time

from data import getdata

import torch
import logging

# logger = logging.getLogger()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def evaluate(model, dataloader, device):
    """
    Calculates MAE for Energy and Forces over a dataset.
    Standard Benchmark for Aspirin:
    - Energy MAE: ~0.002 eV
    - Force MAE:  ~0.015 eV/A
    """
    model.eval() # Switch to evaluation mode (disable dropout, etc.)
    
    mae_energy_sum = 0.0
    mae_force_sum = 0.0
    
    total_molecules = 0
    total_atoms = 0
    
    
    # We do NOT need gradients for metrics, so we turn them off to save memory
    # BUT: We still need 'create_graph=True' inside the force calculation 
    # if we were training. Since we are just testing, we can use inference mode
    # carefully.
    
    # actually, to compute forces, we MUST enable grad on input positions,
    # even in validation.
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # 1. Prepare Data
        pos = batch.pos.requires_grad_(True)
        
        # 2. Forward Pass
        # (Re-implementing the core parts of get_energy_and_forces here 
        # to ensure we don't accidentally train)
        
        # A. Run Model
        energy = model(batch)

        forces = force(energy, pos)
        
        # B. Calculate Forces (Derivative)
        # ones = torch.ones_like(energy)
        # grads = torch.autograd.grad(
        #     outputs=energy,
        #     inputs=pos,
        #     grad_outputs=ones,
        #     create_graph=False, # False because we don't need second derivatives now
        #     retain_graph=False
        # )[0]
        # forces = -1 * grads
        
        # 3. Calculate Errors (MAE)
        # Detach from graph to prevent memory leaks
        e_pred = energy.detach()
        f_pred = forces.detach()
        
        # Energy Error (Per Molecule)
        # Shape: (Batch_Size, 1)
        e_err = torch.abs(e_pred.view(-1) - batch.y_energy.view(-1))
        mae_energy_sum += e_err.sum().item()
        
        # Force Error (Per Atom Component)
        # Shape: (Total_Atoms, 3)
        # We average over all 3 dimensions (x,y,z) and all atoms
        f_err = torch.abs(f_pred - batch.y_forces)
        mae_force_sum += f_err.sum().item()
        
        # 4. Update Counts 
        total_molecules += batch.num_graphs
        total_atoms += (batch.pos.shape[0] * 3) # x, y, z components

    # 5. Final Averages
    mae_energy = mae_energy_sum / total_atoms
    mae_force = mae_force_sum / total_atoms
    
    return mae_energy, mae_force

"""
# Example: path to saved model
checkpoint_path = "checkpoints/model_E181.pt"

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(f"Using device: {device}")

dfeatdim = 25
numatoms = 30
atomembdim = 20

start = time.time()
# Load model architecture
model = PartialCharge(numatoms, atomembdim, dfeatdim)
checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

model = model.to(device)
end1 = time.time()

print(f"Model Loading completed in {end1 - start:.2f} seconds.")



predictions, targets = [], []
posrefs, murefs = [], []
zbatchs = []

_, testloader = getdata(mini=False, batch_size=64)

with torch.no_grad():
    for batch in testloader:
        batch = batch.to(device)
        
        s = time.time()
        zbatch, dist, target, posref, meuref = prepbatch(batch, numatoms)
        # print(f"Batch preparation time: {time.time() - s:.2f} seconds.")
        
        pred = model(zbatch, dist)
        
        predictions.append(pred)
        targets.append(target)
        zbatchs.append(zbatch)
        posrefs.append(posref)
        murefs.append(meuref)

end2 = time.time()
print(f"Inference completed in {end2 - start:.2f} seconds.")

predictions = torch.concatenate(predictions).cpu()
targets = torch.concatenate(targets).cpu()
posrefs = torch.concatenate(posrefs).cpu()
murefs = torch.concatenate(murefs).cpu()
zs = torch.concatenate(zbatchs).cpu()
mask = (zs > 0)

# Evaluations
from evaluate import *


mae = per_atom_mae(predictions, targets, mask).item()
rmse = per_atom_rmse(predictions, targets, mask).item()
pearson = pearson_over_atoms(predictions, targets, mask).item()
dip_metrics = dipole_metrics(predictions, posrefs, murefs)
r2 = r2score(predictions, targets, mask).item()
total_charge_error = ( (predictions * mask).sum(dim=1).abs().mean().item() )

print(f"Evaluation completed in {time.time() - end2:.2f} seconds.")

metrics = {
    "per_atom_MAE": mae,
    "per_atom_RMSE": rmse,
    "pearson": pearson,
    "total_charge_error_mean_abs": total_charge_error,
    "R2": r2,
    "Dipole Vec Error": dip_metrics[0],
    "Dipole Mag Error": dip_metrics[1]
}

persist_metrics(metrics, checkpoint_path)

print("\nEvaluation Metrics:")
print(f"MAE: {mae:.4f} eV")
print(f"RMSE: {rmse:.4f} eV")
print(f"R²: {r2:.4f}")
print(f"Pearson: {pearson:.4f}")
print(f"Total Charge Error (Mean Abs): {total_charge_error:.4f} e")
print(f"Dipole Vector RMSE: {dip_metrics[0]:.4f} Debye")
print(f"Dipole Magnitude RMSE: {dip_metrics[1]:.4f} Debye")

plotcorr(predictions, targets)
"""