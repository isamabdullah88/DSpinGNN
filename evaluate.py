import os
import json
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional import pearson_corrcoef  # optional, or implement yourself
E_A_TO_DEBYE = 4.803206083390294  # e * Angstrom -> Debye
eps = 1e-12



def maskedmean(tensor, mask):
    return (tensor * mask).sum() / mask.sum()

def per_atom_mae(qpred, qref, mask):
    return maskedmean(torch.abs(qpred - qref), mask)

def per_atom_rmse(qpred, qref, mask):
    return torch.sqrt(maskedmean(torch.pow(qpred - qref, 2), mask))

def peratom_mse(qpred, qref, mask):
    return maskedmean(torch.pow(qpred - qref, 2), mask)

def pearson_over_atoms(qpred, qref, mask):
    return pearson_corrcoef(qpred[mask], qref[mask])

def dipole_from_charges(qs, pos):
    # qs: tensor of atoms
    # pos: tensor of positions (N,3)
    mueA = torch.zeros(3).to(qs[0].device)
    for i in range(len(qs)):
        mueA += qs[i]*pos[i, :]

    return mueA * E_A_TO_DEBYE # Debye

def dipole_metrics(qpred, pos, murefs):
    meuAs = []
    for i in range(qpred.size(0)):
        meuA = dipole_from_charges(qpred[i, :], pos[i, :, :])
        meuAs.append(meuA)

    meuAs = torch.stack(meuAs)  # (TN,3)

    meuAserr = (meuAs - murefs)**2
    meuA_rmse = torch.sqrt(meuAserr.sum(dim=1).mean())

    meumag = torch.norm(meuAs, dim=1)
    meumagref = torch.norm(murefs, dim=1)
    meumag_rmse = torch.sqrt(((meumag - meumagref)**2 ).mean())

    return meuA_rmse.item(), meumag_rmse.item()


def totalcharge_error(qpred):
    return torch.abs(qpred).mean()


def r2score(qpred, qref, mask):
    return 1 - torch.sum((qpred - qref)**2) / torch.sum((qref - maskedmean(qref, mask))**2)


def persist_metrics(metrics, checkpoint_path, filename="Results/Results.txt"):
    if not os.path.exists("Results"):
        os.makedirs("Results")

    checkpoint = checkpoint_path.split("/")[-1][:-3]

    with open(filename, "a") as f:
        from datetime import datetime
        f.write(f"\nTest Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint: {checkpoint}\n")
        json.dump(metrics, f, indent=4)
        f.write("\n")


def plotcorr(predictions, targets):
    plt.figure(figsize=(6, 6))

    plt.scatter(targets, predictions, s=10, alpha=0.7)
    
    plt.plot([targets.min(), targets.max()],
            [targets.min(), targets.max()],
            'r--', label='Perfect prediction')

    plt.xlabel("True Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title("Predicted vs True Total Energy")
    plt.legend()
    plt.show()