import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, energy_weight=1.0, force_weight=1000.0):
        self.we = energy_weight
        self.wf = force_weight

    def __call__(self, energy_pred, forces_pred, batch):
        """Calculates and weights the multi-task MSE loss."""
        loss_e = F.mse_loss(energy_pred, batch.y_energy)
        loss_f = F.mse_loss(forces_pred, batch.y_forces)
        
        loss_tot = (self.we * loss_e) + (self.wf * loss_f)
        
        return loss_tot, loss_e, loss_f