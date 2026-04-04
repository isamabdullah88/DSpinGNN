import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, wenergy=1.0, wforce=100.0, wexchange=1.0):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange

    def exchangeloss(self, xpred, batch):
        edgedists = batch.cr_edge_dist
        jvals = batch.y_exchange

        weights = edgedists / edgedists.mean()
        
        wlossx = torch.mean(weights.view(-1, 1) * F.l1_loss(xpred, jvals, reduction='none'))

        return wlossx


    def __call__(self, epred, fpred, xpred, batch):
        """Calculates and weights the multi-task MSE loss."""
        # loss_e = F.mse_loss(epred, batch.y_energy)
        # loss_f = F.mse_loss(fpred, batch.y_forces)
        # loss_x = F.mse_loss(xpred, batch.y_exchange)
        loss_e = torch.tensor(0.0, device=epred.device)  # Placeholder for energy loss
        loss_f = torch.tensor(0.0, device=fpred.device)  # Placeholder for force loss
        # loss_x = self.exchangeloss(xpred, batch)
        lossx = F.l1_loss(xpred, batch.y_exchange)
        
        # loss_tot = (self.we * loss_e) + (self.wf * loss_f) + (self.wx * loss_x)
        loss_tot = lossx * self.wx
        
        return loss_tot, loss_e, loss_f, lossx