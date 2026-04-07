import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, wenergy=1.0, wforce=100.0, wexchange=1.0):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange

    def exchangeloss(self, xpred, jvals, edgedists):
        # 1. Huber Loss instead of L1 to pierce the 0.20 MAE floor
        # delta=0.5 acts like MSE for errors under 0.5meV, and L1 for massive errors
        base_loss = F.huber_loss(xpred, jvals, delta=0.5, reduction='none')

        # 2. Inverted Distance Weighting 
        # (Mean / Distance) ensures short bonds (>1.0) are penalized more than long bonds (<1.0)
        weights = edgedists.mean() / edgedists
        
        # Apply weights and take the mean across the batch
        wlossx = torch.mean(weights.view(-1, 1) * base_loss)
        return wlossx


    def __call__(self, epred, fpred, xpred, batch):
        """Calculates and weights the multi-task MSE loss."""
        losse = F.mse_loss(epred, batch.y_energy)
        lossf = F.mse_loss(fpred, batch.y_forces)

        credges = batch.cr_edge_index[0]
        graphidx = batch.batch[credges]
        
        exchange_mask = batch.exchange[graphidx].view(-1)
        xpredp = xpred[exchange_mask]
        yexchangep = batch.y_exchange[exchange_mask]
        distsp = batch.cr_edge_dist[exchange_mask]

        # SAFEGUARD: Check if this batch actually contains any J data
        if xpredp.numel() > 0:
            # Call our custom weighted Huber loss
            lossx = self.exchangeloss(xpredp, yexchangep, distsp)
        else:
            # Prevent the NaN explosion by creating a detached zero tensor
            lossx = torch.tensor(0.0, device=xpred.device, requires_grad=True)
        
        loss_tot = (self.we * losse) + (self.wf * lossf) + (self.wx * lossx)
        
        return loss_tot, losse, lossf, lossx