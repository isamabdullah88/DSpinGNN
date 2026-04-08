import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, wenergy=1.0, wforce=100.0, wexchange=1.0):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange

    def exchangeloss(self, xpred, jvals, edgedists):
        base_loss = F.huber_loss(xpred, jvals, delta=0.5, reduction='none')
        weights = edgedists.mean() / edgedists
        wlossx = torch.mean(weights.view(-1, 1) * base_loss)
        return wlossx

    def __call__(self, epred, fpred, xpred, batch):
        """Calculates and weights the multi-task MSE loss."""
        
        # PRO-TIP: Always use .view(-1) on scalar targets in MSE to prevent 
        # silent broadcasting bugs where [N, 1] and [N] accidentally create an [N, N] matrix!
        losse = F.mse_loss(epred.view(-1), batch.y_energy.view(-1))
        lossf = F.mse_loss(fpred, batch.y_forces)

        credges = batch.cr_edge_index[0]
        graphidx = batch.batch[credges]
        
        exchange_mask = batch.exchange[graphidx].view(-1)
        xpredp = xpred[exchange_mask]
        yexchangep = batch.y_exchange[exchange_mask]

        # SAFEGUARD: Check if this batch actually contains any J data
        if xpredp.numel() > 0:
            # Using standard MSE for this experiment
            lossx = F.mse_loss(xpredp.view(-1), yexchangep.view(-1))
        else:
            # THE FIX: Keep lossx attached to the graph, but force gradients to 0.0
            lossx = xpred.sum() * 0.0
            
        loss_tot = (self.we * losse) + (self.wf * lossf) + (self.wx * lossx)
        
        return loss_tot, losse, lossf, lossx