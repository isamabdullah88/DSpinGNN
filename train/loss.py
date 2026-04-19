import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, wenergy=1.0, wforce=100.0, wexchange=1.0, short_weight=2.0, cutoff=4.5):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange
        self.short_weight = short_weight
        self.cutoff = cutoff
        self.mag_alpha = 50.5

    def __call__(self, epred, fpred, xpred, batch):
        losse = torch.tensor(0.0, device=epred.device)  # Placeholder energy loss for now
        lossf = torch.tensor(0.0, device=fpred.device)  # Placeholder force loss

        y_true = batch.y_exchange.view(-1)
        y_pred = xpred.view(-1)

        # 1. Base per-edge exchange loss
        # CRITICAL: reduction='none' returns a tensor of shape [Total_edges]
        # unreduced_lossx = F.huber_loss(y_pred, y_true, delta=0.5, reduction='none')

        # 2. Magnitude Weights (Targeting the extreme values)
        # E.g., if true J = 0.0, weight is 1.0. 
        # If true J = 4.0 and alpha = 0.5, weight is 1.0 + (0.5 * 4) = 3.0.
        # magnitude_weights = 1.0 + (self.mag_alpha * torch.abs(y_true))

        # Optional: Add your distance weights back in here if you want them!
        # distances = batch.cr_edge_dist.view(-1)
        # distance_weights = torch.where(distances < self.cutoff, 
        #                                torch.tensor(self.short_weight, dtype=xpred.dtype, device=xpred.device), 
        #                                torch.tensor(1.0, dtype=xpred.dtype, device=xpred.device))
        # total_weights = magnitude_weights * distance_weights

        # 3. Apply the per-edge weights and then take the mean
        # lossx = torch.mean(magnitude_weights * unreduced_lossx)
        lossx = F.l1_loss(y_pred, y_true, reduction='mean')
            
        loss_tot = (self.we * losse) + (self.wf * lossf) + (self.wx * lossx)
        
        return loss_tot, losse, lossf, lossx