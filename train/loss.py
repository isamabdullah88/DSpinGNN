import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, wenergy=1.0, wforce=100.0, wexchange=1.0, short_weight=5.0, cutoff=4.5, mag_alpha=0.5):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange
        self.short_weight = short_weight
        self.cutoff = cutoff
        self.mag_alpha = mag_alpha

    def __call__(self, epred, fpred, xpred, batch):
        losse = F.mse_loss(epred.view(-1), batch.y_energy.view(-1))
        lossf = F.mse_loss(fpred, batch.y_forces)

        # 1. Base per-edge exchange loss
        base_lossx = F.mse_loss(xpred.view(-1), batch.y_exchange.view(-1), reduction='none')

        # 2. Distance Weights
        distances = batch.cr_edge_dist.view(-1)
        
        distance_weights = torch.where(distances < self.cutoff, 
                                       torch.tensor(self.short_weight, dtype=xpred.dtype, device=xpred.device), 
                                       torch.tensor(1.0, dtype=xpred.dtype, device=xpred.device))

        # 3. Magnitude Weights (Targeting the extreme phase-transition values)
        magnitude_weights = 1.0 + (self.mag_alpha * torch.abs(batch.y_exchange.view(-1)))

        # 4. Combine the priors (Max multiplier will be around 5.0 * 6.0 = 30x)
        total_weights = distance_weights * magnitude_weights

        # 5. Apply and average
        lossx = torch.mean(total_weights * base_lossx)
            
        loss_tot = (self.we * losse) + (self.wf * lossf) + (self.wx * lossx)
        
        return loss_tot, losse, lossf, lossx