import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, wenergy=1.0, wforce=100.0, wexchange=1.0, short_weight=10.0, cutoff=4.5):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange
        
        # Parameterize your custom weighting logic
        self.short_weight = short_weight
        self.cutoff = cutoff

    def __call__(self, epred, fpred, xpred, batch):
        """Calculates and weights the multi-task MSE/L1 loss."""
        
        losse = F.mse_loss(epred.view(-1), batch.y_energy.view(-1))
        lossf = F.mse_loss(fpred, batch.y_forces)

        # =========================================================
        # NEW: Distance-Weighted Exchange Loss
        # =========================================================
        
        # 1. Dynamically calculate the distance for every edge.
        # (If your dataset already has an explicit distance attribute like 
        # batch.edge_dist or batch.distances, you can use that directly instead!)
        # row, col = batch.edge_index
        # distances = torch.norm(batch.pos[row] - batch.pos[col], p=2, dim=1)
        cr_edge_dist = batch.cr_edge_dist.view(-1)
        
        # 2. Compute the raw L1 loss for every individual edge (no averaging yet)
        base_lossx = F.l1_loss(xpred.view(-1), batch.y_exchange.view(-1), reduction='none')
        
        # 3. Create a weight tensor: 10.0 if distance < 4.5 Å, else 1.0.
        # Using xpred.device ensures the newly created tensors stay on the active GPU.
        weights = torch.where(cr_edge_dist < self.cutoff, 
                              torch.tensor(self.short_weight, dtype=xpred.dtype, device=xpred.device), 
                              torch.tensor(1.0, dtype=xpred.dtype, device=xpred.device))
        
        # 4. Multiply the element-wise losses by your weights and take the mean
        lossx = torch.mean(weights * base_lossx)
        
        # =========================================================
            
        loss_tot = (self.we * losse) + (self.wf * lossf) + (self.wx * lossx)
        
        return loss_tot, losse, lossf, lossx