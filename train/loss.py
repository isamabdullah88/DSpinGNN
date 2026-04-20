import torch
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, modelname,wenergy=1.0, wforce=100.0, wexchange=1.0, short_weight=2.0, cutoff=4.5):
        self.we = wenergy
        self.wf = wforce
        self.wx = wexchange
        self.short_weight = short_weight
        self.cutoff = cutoff
        self.mag_alpha = 50.5

        self.modelname = modelname

    def __call__(self, epred, fpred, xpred, batch):

        if self.modelname == "StructureModel":
            losse = F.mse_loss(epred, batch.y_energy)
            lossf = F.mse_loss(fpred, batch.y_forces)
            lossx = torch.tensor(0.0, device=epred.device)  # No exchange loss for StructureModel

        elif self.modelname == "ExchangeModel":
            losse = torch.tensor(0.0, device=epred.device)  # Placeholder energy loss for now
            lossf = torch.tensor(0.0, device=fpred.device)  # Placeholder force loss

            y_true = batch.y_exchange.view(-1)
            y_pred = xpred.view(-1)

            lossx = F.l1_loss(y_pred, y_true, reduction='mean')
            
        loss_tot = (self.we * losse) + (self.wf * lossf) + (self.wx * lossx)
        
        return loss_tot, losse, lossf, lossx