import os
import time
import torch
import numpy as np
import wandb
from model import calcforce
from .trainutils import savecheckpoint

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, logger, config, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.config = config
        self.scheduler = scheduler
        
        self.train_size = int(len(self.train_loader.dataset) / self.config.batch_size)
        self.checkpoints_dir = "./checkpoints"
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()

        totlosse = 0.0
        totlossf = 0.0
        totlossx = 0.0
        totloss = 0.0
        totgraphs = 0
        
        for k, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            batch.pos.requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            energy, exchange = self.model(batch)
            forces = calcforce(energy, batch.pos)
            
            loss, losse, lossf, lossx = self.criterion(energy, forces, exchange, batch)
            loss.backward()
            
            self.optimizer.step()
            
            totloss += loss.detach() * batch.num_graphs
            totgraphs += batch.num_graphs
            totlosse += losse.detach() * batch.num_graphs
            totlossf += lossf.detach() * batch.num_graphs
            totlossx += lossx.detach() * batch.num_graphs


        epochloss = (totloss / totgraphs).item()
        epochlosse = (totlosse / totgraphs).item()
        epochlossf = (totlossf / totgraphs).item()
        epochlossx = (totlossx / totgraphs).item()

        wandb.log({
            "Train/train_loss": epochloss,
            "Train/train_loss_energy": epochlosse,
            "Train/train_loss_forces": epochlossf,
            "Train/train_loss_exchange": epochlossx,
            "iter": self.train_size * epoch + k,
            "Train/learning_rate": self.optimizer.param_groups[0]['lr']
        })

    def validate_epoch(self, epoch):
        self.model.eval() 
        self.logger.info("Starting evaluation...")
        
        totloss, totlosse, totlossf, totlossx = 0.0, 0.0, 0.0, 0.0
        totmaee, totmaef, totmaex, totmaex1, totmaex2 = 0.0, 0.0, 0.0, 0.0, 0.0
        totatoms, totgraphs, totedges = 0, 0, 0
        
        # TODO: Clean all this code up please!
        with torch.enable_grad(): 
            for batch in self.val_loader:
                batch = batch.to(self.device)
                batch.pos.requires_grad_(True)
                
                energy, exchange = self.model(batch)
                forces = calcforce(energy, batch.pos)

                # self.logger.info(f"Exchange predictions (SUM): {np.sum(np.abs(exchange.detach().cpu().numpy())):.4f}")
                # self.logger.info(f"Exchange targets (SUM): {np.sum(np.abs(batch.y_exchange.detach().cpu().numpy())):.4f}")

                # shortdist = batch.cr_edge_dist[batch.cr_edge_dist < 4.5][0]
                # shortexchange = exchange[batch.cr_edge_dist < 4.5][0]
                # shorttarget = batch.y_exchange[batch.cr_edge_dist < 4.5][0]
                # self.logger.info(f"Cr-Cr Short edge distances: {shortdist.detach().cpu().numpy()}")
                # self.logger.info(f"Exchange predictions: {shortexchange.detach().cpu().numpy()}")
                # self.logger.info(f"Exchange targets: {shorttarget.detach().cpu().numpy()}")

                # longdist = batch.cr_edge_dist[batch.cr_edge_dist >= 4.5][0]
                # longexchange = exchange[batch.cr_edge_dist >= 4.5][0]
                # longtarget = batch.y_exchange[batch.cr_edge_dist >= 4.5][0]
                # self.logger.info(f"Cr-Cr Long edge distances: {longdist.detach().cpu().numpy()}")
                # self.logger.info(f"Exchange predictions: {longexchange.detach().cpu().numpy()}")
                # self.logger.info(f"Exchange targets: {longtarget.detach().cpu().numpy()}")

                loss, losse, lossf, lossx = self.criterion(energy, forces, exchange, batch)
                
                erre = torch.abs(energy.detach().view(-1) - batch.y_energy.view(-1)).sum()
                errf = torch.abs(forces.detach() - batch.y_forces).sum()

                j1mask = batch.cr_edge_dist < 4.5
                j2mask = batch.cr_edge_dist >= 4.5

                errx1 = torch.abs(exchange[j1mask].detach().view(-1) - batch.y_exchange[j1mask].view(-1)).sum()
                errx2 = torch.abs(exchange[j2mask].detach().view(-1) - batch.y_exchange[j2mask].view(-1)).sum()
                errx = torch.abs(exchange.detach().view(-1) - batch.y_exchange.view(-1)).sum()
                
                numgraphs = batch.num_graphs
                numatoms = batch.pos.shape[0]
                numedges1 = j1mask.sum().item()
                numedges2 = j2mask.sum().item()
                numedges = batch.cr_edge_index.shape[1]
                
                totloss += loss.item() * numgraphs
                totlosse += losse.item() * numgraphs
                totlossf += lossf.item() * numgraphs
                totlossx += lossx.item() * numgraphs
                totmaee += erre
                totmaef += errf
                totmaex += errx

                totmaex1 += errx1
                totmaex2 += errx2

                totgraphs += numgraphs
                totatoms += numatoms
                totedges += numedges
                totedges1 += numedges1
                totedges2 += numedges2


        epochloss = (totloss / totgraphs).item()
        epochlosse = (totlosse / totgraphs).item()
        epochlossf = (totlossf / totgraphs).item()
        epochlossx = (totlossx / totgraphs).item()

        maee = (totmaee / totatoms).item()
        maef = (totmaef / (totatoms * 3)).item()
        maex = (totmaex / totedges).item()
        maex1 = (totmaex1 / numedges1).item() if numedges1 > 0 else 0
        maex2 = (totmaex2 / numedges2).item() if numedges2 > 0 else 0

        self.logger.info("[TRAIN] RESULTS (Validation Set)")
        self.logger.info(f"[TRAIN] Val Loss (MSE):     {epochloss:.5f}")
        self.logger.info(f"[TRAIN] Val Energy (MAE):   {maee:.5f} eV/atom")
        self.logger.info(f"[TRAIN] Val Forces (MAE):   {maef:.5f} eV/A")
        self.logger.info(f"[TRAIN] Val Exchange (MAE): {maex:.5f}")
        self.logger.info(f"[TRAIN] Val Exchange (MAE) - Short Range: {maex1:.5f}")
        self.logger.info(f"[TRAIN] Val Exchange (MAE) - Long Range: {maex2:.5f}")

        wandb.log({
            "Test/MAE-Exchange": maex,
            "Test/MAE-Exchange-Short": maex1,
            "Test/MAE-Exchange-Long": maex2,
            "Test/MAE-Energy": maee,
            "Test/MAE-Force": maef,
            "Test/Validation-Loss": epochloss,
            "Test/Validation-Loss-Energy": epochlosse,
            "Test/Validation-Loss-Forces": epochlossf,
            "Test/Validation-Loss-Exchange": epochlossx,
            "epoch": epoch
        })

    def save_models(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"Epoch-{epoch:04d}.pt")
        latest_ckpath = os.path.join(self.checkpoints_dir, "latest-model.pt")
        
        savecheckpoint(checkpoint_path, epoch, self.model, self.optimizer, loss)
        savecheckpoint(latest_ckpath, epoch, self.model, self.optimizer, loss)
        
        wandb.save(checkpoint_path)
        wandb.save(latest_ckpath)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def fit(self):
        self.logger.info("Starting training loop...")
        for epoch in range(self.config.epochs + 1):
            stime = time.time()
            
            epochloss = self.train_epoch(epoch)
            
            if (epoch + 1) % 1 == 0:
                self.validate_epoch(epoch)
                
                # if self.scheduler is not None:
                    # Step based on Force MAE since forces are the hardest to fit
                    # self.scheduler.step(val_maex)
                
            line = f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {epochloss:.4f}, Time: {(time.time()-stime): .01f}\n" 
            self.logger.info(line)
            
            if epoch % 10 == 0:
                self.save_models(epoch, epochloss)
                
        wandb.finish()