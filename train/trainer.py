import os
import time
import torch
import numpy as np
import wandb
from model import calcforce
from .trainutils import savecheckpoint

class Trainer:
    # ADDED: scheduler=None to the initialization
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, logger, config, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.config = config
        self.scheduler = scheduler  # <--- Store the scheduler
        
        # Safety check for train_size to prevent division by zero in wandb logging
        self.train_size = max(1, int(len(self.train_loader.dataset) / self.config.batch_size))
        self.checkpoints_dir = "./checkpoints"
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        for k, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            batch.pos.requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            energy, exchange = self.model(batch)
            forces = calcforce(energy, batch.pos)
            
            loss, losse, lossf, lossx = self.criterion(energy, forces, exchange, batch)
            loss.backward()
            
            self.optimizer.step()
            
            wandb.log({
                "Train/train_loss": loss.item(),
                "Train/train_loss_energy": losse.item(),
                "Train/train_loss_forces": lossf.item(),
                "Train/train_loss_exchange": lossx.item(),
                "iter": self.train_size * epoch + k,
                "Train/learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            epoch_loss = loss.item()
            
        return epoch_loss

    def validate_epoch(self, epoch):
        self.model.eval() 
        self.logger.info("Starting evaluation...")
        
        total_valloss, total_vallosse, total_vallossf, total_vallossx = 0.0, 0.0, 0.0, 0.0
        total_mae_energy, total_mae_force, total_mae_exchange, maex1, maex2 = 0.0, 0.0, 0.0, 0.0, 0.0
        
        num_val_graphs = 0
        num_val_atoms = 0
        numedges = 0
        numedges1 = 0
        numedges2 = 0
        
        # TODO: Clean all this code up please!
        with torch.enable_grad(): 
            for batch in self.val_loader:
                batch = batch.to(self.device)
                batch.pos.requires_grad_(True)
                
                energy, exchange = self.model(batch)
                forces = calcforce(energy, batch.pos)

                self.logger.info(f"Exchange predictions (SUM): {np.sum(np.abs(exchange.detach().cpu().numpy())):.4f}")
                self.logger.info(f"Exchange targets (SUM): {np.sum(np.abs(batch.y_exchange.detach().cpu().numpy())):.4f}")

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

                valloss, vallosse, vallossf, vallossx = self.criterion(energy, forces, exchange, batch)
                
                e_err = torch.abs(energy.detach().view(-1) - batch.y_energy.view(-1)).sum().item()
                f_err = torch.abs(forces.detach() - batch.y_forces).sum().item()

                j1mask = batch.cr_edge_dist < 4.5
                j2mask = batch.cr_edge_dist >= 4.5

                xerr1 = torch.abs(exchange[j1mask].detach().view(-1) - batch.y_exchange[j1mask].view(-1)).sum().item()
                xerr2 = torch.abs(exchange[j2mask].detach().view(-1) - batch.y_exchange[j2mask].view(-1)).sum().item()
                xerr = torch.abs(exchange.detach().view(-1) - batch.y_exchange.view(-1)).sum().item()
                
                graphs_in_batch = batch.num_graphs
                atoms_in_batch = batch.pos.shape[0]
                edges1 = j1mask.sum().item()
                edges2 = j2mask.sum().item()
                edges = batch.cr_edge_index.shape[1]
                
                total_valloss += valloss.item() * graphs_in_batch
                total_vallosse += vallosse.item() * graphs_in_batch
                total_vallossf += vallossf.item() * graphs_in_batch
                total_vallossx += vallossx.item() * graphs_in_batch
                total_mae_energy += e_err
                total_mae_force += f_err
                total_mae_exchange += xerr

                maex1 += xerr1
                maex2 += xerr2

                num_val_graphs += graphs_in_batch
                num_val_atoms += atoms_in_batch
                numedges += edges
                numedges1 += edges1
                numedges2 += edges2


        avg_valloss = total_valloss / num_val_graphs
        avg_vallosse = total_vallosse / num_val_graphs
        avg_vallossf = total_vallossf / num_val_graphs
        avg_vallossx = total_vallossx / num_val_graphs

        mae_energy_per_atom = total_mae_energy / num_val_atoms
        mae_force_per_comp = total_mae_force / (num_val_atoms * 3)
        maex_peredge = total_mae_exchange / numedges
        maex1_peredge = maex1 / numedges1 if numedges1 > 0 else 0
        maex2_peredge = maex2 / numedges2 if numedges2 > 0 else 0

        self.logger.info("[TRAIN] RESULTS (Validation Set)")
        self.logger.info(f"[TRAIN] Val Loss (MSE):     {avg_valloss:.5f}")
        self.logger.info(f"[TRAIN] Val Energy (MAE):   {mae_energy_per_atom:.5f} eV/atom")
        self.logger.info(f"[TRAIN] Val Forces (MAE):   {mae_force_per_comp:.5f} eV/A")
        self.logger.info(f"[TRAIN] Val Exchange (MAE): {maex_peredge:.5f}")
        self.logger.info(f"[TRAIN] Val Exchange (MAE) - Short Range: {maex1_peredge:.5f}")
        self.logger.info(f"[TRAIN] Val Exchange (MAE) - Long Range: {maex2_peredge:.5f}")

        wandb.log({
            "Test/MAE-Exchange": maex_peredge,
            "Test/MAE-Exchange-Short": maex1_peredge,
            "Test/MAE-Exchange-Long": maex2_peredge,
            "Test/MAE-Energy": mae_energy_per_atom,
            "Test/MAE-Force": mae_force_per_comp,
            "Test/Validation-Loss": avg_valloss,
            "Test/Validation-Loss-Energy": avg_vallosse,
            "Test/Validation-Loss-Forces": avg_vallossf,
            "Test/Validation-Loss-Exchange": avg_vallossx,
            "epoch": epoch
        })
        
        return avg_valloss, mae_force_per_comp, avg_vallossx

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
            
            train_loss = self.train_epoch(epoch)
            
            if (epoch + 1) % 1 == 0:
                val_loss, val_mae_force, val_maex = self.validate_epoch(epoch)
                
                # if self.scheduler is not None:
                    # Step based on Force MAE since forces are the hardest to fit
                    # self.scheduler.step(val_maex)
                
            line = f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {train_loss:.4f}, Time: {(time.time()-stime): .01f}\n" 
            self.logger.info(line)
            
            if epoch % 10 == 0:
                self.save_models(epoch, train_loss)
                
        wandb.finish()