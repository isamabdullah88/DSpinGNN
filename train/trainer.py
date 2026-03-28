import os
import time
import torch
import numpy as np
import wandb
from model import force
from trainutils import savecheckpoint

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, logger, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.config = config
        
        self.train_size = int(len(self.train_loader.dataset) / self.config.batch_size)
        self.checkpoints_dir = "./checkpoints"
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        for k, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            batch.pos.requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            energy = self.model(batch)
            forces = force(energy, batch.pos)
            
            # Debug tracking
            self.logger.info(f"Energy magnitudes: {np.sum(energy.cpu().detach().numpy()):.4f}, GT: {np.sum(batch.y_energy.cpu().numpy()):.4f}")
            self.logger.info(f"Force magnitudes: {np.sum(np.linalg.norm(forces.cpu().detach().numpy(), axis=1)):.4f}, GT: {np.sum(np.linalg.norm(batch.y_forces.cpu().detach().numpy(), axis=1)):.4f}")
            
            # Loss and Backprop
            loss, losse, lossf = self.criterion(energy, forces, batch)
            loss.backward()
            
            # Optional: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Logging
            wandb.log({
                "train_loss": loss.item(),
                "train_loss_energy": losse.item(),
                "train_loss_forces": lossf.item(),
                "iter": self.train_size * epoch + k,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            epoch_loss = loss.item()
            
        return epoch_loss

    def validate_epoch(self, epoch):
        self.model.eval() # Disable dropout/batchnorm for validation
        self.logger.info("Starting evaluation...")
        
        # Track Loss (MSE for the optimizer)
        total_valloss, total_vallosse, total_vallossf = 0.0, 0.0, 0.0
        
        # Track Metrics (MAE for human-readable physics)
        total_mae_energy, total_mae_force = 0.0, 0.0
        
        num_val_graphs = 0
        num_val_atoms = 0
        
        # We MUST enable grad to calculate validation forces!
        with torch.enable_grad(): 
            for batch in self.val_loader:
                batch = batch.to(self.device)
                batch.pos.requires_grad_(True)
                
                # Forward Pass
                energy = self.model(batch)
                forces = force(energy, batch.pos)
                
                # 1. Calculate MSE Losses
                valloss, vallosse, vallossf = self.criterion(energy, forces, batch)
                
                # 2. Calculate Absolute Errors (Detached to save memory)
                e_err = torch.abs(energy.detach().view(-1) - batch.y_energy.view(-1)).sum().item()
                f_err = torch.abs(forces.detach() - batch.y_forces).sum().item()
                
                # 3. Accumulate based on batch geometry
                graphs_in_batch = batch.num_graphs
                atoms_in_batch = batch.pos.shape[0]
                
                total_valloss += valloss.item() * graphs_in_batch
                total_vallosse += vallosse.item() * graphs_in_batch
                total_vallossf += vallossf.item() * graphs_in_batch
                
                total_mae_energy += e_err
                total_mae_force += f_err
                
                num_val_graphs += graphs_in_batch
                num_val_atoms += atoms_in_batch

        # Final MSE Averages
        avg_valloss = total_valloss / num_val_graphs
        avg_vallosse = total_vallosse / num_val_graphs
        avg_vallossf = total_vallossf / num_val_graphs

        # Final MAE Physics Metrics
        mae_energy_per_atom = total_mae_energy / num_val_atoms
        mae_force_per_comp = total_mae_force / (num_val_atoms * 3)

        self.logger.info("[TRAIN] RESULTS (Validation Set)")
        self.logger.info(f"[TRAIN] Val Loss (MSE):     {avg_valloss:.5f}")
        self.logger.info(f"[TRAIN] Val Energy (MAE):   {mae_energy_per_atom:.5f} eV/atom")
        self.logger.info(f"[TRAIN] Val Forces (MAE):   {mae_force_per_comp:.5f} eV/A")
        
        wandb.log({
            "Validation-Loss": avg_valloss,
            "Validation-Loss-Energy": avg_vallosse,
            "Validation-Loss-Forces": avg_vallossf,
            "MAE-Energy/val": mae_energy_per_atom,
            "MAE-Force/val": mae_force_per_comp,
            "epoch": epoch
        })
        
        return avg_valloss, mae_force_per_comp

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
            
            # 1. Train
            train_loss = self.train_epoch(epoch)
            
            # 2. Validate
            if (epoch + 1) % 1 == 0:
                val_loss, val_mae_force = self.validate_epoch(epoch)
                # If using a scheduler: self.scheduler.step(val_mae_force)
                
            # 3. Log Times
            line = f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {train_loss:.4f}, Time: {(time.time()-stime): .01f}\n" 
            self.logger.info(line)
            
            # 4. Checkpoint
            if epoch % 100 == 0:
                self.save_models(epoch, train_loss)
                
        wandb.finish()