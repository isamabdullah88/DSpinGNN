import os
import time
import torch
import wandb
from model import calcforce
from .trainutils import savecheckpoint
from .metrics import MetricsTracker

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
        
        # Initialize trackers
        self.train_metrics = MetricsTracker(device)
        self.val_metrics = MetricsTracker(device)

    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        for k, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            batch.pos.requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            energy, exchange = self.model(batch)
            forces = calcforce(energy, batch.pos)
            
            loss, losse, lossf, lossx = self.criterion(energy, forces, exchange, batch)
            loss.backward()
            self.optimizer.step()
            
            # Instantly update metrics
            self.train_metrics.update_loss(loss, losse, lossf, lossx, batch.num_graphs)

            # self.logger.info(f"[Training]")
            # self.logger.info(f"Sum of absolute exchange values greater than 1.0 in batch: {torch.sum(torch.abs(batch.y_exchange[torch.abs(batch.y_exchange) > 1.0])).item():.4f}")
            # self.logger.info(f"Sum of absolute predicted exchange values greater than 1.0 in batch: {torch.sum(torch.abs(exchange[torch.abs(exchange) > 1.0])).item():.4f}")

        metrics = self.train_metrics.get_averages()

        wandb.log({
            "Train/train_loss": metrics["loss"],
            "Train/train_loss_energy": metrics["losse"],
            "Train/train_loss_forces": metrics["lossf"],
            "Train/train_loss_exchange": metrics["lossx"],
            "iter": self.train_size * epoch + k,
            "Train/learning_rate": self.optimizer.param_groups[0]['lr']
        })
        
        return metrics["loss"]

    def validate_epoch(self, epoch):
        self.model.eval() 
        self.logger.info("Starting evaluation...")
        self.val_metrics.reset()
        
        with torch.enable_grad(): 
            for batch in self.train_loader:
                batch = batch.to(self.device)
                batch.pos.requires_grad_(True)
                
                energy, exchange = self.model(batch)
                forces = calcforce(energy, batch.pos)

                loss, losse, lossf, lossx = self.criterion(energy, forces, exchange, batch)

                # self.logger.info(f"[Validation]")
                # self.logger.info(f"Sum of absolute exchange values greater than 1.0 in batch: {torch.sum(torch.abs(batch.y_exchange[torch.abs(batch.y_exchange) > 1.0])).item():.4f}")
                # self.logger.info(f"Sum of absolute predicted values greater than 1.0 in batch: {torch.sum(torch.abs(exchange[torch.abs(exchange) > 1.0])).item():.4f}")
                
                # Update all metrics cleanly
                self.val_metrics.update_loss(loss, losse, lossf, lossx, batch.num_graphs)
                self.val_metrics.update_mae(energy, forces, exchange, batch)

        metrics = self.val_metrics.get_averages()

        self.logger.info("[VALIDATION] RESULTS (Validation Set)")
        self.logger.info(f"[VALIDATION] Val Loss (MSE):     {metrics['loss']:.5f}")
        self.logger.info(f"[VALIDATION] Val Energy (MAE):   {metrics['maee']:.5f} eV/atom")
        self.logger.info(f"[VALIDATION] Val Forces (MAE):   {metrics['maef']:.5f} eV/A")
        self.logger.info(f"[VALIDATION] Val Exchange (MAE): {metrics['maex']:.5f}")
        self.logger.info(f"[VALIDATION] Val Exchange (MAE) - Short Range: {metrics['maex1']:.5f}")
        self.logger.info(f"[VALIDATION] Val Exchange (MAE) - Long Range: {metrics['maex2']:.5f}")

        wandb.log({
            "Test/MAE-Exchange": metrics["maex"],
            "Test/MAE-Exchange-Short": metrics["maex1"],
            "Test/MAE-Exchange-Long": metrics["maex2"],
            "Test/MAE-Energy": metrics["maee"],
            "Test/MAE-Force": metrics["maef"],
            "Test/Validation-Loss": metrics["loss"],
            "Test/Validation-Loss-Energy": metrics["losse"],
            "Test/Validation-Loss-Forces": metrics["lossf"],
            "Test/Validation-Loss-Exchange": metrics["lossx"],
            "epoch": epoch
        })
        
        return metrics["loss"]

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
                val_loss = self.validate_epoch(epoch)
                
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
            line = f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {epochloss:.4f}, Time: {(time.time()-stime): .01f}\n" 
            self.logger.info(line)
            
            if epoch % 100 == 0:
                self.save_models(epoch, epochloss)
                
        wandb.finish()