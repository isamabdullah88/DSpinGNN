import os

# 1. Force the routing to your local Docker container
# os.environ["WANDB_BASE_URL"] = "http://localhost:8080"

# 2. Force the script to run online
os.environ["WANDB_MODE"] = "online"

# 3. Force your Local API Key 
# Go to http://localhost:8080/authorize in your browser, copy the key, and paste it here:
os.environ["WANDB_API_KEY"] = "b1c9982244acd53ac0d1"

import numpy as np
import time
import wandb

import torch
import torch.optim as optim
import torch.nn.functional as F

from data import getdata
from model import DSpinGNN, force
from trainutils import loadmodel, initialize_shift_scale, count_parameters, savecheckpoint
from test import evaluate

from logger import getlogger



def initwandb(lr, batch_size, epochs, dataset_size, project, runname, WANDB_KEY, wb_notes):
    wandb.login(key=WANDB_KEY)

    # --- 2. INITIALIZE RUN ---
    run = wandb.init(
        entity="isamabdullah88-lahore-university-of-management-sciences",
        project="DSpinGNN",
        name=runname, # Optional: Name this specific attempt
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "architecture": "DSpinGNN",
            "dataset_size": dataset_size
        },
        notes=wb_notes
    )


def criterion(energy, forces, data):
    
    losse = F.mse_loss(energy, data.y_energy)

    lossf = F.mse_loss(forces, data.y_forces)

    we = 1.0
    wf = 1000.0

    losstot = we * losse + wf * lossf

    return losstot, losse, lossf

def train(datasetpath, finetune, batch_size, project, runname, mps=False, lr=1e-2, epochs=5000, ft_runname="", WANDB_KEY="", wb_notes=""):
    logprefix = "[TRAIN] "

    logger.info("Hyperparameters:")
    logger.info(f"  Learning Rate: {lr}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Epochs: {epochs}")

    trainloader, valloader, _ = getdata(datasetpath, batch_size=batch_size)
    trainsize = int(len(trainloader.dataset) / batch_size)
    logger.info('Data loaded')
    logger.info(f"Training samples: {len(trainloader.dataset)}")
    logger.info(f"Validation samples: {len(valloader.dataset)}")
    
    checkpoints_dir = "./checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    # Initialize wandb logging
    initwandb(lr, batch_size, epochs, len(trainloader.dataset), project, runname, WANDB_KEY, wb_notes)


    if finetune:
        # basepath = "isamabdullah88-lahore-university-of-management-sciences/Thesis_NequIP_Aspirin"
        # runpath = os.path.join(basepath, ft_runname)
        # restored_ckpt = wandb.restore('results/checkpoints/latest-model.pt', run_path=runpath)
        # checkpoint_path = restored_ckpt.name
        checkpoint_path = os.path.join("./checkpoints", "latest-model.pt")
        
        model = loadmodel(checkpoint_path, mps=mps)
        logger.info(f"Loaded model from {checkpoint_path} for finetuning.")
    else:
        model = DSpinGNN(mps=mps)
        logger.info("Initialized new model for training.")
        # --- PLACE THIS BEFORE YOUR TRAINING LOOP ---
        # model = initialize_shift_scale(model, trainloader)

    logger.info("###################################")
    # print('Model Architecture: \n', model)
    logger.info("###################################")
    paramscount = count_parameters(model)
    logger.info(f"Total Parameters:     {paramscount:,}")

    if mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # ADDED: The LR Scheduler to prevent gradient plateau/explosions
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6, verbose=True
    # )

    logger.info("Starting training...")
    for epoch in range(epochs+1):
        
        # FIX 1: Ensure model is back in training mode every epoch
        model.train() 
        stime = time.time()
        
        for k, batch in enumerate(trainloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            batch.pos.requires_grad_(True)
            
            energy = model(batch)
            # NOTE: ensure force() uses create_graph=True internally so backward() works
            forces = force(energy, batch.pos)
            # logger.info(f"{logprefix}Forces: {np.linalg.norm(forces.cpu().detach().numpy(), axis=1)}")
            
            # logger.info(f"Energy shape: {energy.shape}, Ground Truth Energy shape: {batch.y_energy.shape}")
            # logger.info(f"Forces shape: {forces.shape}, Ground Truth Forces shape: {batch.y_forces.shape}")
            logger.info(f"Energy magnitudes: {np.sum(energy.cpu().detach().numpy())}, Ground Truth Energy magnitudes: {np.sum(batch.y_energy.cpu().numpy())}")
            logger.info(f"Force magnitudes: {np.sum(np.linalg.norm(forces.cpu().detach().numpy(), axis=1))}, Ground Truth Force magnitudes: {np.sum(np.linalg.norm(batch.y_forces.cpu().detach().numpy(), axis=1))}")
            loss, losse, lossf = criterion(energy, forces, batch)

            loss.backward()
            
            # ADDED: Gradient Clipping to prevent the Epoch 1200 Explosion
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({
                "train_loss": loss.item(),
                "train_loss_energy": losse.item(),
                "train_loss_forces": lossf.item(),
                "iter": trainsize*epoch + k,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
        # --- Validation Phase ---
        if (epoch + 1) % 1 == 0:
            logger.info("Starting evaluation...")
            # Note: evaluate() sets model.eval() internally!
            mae_energy, mae_force = evaluate(model, valloader, device=device)
            
            # FIX 2: Correctly accumulate validation losses
            total_valloss = 0.0
            total_vallosse = 0.0
            total_vallossf = 0.0
            num_val_graphs = 0
            
            # We must enable grad for validation force calculation
            with torch.enable_grad(): 
                for k, batch in enumerate(valloader):
                    batch = batch.to(device)
                    batch.pos.requires_grad_(True)
                    
                    energy = model(batch)
                    forces = force(energy, batch.pos)
                    
                    valloss, vallosse, vallossf = criterion(energy, forces, batch)
                    
                    # Accumulate based on batch size
                    graphs_in_batch = batch.num_graphs
                    total_valloss += valloss.item() * graphs_in_batch
                    total_vallosse += vallosse.item() * graphs_in_batch
                    total_vallossf += vallossf.item() * graphs_in_batch
                    num_val_graphs += graphs_in_batch

            # Calculate actual averages
            avg_valloss = total_valloss / num_val_graphs
            avg_vallosse = total_vallosse / num_val_graphs
            avg_vallossf = total_vallossf / num_val_graphs

            logger.info(f"{logprefix}RESULTS (Validation Set)")
            logger.info(f"{logprefix}Validation Loss: {avg_valloss:.5f}")
            logger.info(f"{logprefix}Validation Loss Energy: {avg_vallosse:.5f}")
            logger.info(f"{logprefix}Validation Loss Forces: {avg_vallossf:.5f}")
            
            wandb.log({
                "Validation-Loss": avg_valloss,
                "Validation-Loss-Energy": avg_vallosse,
                "Validation-Loss-Forces": avg_vallossf,
                "MAE-Energy/val": mae_energy,
                "MAE-Force/val": mae_force,
                "epoch": epoch
            })
            
            # ADDED: Step the scheduler based on Force MAE
            # scheduler.step(mae_force)

            line = f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time: {(time.time()-stime): .01f}\n" 
            logger.info(line)
        
        if epoch % 100 == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"Epoch-{epoch:04d}.pt")
            latest_ckpath = os.path.join(checkpoints_dir, "latest-model.pt")
            
            # FIX 3: Save to disk FIRST, then tell WandB to upload it
            savecheckpoint(checkpoint_path, epoch, model, optimizer, loss)
            savecheckpoint(latest_ckpath, epoch, model, optimizer, loss)
            
            wandb.save(checkpoint_path)
            wandb.save(latest_ckpath)

            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
    wandb.finish()
    # f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DSpinGNN model.')
    parser.add_argument('--datasetpath', default="./DataSets/GNN/RattleGNN.pth",
                       type=str, help='Path to the dataset.')
    parser.add_argument('--project', default="DSpinGNN", type=str, help='WandB project name.')
    parser.add_argument('--runname', default="Run_01_1k_Samples", type=str, help='WandB run name.')
    parser.add_argument('--mps', default=False, type=bool, help='Specify if the code is running on Mac/Cuda.')
    parser.add_argument('--finetune', default=False, type=bool, help='Fine-tune from a ' \
    'pre-trained model.')
    parser.add_argument('--ft_runname', default="Run_00_FullData", type=str, help='WandB run name path of the model to fine-tune from.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate for optimizer.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs.')
    parser.add_argument('--WANDB_KEY', default="", type=str, help='WandB API Key.')
    parser.add_argument('--wb_notes', default="", type=str, help='Notes to add to the WandB run.')
    args = parser.parse_args()


    logger = getlogger()
    train(args.datasetpath, finetune=args.finetune, batch_size=args.batch_size, project=args.project,
          runname=args.runname, ft_runname=args.ft_runname, mps=args.mps, epochs=args.epochs,
          WANDB_KEY=args.WANDB_KEY, lr=args.lr, wb_notes=args.wb_notes)