
import os
import time
import wandb

import torch
import torch.optim as optim
import torch.nn.functional as F

from data import getdata
from model import NequIP, force
from trainutils import loadmodel, initialize_shift_scale, count_parameters, savecheckpoint
from test import evaluate


def initwandb(lr, batch_size, epochs, dataset_size, project, runname, WANDB_KEY):
    wandb.login(key=WANDB_KEY)

    # --- 2. INITIALIZE RUN ---
    run = wandb.init(
        project=project,
        name=runname, # Optional: Name this specific attempt
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "architecture": "NequIP",
            "dataset_size": dataset_size
        }
    )


def criterion(energy, forces, data):
    
    losse = F.mse_loss(energy, data.y)

    lossf = F.mse_loss(forces, data.forces)

    we = 1.0
    wf = 100.0

    losstot = we * losse + wf * lossf

    return losstot

def train(data_dir, finetune, batch_size, project, runname, mps=False, lr=1e-2, epochs=5000, ft_runname="", WANDB_KEY=""):

    trainloader, valloader, _ = getdata(data_dir, mini=False, batch_size=batch_size)
    trainsize = int(len(trainloader.dataset) / batch_size)
    print('Data loaded')
    
    checkpoints_dir = "./checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    # Initialize wandb logging
    initwandb(lr, batch_size, epochs, len(trainloader.dataset), project, runname, WANDB_KEY)
    f = open('training-logs.txt', 'w')


    if finetune:
        # basepath = "isamabdullah88-lahore-university-of-management-sciences/Thesis_NequIP_Aspirin"
        # runpath = os.path.join(basepath, ft_runname)
        # restored_ckpt = wandb.restore('results/checkpoints/latest-model.pt', run_path=runpath)
        # checkpoint_path = restored_ckpt.name
        checkpoint_path = os.path.join("./checkpoints", "latest-model.pt")
        
        model = loadmodel(checkpoint_path, mps=mps)
        print(f"Loaded model from {checkpoint_path} for finetuning.")
    else:
        model = NequIP(mps=mps)
        print("Initialized new model for training.")
        # --- PLACE THIS BEFORE YOUR TRAINING LOOP ---
        model = initialize_shift_scale(model, trainloader)

    print("###################################")
    print('Model Architecture: \n', model)
    print("###################################")
    paramscount = count_parameters(model)
    print(f"Total Parameters:     {paramscount:,}")

    if mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)


    model.train()
    
    print("Starting training...")
    for epoch in range(epochs+1):

        stime = time.time()
        for k, batch in enumerate(trainloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pos = batch.pos.requires_grad_(True)
            
            # Forward pass with the model
            energy = model(batch.z, pos, batch.batch)

            forces = force(energy, pos)
            
            loss = criterion(energy, forces, batch)

            loss.backward()
            optimizer.step()

            # writer.add_scalar('Batch-Loss/train', loss.item(), trainsize*epoch + k)
            wandb.log({
                "train_loss": loss,
                "iter": trainsize*epoch + k,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
            
        # --- save model every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            
            checkpoint_path = os.path.join(checkpoints_dir, f"Epoch-{epoch:04d}.pt")
            latest_ckpath = os.path.join(checkpoints_dir, "latest-model.pt")

            savecheckpoint(checkpoint_path, epoch, model, optimizer, loss)
            savecheckpoint(latest_ckpath, epoch, model, optimizer, loss)



            mae_energy, mae_force = evaluate(model, valloader, device=device)
            # writer.add_scalar('MAE-Energy/val', mae_energy, epoch)
            # writer.add_scalar('MAE-Force/val', mae_force, epoch)
            wandb.log({
                "MAE_Energy/val": mae_energy,
                "MAE_Force/val": mae_force,
                "epoch": epoch
            })
            
            print("Saving model to wandb...")
            wandb.save(checkpoint_path)
            wandb.save(latest_ckpath)

            line = f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time Taken for single epoch: {(time.time()-stime): .01f}\n" 
            print(line)
            f.write(line)
            
    wandb.finish()
    f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train NequIP model.')
    parser.add_argument('--data_dir', default="/content/drive/My Drive/MS-Physics/ML-DFT/NequIP/Data",
                       type=str, help='Base directory for data.')
    parser.add_argument('--project', default="NequIP_Aspirin", type=str, help='WandB project name.')
    parser.add_argument('--runname', default="Run_01_1k_Samples", type=str, help='WandB run name.')
    parser.add_argument('--mps', default=False, type=bool, help='Specify if the code is running on Mac/Cuda.')
    parser.add_argument('--finetune', default=False, type=bool, help='Fine-tune from a ' \
    'pre-trained model.')
    parser.add_argument('--ft_runname', default="Run_00_FullData", type=str, help='WandB run name path of the model to fine-tune from.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate for optimizer.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs.')
    parser.add_argument('--WANDB_KEY', default="", type=str, help='WandB API Key.')
    args = parser.parse_args()


    train(args.data_dir, finetune=args.finetune, batch_size=args.batch_size, project=args.project,
          runname=args.runname, ft_runname=args.ft_runname, mps=args.mps, epochs=args.epochs,
          WANDB_KEY=args.WANDB_KEY, lr=args.lr)