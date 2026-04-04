import os
import argparse
import torch
import torch.optim as optim
import wandb

from data import getdata
from model import DSpinGNN
from train import load_checkpoint, count_parameters, MultiTaskLoss, Trainer
from logger import getlogger

def setup_wandb(args, dataset_size):
    os.environ["WANDB_MODE"] = "online"
    if args.WANDB_KEY:
        wandb.login(key=args.WANDB_KEY)

    wandb.init(
        entity="isamabdullah88-lahore-university-of-management-sciences",
        project=args.project,
        name=args.runname,
        config=vars(args), # Auto-logs all argparse arguments
        notes=args.wb_notes
    )

def main(args):
    logger = getlogger()
    logger.info(f"Hyperparameters: LR={args.lr}, Batch={args.batch_size}, Epochs={args.epochs}")

    # 1. Device Setup
    if args.mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    trainloader, valloader, _ = getdata(args.datasetpath, batch_size=args.batch_size)
    logger.info(f"Training samples: {len(trainloader.dataset)}, Validation samples: {len(valloader.dataset)}")

    # 3. Weights & Biases
    setup_wandb(args, len(trainloader.dataset))

    # 4. Model Initialization
    model = DSpinGNN(mps=args.mps)
    model = model.to(device)
    
    if args.finetune:
        checkpoint_path = os.path.join("./checkpoints", "latest-model.pt")
        # Pass the instantiated model into our new generic loader
        model, _, start_epoch, _ = load_checkpoint(model, checkpoint_path, device)
        logger.info(f"Loaded model from {checkpoint_path} starting at epoch {start_epoch}.")
    else:
        logger.info("Initialized new DSpinGNN model.")
        # OPTIONAL: Run your Least Squares initialization here!
        # z_map = {24: 0, 53: 1} # Cr: 0, I: 1
        # model = initialize_shift_scale(model, trainloader, z_map, device)

    model = model.to(device)
    logger.info(f"Total Parameters: {count_parameters(model):,}")

    # 5. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Define the Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6, verbose=True
    )

    criterion = MultiTaskLoss(wenergy=1.0, wforce=100.0, wexchange=1.0)

    # 6. Initialize Trainer & Run
    trainer = Trainer(
        model=model,
        train_loader=trainloader,
        val_loader=trainloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        config=args,
        scheduler=scheduler
    )
    
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DSpinGNN model.')
    parser.add_argument('--datasetpath', default="./DataSets/GNN/RattleGNN.pth", type=str)
    parser.add_argument('--project', default="DSpinGNN", type=str)
    parser.add_argument('--runname', default="Run_01_1k_Samples", type=str)
    parser.add_argument('--mps', default=False, type=bool)
    parser.add_argument('--finetune', default=False, type=bool)
    parser.add_argument('--ft_runname', default="Run_00_FullData", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--WANDB_KEY', default="", type=str)
    parser.add_argument('--wb_notes', default="", type=str)
    
    args = parser.parse_args()
    main(args)