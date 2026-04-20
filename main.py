import os
import argparse
import torch
import torch.optim as optim
import wandb

from data import getdata
from model import StructureGNN, ExchangeMLP
from train import load_checkpoint, count_parameters, MultiTaskLoss, Trainer, initialize_shift_scale
from logger import getlogger


def init_optimizer(modelname, model):
    if modelname == "StructureModel":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif modelname == "ExchangeMLP":
        physics_params = []
        mlp_params = []

        # Separate the parameters based on their module names
        for name, param in model.named_parameters():
            if 'physics_base' in name:
                physics_params.append(param)
            elif 'ml_residual' in name:
                mlp_params.append(param)

        # The Physics gets to move fast; the MLP is heavily restricted
        optimizer = torch.optim.AdamW([
            {'params': physics_params, 'lr': 1e-3, 'weight_decay': 0.0}, 
            {'params': mlp_params, 'lr': args.lr, 'weight_decay': 0.4}     
        ])
    return optimizer

def setup_wandb(args, dataset_size):
    os.environ["WANDB_MODE"] = "online"
    if args.WANDB_KEY:
        wandb.login(key=args.WANDB_KEY)

    wandb.init(
        entity="isamabdullah88-lahore-university-of-management-sciences",
        project=args.project,
        name=args.runname,
        config=vars(args),
        notes=args.wb_notes
    )

def main(args):
    logger.info(f"Hyperparameters: LR={args.lr}, Batch={args.batch_size}, Epochs={args.epochs}")

    # Device
    if args.mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data Loading
    trainloader, valloader, _ = getdata(args.datasetpath, batch_size=args.batch_size)
    logger.info(f"Training samples: {len(trainloader.dataset)}, Validation samples: {len(valloader.dataset)}")

    # Setup wandb
    setup_wandb(args, len(trainloader.dataset))

    # Model Construction
    if args.modelname == "StructureModel":
        logger.info("Initializing StructureModel architecture.")
        model = StructureGNN()
    elif args.modelname == "ExchangeMLP":
        logger.info("Initializing ExchangeMLP architecture.")
        model = ExchangeMLP()
    
    wandb.config.update({"Model_Architecture": str(model)})
    wandb.save("model/*.py", base_path="./") # Save model architecture files to wandb for reproducibility
    
    if args.finetune:
        checkpoint_path = os.path.join("./checkpoints", "latest-model.pt")
        model, _, start_epoch, _ = load_checkpoint(model, checkpoint_path, device)
        logger.info(f"Loaded model from {checkpoint_path} starting at epoch {start_epoch}.")
    else:
        logger.info("Initialized new StructureGNN model.")

        if args.modelname == "StructureModel":
            z_map = {24: 0, 53: 1} 
            model = initialize_shift_scale(model, trainloader, z_map, device)
            logger.info("Initialized shift and scale parameters based on training data.")


    model = model.to(device)
    logger.info(f"Total Parameters: {count_parameters(model):,}")

    # Optimizer & Loss
    optimizer = init_optimizer(args.modelname, model)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=200, min_lr=1e-3, verbose=True
    )

    # Loss function with specified weights for each task
    criterion = MultiTaskLoss(args.modelname, wenergy=1.0, wforce=100.0, wexchange=10.0)

    # Train
    trainer = Trainer(
        modelname=args.modelname,
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        config=args,
        scheduler=scheduler
    )
    
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train StructureGNN model.')
    parser.add_argument('--datasetpath', default="./DataSets/GNN/RattleGNN.pth", type=str)
    parser.add_argument('--project', default="StructureGNN", type=str)
    parser.add_argument('--runname', default="Run_01_1k_Samples", type=str)
    parser.add_argument('--modelname', default="ExchangeMLP", type=str, choices=["StructureModel", "ExchangeMLP"])

    parser.add_argument('--mps', action='store_true', help='Enable MPS (Metal Performance Shaders)')
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning mode')

    parser.add_argument('--ft_runname', default="Run_00_FullData", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--WANDB_KEY', default="", type=str)
    parser.add_argument('--wb_notes', default="", type=str)
    
    args = parser.parse_args()

    logger = getlogger()
    
    main(args)