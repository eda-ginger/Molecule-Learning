import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
import sys
import numpy as np
from pathlib import Path
import argparse # For argparse.Namespace
import logging # Added logging
from tqdm import tqdm
import wandb
import json # Added for saving metrics

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import get_model
from config import set_config # Keep for potential use, though args come from main.py
from utils.tools import set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint
from utils.metrics import calculate_regression_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='"%(asctime)s [%(levelname)s] %(message)s"',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


from torch_geometric.data import Dataset, Data, Batch

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    @staticmethod
    def collate_fn(batch):
        ligands = Batch.from_data_list([item['Drug_Rep'] for item in batch])
        proteins = Batch.from_data_list([item['Target_Rep'] for item in batch])
        return ligands, proteins



def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, fold_idx):
    model.train()
    total_loss = 0
    
    # Create progress bar
    pbar = tqdm(train_loader, desc=f'Fold {fold_idx} - Epoch {epoch}', 
                leave=True, dynamic_ncols=True)
    
    for batch_idx, inputs in enumerate(pbar):
        ligands, sequences = inputs
        affinities = ligands.y
        
        ligands = ligands.to(device)
        sequences = sequences.to(device)
        affinities = affinities.to(device)

        optimizer.zero_grad()
        
        predictions = model((ligands, sequences))
        loss = loss_fn(predictions, affinities)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_reals = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            ligands, sequences = inputs
            affinities = ligands.y

            ligands = ligands.to(device)
            sequences = sequences.to(device)
            affinities = affinities.to(device)

            predictions = model((ligands, sequences))
            loss = loss_fn(predictions, affinities)
            
            total_loss += loss.item()
            all_preds.append(predictions.detach())
            all_reals.append(affinities.detach())
            
    avg_loss = total_loss / len(data_loader)
    all_preds_cat = torch.cat(all_preds, dim=0)
    all_reals_cat = torch.cat(all_reals, dim=0)
    
    eval_metrics = calculate_regression_metrics(all_preds_cat, all_reals_cat)
    return avg_loss, eval_metrics

def Train_CV(args):
    device = get_device(args)
    
    # path
    cache_path = Path(args.cache_dir) / 'davis' / 'feature' / args.feature
    logger.info(f"Cache path: {cache_path}")
    
    # load dataset            
    trn_pkl = cache_path / f'trn.pkl'
    with open(trn_pkl, 'rb') as f:
        trn_samples = pickle.load(f)
    train_dataset = CustomDataset(trn_samples)

    val_pkl = cache_path / f'val.pkl'
    with open(val_pkl, 'rb') as f:
        val_samples = pickle.load(f)
    val_dataset = CustomDataset(val_samples)
    logger.info(f"TRN: {len(train_dataset)}, VAL: {len(val_dataset)}")
    
    tst_pkl = cache_path / f'tst.pkl'
    with open(tst_pkl, 'rb') as f:
        tst_samples = pickle.load(f)
    tst_dataset = CustomDataset(tst_samples)
    logger.info(f"TST: {len(tst_dataset)}")
    
    # main loop
    seeds = [16875, 33928, 40000]
    all_test_metrics = []
    
    for s_idx, seed in enumerate(seeds):
        fold_idx = s_idx + 1

        # args are passed from main.py
        logger.info(f"Seed: {seed}")
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
    
        logger.info(f"\n--- Starting Fold {s_idx+1}/{len(seeds)} ---")

        # Initialize wandb for each fold
        if args.use_wandb:
            wandb.init(
                project=args.project,
                name=f"Fold_{fold_idx}",
                config=vars(args)
            )
            
        # create dataloaders
        trn_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CustomDataset.collate_fn, num_workers=args.n_workers, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=CustomDataset.collate_fn, num_workers=args.n_workers)
        
        # create model
        model = get_model(args).to(device)
        logger.info(f"Feature: {args.feature} | Parameters: {count_parameters(model)}")
        logger.info(f'Model Architecture:\n{model}')

        optimizer = optim.AdamW(model.parameters())
        loss_fn = nn.MSELoss(reduction='mean')

        best_val_metric = float('inf')
        epochs_no_improve = 0
        start_epoch = 0
        
        fold_output_dir = Path('logs') / Path(args.project) / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        logger.info(f"Starting training for fold {fold_idx} from epoch {start_epoch+1}")
        for epoch in range(start_epoch + 1, args.n_epochs + 1):
            train_loss = train_one_epoch(model, trn_loader, optimizer, loss_fn, device, epoch, fold_idx)
            val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)

            if args.aim == 'rmse':
                current_val_metric = val_metrics['rmse']
            elif args.aim == 'mse':
                current_val_metric = val_loss
            else:
                raise ValueError(f"Invalid aim: {args.aim}")

            if current_val_metric < best_val_metric:
                logger.info(f"New best validation {args.aim.upper()}: {current_val_metric:.4f}")
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                
                # Save best model for this fold
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_val_metric,
                    'args': vars(args) # Save args as dict
                }, is_best=True, filename='model_best.pt', output_dir=fold_output_dir)
                
                # Also save a copy with fold information in project root
                project_root = Path('logs') / Path(args.project)
                project_root.mkdir(parents=True, exist_ok=True)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_val_metric,
                    'args': vars(args) # Save args as dict
                }, is_best=True, filename=f'model_best_fold_{fold_idx}.pt', output_dir=project_root)
            else:
                epochs_no_improve += 1
            
            # Log losses to wandb
            if args.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss})

            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} for fold {fold_idx}")
                break
        
        # fold finished
        logger.info(f"Finished training for fold {fold_idx}. Best Val Metric ({args.aim.upper()}): {best_val_metric:.4f}")

        # evaluate best model on test sets
        logger.info(f"Evaluating best model from fold {fold_idx} on test sets...")
        best_model_path = fold_output_dir / "model_best.pt"
        
        model_test = get_model(args).to(device)
        model_test = load_checkpoint(best_model_path, model_test) 

        # testing
        test_loader = DataLoader(tst_dataset, batch_size=300, shuffle=False, 
                                collate_fn=CustomDataset.collate_fn, num_workers=args.n_workers)
        _, test_metrics = evaluate(model_test, test_loader, loss_fn, device)
        all_test_metrics.append(test_metrics)
        
        # Log individual fold test performance
        logger.info(f"\n--- Fold {fold_idx} Test Results ---")
        for metric, value in test_metrics.items():
            logger.info(f"Fold {fold_idx} Test {metric}: {value:.4f}")
        logger.info(f"--- End Fold {fold_idx} Results ---\n")
            
        # save fold's test metrics
        fold_metrics_file = fold_output_dir / f"TST_metrics.json"
        test_metrics_serializable = {k: float(v) for k, v in test_metrics.items()}
        with open(fold_metrics_file, 'w') as f:
            json.dump(test_metrics_serializable, f, indent=4)

        if args.use_wandb:
            wandb.log({
                f"TST/rmse": test_metrics['rmse'],
                f"TST/mae": test_metrics['mae'],
                f"TST/sd": test_metrics['sd'],
                f"TST/pcc": test_metrics['pcc'],
                f"TST/r2": test_metrics['r2'],
                f"TST/ci": test_metrics['ci']
                })

        # Finish wandb run for this fold
        if args.use_wandb:
            wandb.finish()

    logger.info("Training and evaluation finished.")
    
    # Calculate average and standard deviation of test metrics across all folds
    metrics_names = list(all_test_metrics[0].keys())
    avg_test_metrics = {}
    std_test_metrics = {}
    
    for metric in metrics_names:
        values = [fold_metrics[metric] for fold_metrics in all_test_metrics]
        avg_test_metrics[metric] = float(np.mean(values))
        std_test_metrics[metric] = float(np.std(values))
    
    # Save average metrics to file
    metrics_file = Path('logs') / Path(args.project) / "average_test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(avg_test_metrics, f, indent=4)
    
    logger.info(f"Average test metrics saved to {metrics_file}")
    
    # Log and save results
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    # Print individual fold results
    logger.info("\nIndividual Fold Results:")
    logger.info("-" * 40)
    for fold_idx, fold_metrics in enumerate(all_test_metrics, 1):
        logger.info(f"Fold {fold_idx}:")
        for metric in metrics_names:
            logger.info(f"  {metric}: {fold_metrics[metric]:.4f}")
        logger.info("")
    
    # Print average results
    logger.info("\nCross-Validation Average Results:")
    logger.info("-" * 40)
    for metric in metrics_names:
        avg_value = avg_test_metrics[metric]
        std_value = std_test_metrics[metric]
        result_line = f"{metric}: {avg_value:.4f} (±{std_value:.4f})"
        logger.info(result_line)
    
    logger.info("="*60)
    
    # Save detailed results to file
    result_file = Path('logs') / Path(args.project) / "result.log"
    with open(result_file, 'a+') as f:
        # Write project information
        f.write(f"\n{'='*60}\n")
        f.write(f"Project: {args.project}\n")
        f.write(f"Feature: {args.feature}\n")
        f.write(f"Parameters: {count_parameters(model):,}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.n_epochs}\n")
        f.write(f"Cross-validation Folds: {len(seeds)}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"{'='*60}\n\n")
        
        # Write model architecture
        f.write("Model Architecture:\n")
        f.write(str(model))
        f.write(f"\n\n{'='*60}\n")
        
        # Write individual fold results
        f.write("Individual Fold Test Results:\n")
        f.write("-" * 40 + "\n")
        for fold_idx, fold_metrics in enumerate(all_test_metrics, 1):
            f.write(f"Fold {fold_idx}:\n")
            for metric in metrics_names:
                f.write(f"  {metric}: {fold_metrics[metric]:.4f}\n")
            f.write("\n")
        
        # Write average results
        f.write("Cross-Validation Average Results:\n")
        f.write("-" * 40 + "\n")
        for metric in metrics_names:
            avg_value = avg_test_metrics[metric]
            std_value = std_test_metrics[metric]
            result_line = f"{metric}: {avg_value:.4f} (±{std_value:.4f})"
            f.write(result_line + "\n")
        
        f.write(f"\n{'='*60}\n")

if __name__ == "__main__":
    args = set_config()
    Train_CV(args)
