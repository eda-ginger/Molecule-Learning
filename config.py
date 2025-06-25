import argparse
import numpy as np

def set_config():
    parser = argparse.ArgumentParser()
    
    # Data arguments from prepare.py (assuming these are still needed for context)
    parser.add_argument('--feature', type=str, default='FP-Morgan', choices=['CNN','2D-GNN', '3D-GNN', 'FP-Morgan', 'FP-MACCS'], help='Feature type')
    parser.add_argument('--cache_dir', type=str, default='./data/', help='Directory for cached data and splits')
    parser.add_argument('--data_name', type=str, default='davis', help='Dataset name')

    # Training process arguments
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=9999, help='Patience for early stopping based on validation loss')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for DataLoader (0 for main process)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use for training')
    parser.add_argument('--model_type', type=str, default='test', choices=['test', 'simple'], help='Model type: test or simple')

    # details
    parser.add_argument('--loss', type=str, default='mse_mean', choices=['mse_mean', 'mse_sum'], help='Loss function for training')
    parser.add_argument('--aim', type=str, default='rmse', choices=['rmse', 'mse'], help='Aim for training')

    # Logging arguments
    parser.add_argument('--project', type=str, default='Basic', help='Directory to save checkpoints and logs')
    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=True, help='Use Weights & Biases for logging')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_config()
    print(args)

