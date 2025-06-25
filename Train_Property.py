import os
import shutil
import warnings
warnings.filterwarnings(action='ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import argparse
import numpy as np
import pandas as pd
import deepchem as dc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from tqdm import tqdm
# from loader import MoleculeDataset
from utils.splitters import scaffold_split, random_split, random_scaffold_split
from sklearn.metrics import roc_auc_score

import wandb


criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        # Check for gradient issues
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"Average training loss: {avg_loss:.4f}")
    return avg_loss


def eval(args, model, device, loader, test=False):
    model.eval()
    y_true = []
    y_scores = []
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model(batch)
            y = batch.y.view(pred.shape).to(torch.float64)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

        # Calculate loss only if not test
        if not test:
            is_valid = y**2 > 0
            loss_mat = criterion(pred.double(), (y+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            total_loss += loss.item()
            total_batches += 1

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    
    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    avg_auc = sum(roc_list)/len(roc_list) if len(roc_list) > 0 else 0.0
    
    if test:
        return avg_auc
    else:
        avg_loss = total_loss / total_batches
        return avg_auc, avg_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    # parser.add_argument('--epochs', type=int, default=50,
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'bace', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    # parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="random", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--project', type=str, default='property_prediction', help='wandb project name')
    parser.add_argument('--use_regularization', action='store_true', default=False, help='use batch normalization and dropout')
    args = parser.parse_args()

    # 3-fold cross validation
    seeds = [16875, 33928, 40000]
    all_test_results = []

    for fold_idx, seed in enumerate(seeds):
        print(f"\n=== Fold {fold_idx + 1}/3 (Seed: {seed}) ===")
        
        # Initialize wandb for this fold
        if args.use_wandb:
            wandb.init(
                project=args.project,
                name=f"Fold_{fold_idx+1}",
                config=vars(args)
            )

        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        #Bunch of classification tasks
        if args.dataset == "tox21":
            num_tasks = 12
        elif args.dataset == "hiv":
            num_tasks = 1
        elif args.dataset == "pcba":
            num_tasks = 128
        elif args.dataset == "muv":
            num_tasks = 17
        elif args.dataset == "bace":
            num_tasks = 1
        elif args.dataset == "bbbp":
            num_tasks = 1
        elif args.dataset == "toxcast":
            num_tasks = 617
        elif args.dataset == "sider":
            num_tasks = 27
        elif args.dataset == "clintox":
            num_tasks = 2
        elif args.dataset == "freesolv":
            num_tasks = 1
        elif args.dataset == "esol":
            num_tasks = 1
        elif args.dataset == "lipo":
            num_tasks = 1
        else:
            raise ValueError("Invalid dataset name.")

        #set up dataset
        # from utils.loader import CustomMoleculeNet, MoleculeDataset
        from utils.molecule_feature import CustomMoleculeNet
        dataset = CustomMoleculeNet('dataset/', name=args.dataset.upper())
        
        # dataset = CustomMoleculeNet('dataset/', name='bace'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='bbbp'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='hiv'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='tox21'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='toxcast'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='sider'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='clintox'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='freesolv'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='esol'.upper())
        # dataset = CustomMoleculeNet('dataset/', name='lipo'.upper())
        
        
        # smiles_list = pd.read_csv('dataset/' + 'bace' + '/processed/smiles.csv', header=None)[0].tolist()
        # train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        # train_loader = DataLoader(dataset, batch_size=16, shuffle=True)    
        # y_true = []
        # for batch in train_loader:
        #     # print(batch)
        #     y_true.append(batch.y)
        #     # break
        
        # y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        # y_true.shape
        # y_true.sum(0)
        
        # roc_list = []
        # for i in range(y_true.shape[1]):
        #     # #AUC is only defined when there is at least one positive data.
        #     # if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
        #     #     is_valid = y_true[:,i]**2 > 0
        #     #     roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            
        #     #AUC is only defined when there is at least one positive data.
        #     positive_count = np.sum(y_true[:,i] == 1)
        #     negative_count = np.sum(y_true[:,i] == -1)
            
        #     if positive_count > 0 and negative_count > 0:
        #         is_valid = y_true[:,i]**2 > 0
        #         roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
        #     else:
        #         print(f"Task {i}: positive={positive_count}, negative={negative_count}")

        
        print(dataset)
        
        if args.split == "scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = seed)
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        print(train_dataset[0])

        # check label
        def check_label(dt, name):
            print(name)
            dty = np.array(dt.y)
            for i in range(dty.shape[1]):
                #AUC is only defined when there is at least one positive data.
                positive_count = np.sum(dty[:,i] == 1)
                negative_count = np.sum(dty[:,i] == -1)
                print(f"Task {i}: positive={positive_count}, negative={negative_count}")
            print('-'*10)

        def check_model_params(model):
            print("=== Model Parameter Statistics ===")
            total_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_count = param.numel()
                    total_params += param_count
                    print(f"{name}: {param_count} params, mean={param.data.mean():.4f}, std={param.data.std():.4f}")
            print(f"Total trainable parameters: {total_params}")
            print("="*40)

        check_label(train_dataset, 'train')
        check_label(valid_dataset, 'valid')
        check_label(test_dataset, 'test')
        
        # loader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        #set up model
        from process.models import Property_test
        print('num_tasks', num_tasks)
        model = Property_test(feature_type='2D-GNN', num_tasks=num_tasks, use_regularization=args.use_regularization)
        print(model)
        
        # Check model parameters
        check_model_params(model)

        # from process.model import GNN_graphpred
        # model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        # if not args.input_model_file == "":
        #     model.from_pretrained(args.input_model_file)
        
        model.to(device)

        #set up optimizer
        # #different learning rate for different part of GNN
        # model_param_group = []
        # model_param_group.append({"params": model.gnn.parameters()})
        # if args.graph_pooling == "attention":
        #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        train_loss_list = []

        best_val_acc = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            # Training with loss tracking
            train_loss = train(args, model, device, train_loader, optimizer)
            train_loss_list.append(train_loss)

            print("====Evaluation")
            if args.eval_train:
                train_acc, train_loss = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            
            # Only evaluate validation performance during training
            val_acc, val_loss = eval(args, model, device, val_loader)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")
            print(f"Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

            val_acc_list.append(val_acc)
            train_acc_list.append(train_acc)

            # Track best validation performance and save model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                print(f"New best validation performance: {best_val_acc:.4f} at epoch {best_epoch}")

            print("")

            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch
                })

        # Load best model for final test evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {best_epoch}")
        
        # Evaluate test performance only at the end
        print("====Final Test Evaluation")
        test_acc = eval(args, model, device, test_loader, test=True)
        test_acc_list.append(test_acc)
        
        print(f"Best validation performance: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Test performance: {test_acc:.4f}")

        # Log final test result to wandb
        if args.use_wandb:
            wandb.log({"test_acc": test_acc})
            wandb.finish()

        all_test_results.append(test_acc)

        with open('result.log', 'a+') as f:
            f.write(f"{args.dataset} fold_{fold_idx+1} {test_acc}\n")

    # Print average results
    avg_test_acc = np.mean(all_test_results)
    std_test_acc = np.std(all_test_results)
    print(f"\n=== Final Results ===")
    print(f"Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Individual Results: {[f'{acc:.4f}' for acc in all_test_results]}")

    with open('result.log', 'a+') as f:
        f.write(f"{args.dataset} average {avg_test_acc:.4f} ± {std_test_acc:.4f}\n")

if __name__ == "__main__":
    main()