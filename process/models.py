########################################################################################################################
########## Import
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm

from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

CHARISOSMILEN = 65
CHARPROTLEN = 25

########################################################################################################################
########## Models
########################################################################################################################

class TargetNet(torch.nn.Module):
    def __init__(self, n_filters=32):
        super(TargetNet, self).__init__()
        
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, 128) # batch, 1000, 128 -> batch, 128, 1000
        self.conv_xt_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8) # batch, 32, 993
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8) # batch, 64, 986
    
    def conv_module(self, x, conv1, conv2):
        x = conv1(x)
        x = F.relu(x)
        x = conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, target):
        embedded_xt = self.embedding_xt(target.x).permute(0, 2, 1)
        conv_xt = self.conv_module(embedded_xt, self.conv_xt_1, self.conv_xt_2)
        return conv_xt
    

class FPNet(torch.nn.Module):
    def __init__(self, dim, out=64, use_regularization=True):
        super(FPNet, self).__init__()
        self.layer1 = torch.nn.Linear(dim, 256)
        self.layer2 = torch.nn.Linear(256, out)
        
        # Optional batch norm and dropout
        self.use_regularization = use_regularization
        if use_regularization:
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(out)
            self.dropout = nn.Dropout(0.1)
        
    def forward(self, drug):
        feats = drug.x.float()
        feats = self.layer1(feats)
        feats = F.relu(feats)
        if self.use_regularization:
            feats = self.bn1(feats)
            feats = self.dropout(feats)
        
        feats = self.layer2(feats)
        feats = F.relu(feats)
        if self.use_regularization:
            feats = self.bn2(feats)
            feats = self.dropout(feats)
        return feats


class CNNNet(torch.nn.Module):
    def __init__(self, n_filters=32, use_regularization=True):
        super(CNNNet, self).__init__()
        self.embedding_xd = nn.Embedding(CHARISOSMILEN + 1, 128) # batch, 100, 128 -> batch, 128, 100
        self.conv_xd_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=4) # batch, 32, 97
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=4) # batch, 64, 94
        
        # Optional batch norm and dropout
        self.use_regularization = use_regularization
        if use_regularization:
            self.bn1 = nn.BatchNorm1d(n_filters)
            self.bn2 = nn.BatchNorm1d(n_filters * 2)
            self.dropout = nn.Dropout(0.1)
    
    def conv_module(self, x, conv1, conv2):
        x = conv1(x)
        x = F.relu(x)
        if self.use_regularization:
            x = self.bn1(x)
            x = self.dropout(x)
        
        x = conv2(x)
        x = F.relu(x)
        if self.use_regularization:
            x = self.bn2(x)
            x = self.dropout(x)
        
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, drug):
        embedded_xd = self.embedding_xd(drug.x).permute(0, 2, 1)
        conv_xd = self.conv_module(embedded_xd, self.conv_xd_1, self.conv_xd_2)
        return conv_xd
    

class GCNNet(torch.nn.Module):
    def __init__(self, dim=133, use_regularization=True):
        super(GCNNet, self).__init__()
        self.layer1 = GCNConv(dim, dim)
        self.layer2 = GCNConv(dim, dim)
        
        # Optional batch norm and dropout
        self.use_regularization = use_regularization
        if use_regularization:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
            self.dropout = nn.Dropout(0.1)
        
    def forward(self, drug):
        feats, edge_index, batch = drug.x.float(), drug.edge_index, drug.batch
        x = self.layer1(feats, edge_index)
        x = F.relu(x)
        if self.use_regularization:
            x = self.bn1(x)
            x = self.dropout(x)
        
        x = self.layer2(x, edge_index)
        x = F.relu(x)
        if self.use_regularization:
            x = self.bn2(x)
            x = self.dropout(x)
        
        x = gap(x, batch)
        return x


from process.egnn import EGNN

class EGNN_Encoder(nn.Module):
    def __init__(
        self,
        in_node_nf=27,
        hidden_nf=64,
        out_node_nf=64,
        in_edge_nf=6,
        device="cuda",
        act_fn=nn.SiLU(),
        n_layers=12,
        residual=True,
        attention=True,
        normalize=False,
        tanh=False,
    ):
        super(EGNN_Encoder, self).__init__()

        # Initialize the EGNN model
        self.egnn = EGNN(
            in_node_nf,
            hidden_nf,
            out_node_nf,
            in_edge_nf,
            device,
            act_fn,
            n_layers,
            residual,
            attention,
            normalize,
            tanh,
        )

    def forward(self, h, x, edges, edge_attr):
        # Pass through EGNN layers to get updated node features and coordinates
        h, x = self.egnn(h, x, edges, edge_attr)
        return h
    

class DTA_test(torch.nn.Module):
    def __init__(self, feature_type, use_regularization=False):
        super(DTA_test, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.drug_net = FPNet(dim=1024, use_regularization=use_regularization)
            drug_out = 64
        elif feature_type == 'FP-MACCS':
            self.drug_net = FPNet(dim=167, use_regularization=use_regularization)
            drug_out = 64
        elif feature_type == '2D-GNN':
            self.drug_net = GCNNet(dim=133, use_regularization=use_regularization)
            drug_out = 133
        elif feature_type == '3D-GNN':
            self.drug_net = EGNN_Encoder(n_layers=2)
            drug_out = 64
        elif feature_type == 'CNN':
            self.drug_net = CNNNet(use_regularization=use_regularization)
            drug_out = 64
            
        self.drug_mlp = nn.Sequential(
            nn.Linear(drug_out, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )
        
        self.target_net = TargetNet() # 64
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    
        
    def forward(self, data):
        drug, target = data
        drug_feats = self.drug_net(drug)
        drug_feats = self.drug_mlp(drug_feats)
        target_feats = self.target_net(target)
        xc = torch.cat((drug_feats, target_feats), 1)
        xc = self.predictor(xc)
        return xc



class DTA_simple(torch.nn.Module): # not use batch norm
    def __init__(self, feature_type, use_regularization=False):
        super(DTA_simple, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.drug_net = FPNet(dim=1024, use_regularization=use_regularization)
            drug_out = 64
        elif feature_type == 'FP-MACCS':
            self.drug_net = FPNet(dim=167, use_regularization=use_regularization)
            drug_out = 64
        elif feature_type == '2D-GNN':
            self.drug_net = GCNNet(dim=133, use_regularization=use_regularization)
            drug_out = 133
        elif feature_type == '3D-GNN':
            self.drug_net = EGNN_Encoder(n_layers=2)
            drug_out = 64
        elif feature_type == 'CNN':
            self.drug_net = CNNNet(use_regularization=use_regularization)
            drug_out = 64
            
        self.drug_mlp = nn.Sequential(
            nn.Linear(drug_out, 64),
            nn.ReLU(),
        )
        
        self.target_net = TargetNet() # 64
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    
        
    def forward(self, data):
        drug, target = data
        drug_feats = self.drug_net(drug)
        drug_feats = self.drug_mlp(drug_feats)
        target_feats = self.target_net(target)
        xc = torch.cat((drug_feats, target_feats), 1)
        xc = self.predictor(xc)
        return xc    

class Property_test(torch.nn.Module):
    def __init__(self, feature_type, num_tasks, use_regularization=False):
        super(Property_test, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.molnet = FPNet(dim=1024, use_regularization=use_regularization)
            mol_out = 64
        elif feature_type == 'FP-MACCS':
            self.molnet = FPNet(dim=167, use_regularization=use_regularization)
            mol_out = 64
        elif feature_type == '2D-GNN':
            self.molnet = GCNNet(dim=133, use_regularization=use_regularization)
            mol_out = 133
        elif feature_type == '3D-GNN':
            self.molnet = EGNN_Encoder(n_layers=2)
            mol_out = 64
        elif feature_type == 'CNN':
            self.molnet = CNNNet(use_regularization=use_regularization)
            mol_out = 64
            
        self.mol_mlp = nn.Sequential(
            nn.Linear(mol_out, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_tasks),
        )
    
        
    def forward(self, data):
        mol_feats = self.molnet(data)
        mol_feats = self.mol_mlp(mol_feats)
        pred = self.predictor(mol_feats)
        return pred


    

class Property_simple(torch.nn.Module): # not use batch norm
    def __init__(self, feature_type, num_tasks, use_regularization=False):
        super(Property_simple, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.molnet = FPNet(dim=1024, use_regularization=use_regularization)
            mol_out = 64
        elif feature_type == 'FP-MACCS':
            self.molnet = FPNet(dim=167, use_regularization=use_regularization)
            mol_out = 64
        elif feature_type == '2D-GNN':
            self.molnet = GCNNet(dim=133, use_regularization=use_regularization)
            mol_out = 133
        elif feature_type == '3D-GNN':
            self.molnet = EGNN_Encoder(n_layers=2)
            mol_out = 64
        elif feature_type == 'CNN':
            self.molnet = CNNNet(use_regularization=use_regularization)
            mol_out = 64
            
        self.mol_mlp = nn.Sequential(
            nn.Linear(mol_out, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_tasks),
        )
        
    def forward(self, data):
        mol_feats = self.molnet(data)
        mol_feats = self.mol_mlp(mol_feats)
        pred = self.predictor(mol_feats)
        return pred


if __name__ == "__main__":
    egnn = EGNN_Encoder(device='cpu')
    
    from process.t_graph import Graph_Featurizer
    from openbabel import pybel
    
    lig = next(pybel.readfile("sdf", './1aq7_ligand.sdf'))
    lig = pybel.readstring("smi", "CCCC")
    ligand_features, ligand_coords = Graph_Featurizer().get_node_features(lig, source='ligand', complex_bool=False)
    lig_edges, lig_attrs = Graph_Featurizer().get_bond_based_edges(lig)
    print(ligand_features.shape, ligand_coords.shape, len(lig_edges), len(lig_attrs))
    
    lig.addh()
    lig.make3D()
    lig.localopt()
    lig.write("test.sdf")
    
    ligand_features=torch.tensor(ligand_features, dtype=torch.float)
    ligand_coords = torch.tensor(ligand_coords, dtype=torch.float32)
    ligand_edges=torch.tensor(lig_edges, dtype=torch.long).t().contiguous()
    ligand_edge_attr=torch.tensor(lig_attrs, dtype=torch.float)
    
    print(ligand_features.shape, ligand_coords.shape, ligand_edges.shape, ligand_edge_attr.shape)
    
    egnn(ligand_features, ligand_coords, ligand_edges, ligand_edge_attr)
    
