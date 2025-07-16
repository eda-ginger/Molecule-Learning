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

CHARISOSMILEN = 68
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
    def __init__(self, dim, out=64):
        super(FPNet, self).__init__()
        self.layer1 = torch.nn.Linear(dim, 256)
        self.layer2 = torch.nn.Linear(256, out)
        
    def forward(self, drug):
        feats = drug.x.float()
        feats = self.layer1(feats)
        feats = F.relu(feats)
        feats = self.layer2(feats)
        feats = F.relu(feats)
        return feats


class CNNNet(torch.nn.Module):
    def __init__(self, n_filters=32):
        super(CNNNet, self).__init__()
        self.embedding_xd = nn.Embedding(CHARISOSMILEN + 1, 128) # batch, 100, 128 -> batch, 128, 100
        self.conv_xd_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=4) # batch, 32, 97
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=4) # batch, 64, 94
    
    def conv_module(self, x, conv1, conv2):
        x = conv1(x)
        x = F.relu(x)
        x = conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, drug):
        embedded_xd = self.embedding_xd(drug.x).permute(0, 2, 1)
        conv_xd = self.conv_module(embedded_xd, self.conv_xd_1, self.conv_xd_2)
        return conv_xd
    

class GCNNet(torch.nn.Module):
    def __init__(self, dim=133, out=False):
        super(GCNNet, self).__init__()
        if out:
            self.layer1 = GCNConv(dim, 128)
            self.layer2 = GCNConv(128, 128)
        else:
            self.layer1 = GCNConv(dim, dim)
            self.layer2 = GCNConv(dim, dim)
        
    def forward(self, drug):
        feats, edge_index, batch = drug.x.float(), drug.edge_index, drug.batch
        x = self.layer1(feats, edge_index)
        x = F.relu(x)
        x = self.layer2(x, edge_index)
        x = F.relu(x)
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
    
from model.schnet import SchNet

class Net3D(torch.nn.Module):
    def __init__(self):
        super(Net3D, self).__init__()
        self.layer = SchNet()
        
    def forward(self, drug):
        feats = self.layer(drug.z, drug.pos, drug.batch)
        return feats


class DTA_norm(torch.nn.Module):
    def __init__(self, feature_type):
        super(DTA_norm, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.drug_net = FPNet(dim=1024)
            drug_out = 64
        elif feature_type == 'FP-MACCS':
            self.drug_net = FPNet(dim=167)
            drug_out = 64
        elif feature_type == 'ChemBERTa':
            self.drug_net = FPNet(dim=384)
            drug_out = 64
        elif feature_type == '2D-GNN':
            self.drug_net = GCNNet(dim=133)
            drug_out = 133
        elif feature_type == '3D-GNN-egnn':
            self.drug_net = EGNN_Encoder(n_layers=2)
            drug_out = 64
        elif feature_type == '3D-GNN':
            self.drug_net = Net3D()
            drug_out = 64
        elif feature_type == 'CNN':
            self.drug_net = CNNNet()
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
    def __init__(self, feature_type):
        super(DTA_simple, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.drug_net = FPNet(dim=1024)
            drug_out = 64
        elif feature_type == 'FP-MACCS':
            self.drug_net = FPNet(dim=167)
            drug_out = 64
        elif feature_type == 'ChemBERTa':
            self.drug_net = FPNet(dim=384)
            drug_out = 64
        elif feature_type == '2D-GNN':
            self.drug_net = GCNNet(dim=133)
            drug_out = 133
        elif feature_type == '3D-GNN':
            self.drug_net = Net3D()
            drug_out = 64
        elif feature_type == '3D-GNN-egnn':
            self.drug_net = EGNN_Encoder(n_layers=2)
            drug_out = 64
        elif feature_type == 'CNN':
            self.drug_net = CNNNet()
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

class Property_norm(torch.nn.Module):
    def __init__(self, feature_type, num_tasks):
        super(Property_norm, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.molnet = FPNet(dim=1024)
            mol_out = 64
        elif feature_type == 'FP-MACCS':
            self.molnet = FPNet(dim=167)
            mol_out = 64
        elif feature_type == 'ChemBERTa':
            self.molnet = FPNet(dim=384)
            mol_out = 64
        elif feature_type == '2D-GNN':
            self.molnet = GCNNet(dim=133)
            mol_out = 133
        elif feature_type == '3D-GNN':
            self.molnet = Net3D()
            mol_out = 64
        elif feature_type == '3D-GNN-egnn':
            self.molnet = EGNN_Encoder(n_layers=2)
            mol_out = 64
        elif feature_type == 'CNN':
            self.molnet = CNNNet()
            mol_out = 64
            
        self.mol_mlp = nn.Sequential(
            nn.Linear(mol_out, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
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
    def __init__(self, feature_type, num_tasks):
        super(Property_simple, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.molnet = FPNet(dim=1024)
            mol_out = 64
        elif feature_type == 'FP-MACCS':
            self.molnet = FPNet(dim=167)
            mol_out = 64
        elif feature_type == 'ChemBERTa':
            self.molnet = FPNet(dim=384)
            mol_out = 64
        elif feature_type == '2D-GNN':
            self.molnet = GCNNet(dim=133)
            mol_out = 133
        
        # 2D-GNN-copy (2)
        elif feature_type == '2D-GNN-copy-2L-GIN':
            self.molnet = BaseGNN(num_layers=2, emb=False, bf_dim=2)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-2L-GIN-emb':
            self.molnet = BaseGNN(num_layers=2)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-5L-GIN':
            self.molnet = BaseGNN(num_layers=5, emb=False, bf_dim=2)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-5L-GIN-emb':
            self.molnet = BaseGNN(num_layers=5)
            mol_out = 300

        elif feature_type == '2D-GNN-copy-2L-GCN':
            self.molnet = BaseGNN(num_layers=2, gnn_type='gcn', emb=False, bf_dim=2)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-2L-GCN-emb':
            self.molnet = BaseGNN(num_layers=2, gnn_type='gcn')
            mol_out = 300
        elif feature_type == '2D-GNN-copy-5L-GCN':
            self.molnet = BaseGNN(num_layers=5, gnn_type='gcn', emb=False, bf_dim=2)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-5L-GCN-emb':
            self.molnet = BaseGNN(num_layers=5, gnn_type='gcn')
            mol_out = 300
        
        # 2D-GNN-copy > emb > linear
        elif feature_type == '2D-GNN-copy-2L-GIN-emb-fit':
            self.molnet = BaseGNN(num_layers=2, fit=True)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-5L-GIN-emb-fit':
            self.molnet = BaseGNN(num_layers=5, fit=True)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-2L-GCN-emb-fit':
            self.molnet = BaseGNN(num_layers=2, gnn_type='gcn', fit=True)
            mol_out = 300
        elif feature_type == '2D-GNN-copy-5L-GCN-emb-fit':
            self.molnet = BaseGNN(num_layers=5, gnn_type='gcn', fit=True)
            mol_out = 300
            
        # 2D-GNN-copy2 (106) - 1
        elif feature_type == '2D-GNN-copy2-2L-GIN':
            self.molnet = BaseGNN(num_layers=2, emb=False, bf_dim=106)
            mol_out = 300
        elif feature_type == '2D-GNN-copy2-5L-GIN':
            self.molnet = BaseGNN(num_layers=5, emb=False, bf_dim=106)
            mol_out = 300
        elif feature_type == '2D-GNN-copy2-2L-GCN':
            self.molnet = BaseGNN(num_layers=2, gnn_type='gcn', emb=False, bf_dim=106)
            mol_out = 300
        elif feature_type == '2D-GNN-copy2-5L-GCN':
            self.molnet = BaseGNN(num_layers=5, gnn_type='gcn', emb=False, bf_dim=106)
            mol_out = 300

        # 2D-GNN-copy3 (27)
        elif feature_type == '2D-GNN-copy3-2L-GIN':
            self.molnet = BaseGNN(num_layers=2, emb=False, bf_dim=27)
            mol_out = 300
        elif feature_type == '2D-GNN-copy3-5L-GIN':
            self.molnet = BaseGNN(num_layers=5, emb=False, bf_dim=27)
            mol_out = 300
        elif feature_type == '2D-GNN-copy3-2L-GCN':
            self.molnet = BaseGNN(num_layers=2, gnn_type='gcn', emb=False, bf_dim=27)
            mol_out = 300
        elif feature_type == '2D-GNN-copy3-5L-GCN':
            self.molnet = BaseGNN(num_layers=5, gnn_type='gcn', emb=False, bf_dim=27)
            mol_out = 300

        # 2D-GNN-tuto (133) - 0
        elif feature_type == '2D-GNN-tuto-2L-GIN':
            self.molnet = BaseGNN(num_layers=2, emb=False, bf_dim=133)
            mol_out = 300
        elif feature_type == '2D-GNN-tuto-5L-GIN':
            self.molnet = BaseGNN(num_layers=5, emb=False, bf_dim=133)
            mol_out = 300
        elif feature_type == '2D-GNN-tuto-2L-GCN':
            self.molnet = BaseGNN(num_layers=2, gnn_type='gcn', emb=False, bf_dim=133)
            mol_out = 300
        elif feature_type == '2D-GNN-tuto-5L-GCN':
            self.molnet = BaseGNN(num_layers=5, gnn_type='gcn', emb=False, bf_dim=133)
            mol_out = 300        
        
        elif feature_type == '3D-GNN':
            self.molnet = Net3D()
            mol_out = 64
        elif feature_type == '3D-GNN-egnn':
            self.molnet = EGNN_Encoder(n_layers=2)
            mol_out = 64
        elif feature_type == 'CNN':
            self.molnet = CNNNet()
            mol_out = 64
            
        self.predictor = nn.Sequential(
            nn.Linear(mol_out, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_tasks),
        )
        
    def forward(self, data):
        mol_feats = self.molnet(data)
        pred = self.predictor(mol_feats)
        return pred


class BaseGNN(torch.nn.Module):
    def __init__(self, num_layers=5, gnn_type='gin', emb_dim=300, bf_dim=2, emb=True, fit=False):
        super(BaseGNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.emb_dim = emb_dim
        
        self.emb = emb
        self.fit = fit
        
        if self.emb:        
            if not self.fit:
                self.x_embedding1 = nn.Embedding(120, emb_dim)
                self.x_embedding2 = nn.Embedding(8, emb_dim)
                
            else:
                self.x_embedding1 = nn.Embedding(120, 101)
                self.x_embedding2 = nn.Embedding(8, 5)
                self.linear = nn.Linear(106, emb_dim)
                
            nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.linear = nn.Linear(bf_dim, emb_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                from model.base_model import GCNConv as customGCNConv
                self.layers.append(customGCNConv(emb_dim, emb_dim))
            elif gnn_type == 'gat':
                from model.base_model import GATConv as customGATConv
                self.layers.append(customGATConv(emb_dim, emb_dim))
            elif gnn_type == 'gin':
                from model.base_model import GINConv as customGINConv
                self.layers.append(customGINConv(emb_dim, emb_dim))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
                
        # BatchNorm
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        
    def forward(self, drug):
        x, edge_index, edge_attr, batch = drug.x, drug.edge_index, drug.edge_attr, drug.batch
        
        if self.emb:
            if not self.fit:
                x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
            else: 
                # concat & linear
                x = torch.cat([self.x_embedding1(x[:, 0]), self.x_embedding2(x[:, 1])], dim=1)
                x = self.linear(x)
        else:
            x = self.linear(x.float())
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)  # edge_attr 전달
            x = self.batch_norms[i](x)
            x = F.relu(x)
        
        x = gap(x, batch)
        return x

if __name__ == "__main__":
    
    model = Property_simple('CNN', 3)
    molnet_param =sum([p.numel() for p in model.molnet.parameters() if p.requires_grad])
    predictor_param =sum([p.numel() for p in model.predictor.parameters() if p.requires_grad])
    print(f'Number of Molecule Encoder parameter: {molnet_param:,}')
    print(f'Number of Predictor parameter: {predictor_param:,}')
    print(f'Number of learnable parameter: {molnet_param + predictor_param:,}')

    model_param_group = [{"params": model.parameters()}]
    ## calculate the number of learnable parameter
    total_param = 0
    for group in model_param_group:
        total_param += sum(param.numel() for param in group['params'] if param.requires_grad)
    print(f'Number of learnable parameter: {total_param:,}')
