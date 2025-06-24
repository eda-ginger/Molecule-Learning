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
########## DeepDTA
########################################################################################################################


class DeepDTA(torch.nn.Module):
    def __init__(self, n_filters=32):
        super(DeepDTA, self).__init__()
        self.relu = nn.ReLU()
        self.n_filters = n_filters

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(CHARISOSMILEN + 1, 128) # batch, 100, 128 -> batch, 128, 100
        self.conv_xd_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=4) # batch, 32, 97
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=4) # batch, 64, 94
        self.conv_xd_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=4) # batch, 96, 91

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, 128) # batch, 1000, 128 -> batch, 128, 1000
        self.conv_xt_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8) # batch, 32, 993
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8) # batch, 64, 986
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=8) # batch, 96, 979

        # dense
        self.classifier = nn.Sequential(
            nn.Linear(n_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def conv_module(self, x, conv1, conv2, conv3):
        x = conv1(x)
        x = F.relu(x)
        x = conv2(x)
        x = F.relu(x)
        x = conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, data):
        drug, target, _ = data
        xd, xt = drug.x, target.x
        
        # drug
        embedded_xd = self.embedding_xd(xd).permute(0, 2, 1)
        conv_xd = self.conv_module(embedded_xd, self.conv_xd_1, self.conv_xd_2, self.conv_xd_3)

        # protein
        embedded_xt = self.embedding_xt(xt).permute(0, 2, 1)
        conv_xt = self.conv_module(embedded_xt, self.conv_xt_1, self.conv_xt_2, self.conv_xt_3)
        
        # dense
        xc = torch.cat((conv_xd, conv_xt), 1)
        xc = self.classifier(xc)
        return xc

########################################################################################################################
########## GraphDTA
########################################################################################################################

# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        drug, target, _ = data 

        # get graph input
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        # # get protein input
        # target = data.target
        target = target.x

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(CHARPROTLEN + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        drug, target, _ = data
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        # target = data.target
        target = target.x

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = gap(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


########################################################################################################################
########## EGNN (3D)
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
    def __init__(self, dim=78):
        super(GCNNet, self).__init__()
        self.layer1 = GCNConv(dim, dim)
        self.layer2 = GCNConv(dim, dim)
        
    def forward(self, drug):
        feats, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x = self.layer1(feats, edge_index)
        x = self.layer2(x, edge_index)
        x = gap(x, batch)
        return x


from egnn_pytorch import EGNN

class EGNNNet(torch.nn.Module):
    def __init__(self, dim=78):
        super(EGNNNet, self).__init__()
        self.layer1 = EGNN(dim)
        self.layer2 = EGNN(dim)

    def forward(self, drug):
        feats = drug.x.unsqueeze(0).float()
        coors = torch.tensor(drug.pos).to(drug.x.device).float()
        
        feats, coors = self.layer1(feats, coors)
        feats, coors = self.layer2(feats, coors)
        
        feats = feats.squeeze(0)
        batch = torch.zeros(feats.shape[0], dtype=torch.long, device=feats.device)

        graph_emb = gap(feats, batch)
        return graph_emb
    
    
class FPNet(torch.nn.Module):
    def __init__(self, dim, out=64):
        super(FPNet, self).__init__()
        self.layer1 = torch.nn.Linear(dim, 256)
        self.layer2 = torch.nn.Linear(256, out)
        
    def forward(self, drug):
        feats = drug.x.float()
        feats = self.layer1(feats)
        feats = self.layer2(feats)
        return feats

class DTA_test(torch.nn.Module):
    def __init__(self, feature_type):
        super(DTA_test, self).__init__()
        
        if feature_type == 'FP-Morgan':
            self.drug_net = FPNet(dim=1024)
            drug_out = 64
        elif feature_type == 'FP-MACCS':
            self.drug_net = FPNet(dim=167)
            drug_out = 64
        elif feature_type == '2D-GNN':
            self.drug_net = GCNNet()
            drug_out = 78
        elif feature_type == '3D-GNN':
            self.drug_net = EGNNNet()
            drug_out = 78
        elif feature_type == 'CNN':
            self.drug_net = CNNNet()
            drug_out = 64
            
        self.drug_mlp = nn.Linear(drug_out, 64)
        
        self.target_net = TargetNet() # 64
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
        
    def forward(self, data):
        drug, target = data
        drug_feats = self.drug_net(drug)
        drug_feats = self.drug_mlp(drug_feats)
        target_feats = self.target_net(target)
        xc = torch.cat((drug_feats, target_feats), 1)
        xc = self.predictor(xc)
        return xc

