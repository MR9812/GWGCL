import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import random
import torch
from bn import *


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_norm=False):
        super().__init__()

        self.n_layers = n_layers
        self.use_norm = use_norm
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        
        if self.use_norm == True:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
                if self.use_norm == True:
                    self.bns.append(torch.nn.BatchNorm1d(hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = self.convs[i](graph, x)
            if self.use_norm == True:
                x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](graph, x)

        return x

class GWGRL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_groups):
        super().__init__()
        
        self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        #self.bn = torch.nn.BatchNorm1d(out_dim)
        self.bn = GroupWhitening1d(num_features=out_dim, num_groups=num_groups)
        #self.bn = Whitening1d(num_features=out_dim)

    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph, feat):
        h = self.backbone(graph, feat)
        h = self.bn(h)
        
        return h




