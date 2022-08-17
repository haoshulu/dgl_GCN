import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv



class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats*2, h_feats*2)
        self.conv2 = GraphConv(h_feats*2, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)
        self.dropout = dropout

    def forward(self, g, in_feat1, in_feat2):
        h = self.conv1(g, torch.cat([in_feat1, in_feat2],1))
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv3(g, h)
        h = torch.tensor(torch.max(h),requires_grad=True)

        return h
# class GCNModel(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes, dropout):
#         super(GCNModel, self).__init__()
#         self.conv1 = GCNLayer(in_feats, h_feats)
#         self.conv2 = GCNLayer(h_feats, num_classes)
#         self.dropout = dropout
#
#     def forward(self, g, in_feat1, in_feat2):
#         h = self.conv1(g, torch.cat([in_feat1, in_feat2],1))
#         h = F.relu(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         h = self.conv2(g, h)
#         return h
