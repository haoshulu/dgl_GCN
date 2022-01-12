import torch
import torch.nn.functional as F
import dgl
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl.function as fn


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats*2, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat1, in_feat2):
        h = self.conv1(g, torch.cat([in_feat1, in_feat2],1))
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv1_expand = GraphConv(in_feats + h_feats, h_feats)
#         self.conv2_expand = GraphConv(in_feats + h_feats, h_feats)
#         self.conv3 = GraphConv(h_feats, num_classes)
#
#
#     def forward(self, g, in_feat1, in_feat2):
#         h = self.conv1(g, in_feat1)
#         h = F.relu(h)
#         for i in range(4):
#             h = self.conv2_expand(g, torch.cat([in_feat2, h], 1))
#             h = self.conv1_expand(g, torch.cat([in_feat1, h], 1))
#
#         h = self.conv3(g, h)
#         g.ndata['h'] = h
#         return dgl.mean_nodes(g, 'h')

