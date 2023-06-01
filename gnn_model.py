import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv

class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, 
                hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True, 
                feature_fusion=None, class_num=86):
        super(GIN_Net2, self).__init__()
        # 是否使用跳级联结构
        self.use_jk = use_jk
        # 是否训练epsilon
        self.train_eps = train_eps
        # 特征融合方式
        self.feature_fusion = feature_fusion

        self.fc1 = nn.Linear(in_feature, gin_in_feature)
        
        # 基于图卷积神经网络（GCN）的节点嵌入方法。它通过对节点的邻居进行聚合，生成新的节点表示。
        # 这里使用的GINConv是根据给定的一系列转换函数对输入进行聚合的算子。
        self.gin_conv1 = GINConv( 
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        
        # GINConv中的一个超参数，控制是否使用自适应的邻接矩阵。
        # 如果为True，则表示将邻接矩阵用于训练模型；如果为False，则表示使用恒等矩阵来代替邻接矩阵。
        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        # 跳级联结构
        # 一种汇集跳数信息的方法，用于提取节点特征。它通过将多个GINConv层的输出进行拼接或求和等操作，得到整个图的特征表示。
        # 这里的mode='cat'表示使用拼接操作，将多个GINConv层的输出拼接成一个长向量。
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers*hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)
    
    def reset_parameters(self):
        
        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()
        
        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()
    
    def forward(self, x, edge_index, train_edge_id, p=0.5):
        '''
        x.shape = [num_protein, in_len, embedding_size]
        edge_index.shape = [2, num_edge]
        train_edge_id = [num_train_edge]
        '''
        x = self.fc1(x)
        # x.shape = [num_of_protein, gin_in_feature]

        x = self.gin_conv1(x, edge_index)
        # x.shape = [num_of_protein, hidden]
        
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs += [x]

        if self.use_jk:
            x = self.jump(xs)
        
        # x.shape = [num_of_protein, hidden]
        x = F.relu(self.lin1(x))
        # x.shape = [num_of_protein, hidden]
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)
        # x.shape = [num_of_protein, hidden]
        # x  = torch.add(x, x_)

        node_id = edge_index[:, train_edge_id]
        # node_id.shape = [2, num_train_edge]
        
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc2(x)

        return x