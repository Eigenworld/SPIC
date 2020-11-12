import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class P_GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, k=2, negative_slope=0.2, dropout=0.6, bias=True, improved=False,drop=False,**kwargs):
        super(P_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bias = bias
        self.improved = improved
        self.drop = drop

        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(1, 2 * out_channels))
      
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        nn.init.xavier_normal_(self.att, gain=gain)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    ''' def reset_parameters(self):
        self.lin.reset_parameters()
        #self.lin2.reset_parameters()
        glorot(self.att)'''
    
    def My_norms(self, x, edge_index, num_nodes, improved=False, drop=False):
        edge_index_j = edge_index[0]
        edge_index_i = edge_index[1]
        x_j = x[edge_index_j]
        x_i = x[edge_index_i]
        
        alpha0 = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha1 = (torch.cat([x_j, x_i], dim=-1) * self.att).sum(dim=-1)
        alpha = (alpha0 + alpha1)/2.0

        if improved:
            self_weight = edge_index_i== edge_index_j
            alpha[self_weight] += 1.0
        
        #alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        
        alpha = softmax(alpha, edge_index_i, num_nodes)
        
        # Sample attention coefficients stochastically.
        if self.drop:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return alpha
    
    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.lin(x)
        
        norm = self.My_norms(x, edge_index, x.size(0))
            
        for _ in range(self.k):

            x = self.propagate(edge_index, size=size, x=x, norm=norm)
        
        #x = self.lin(x)
        return x


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def __repr__(self):
        return '{}({}, {}, k={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.k)