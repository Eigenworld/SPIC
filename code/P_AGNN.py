import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops


class P_AGNNConv(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        beta:Hyperparameter for the feature similarity.
        K (int): Number of iterations
        cached (bool, optional): If set to :obj:True, the layer will cache
            the computation of :math:`(\alpha I + P)^k` 
            on first execution, and will use the cached version for further executions.
            This parameter should only be set to :obj:True in transductivelearning scenarios. 
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        improved (bool): Improve the weight of self loops.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, beta=1.0, K=1, cached=True, bias=True,
                 improved=False, **kwargs):

        super(P_AGNNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.beta = beta
        self.cached = cached
        self.improved = improved

        self.lin = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        #self.beta.data.fill_(1)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached:
            x = self.lin(x)

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            
            x_norm = F.normalize(x, p=2, dim=-1)
            
            edge_index, norm = self.My_norms(x_norm, edge_index, x.size(0), edge_weight,self.improved
                                            ,dtype=x.dtype)
            # x是特征--迭代初始space； 而x_norm 是用来计算权值矩阵/相似度矩阵的！
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x
        #最后阶段乘上训练权值
        if self.cached:
            x = self.lin(self.cached_result)
        return x
    
    def My_norms(self, x_norm, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        
        edge_index_j = edge_index[0]
        edge_index_i = edge_index[1]
        x_norm_j = x_norm[edge_index_j]
        x_norm_i = x_norm[edge_index_i]
        alpha = self.beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes)
        return edge_index, alpha
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
