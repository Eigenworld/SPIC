import torch
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import uniform


class D_inv_AConv(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Number of iterations
        cached (bool, optional): If set to :obj:True, the layer will cache
            the computation of :math:`{(\beta I + \mathbf{{\hat{D}}}^{-1}\mathbf{\hat{A})}}^k` 
            on first execution, and will use the cached version for further executions.
            This parameter should only be set to :obj:True in transductivelearning scenarios. 
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        improve (bool): Improve the weight of self loops.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, K=1, cached=True, bias=True,
                 improve=False, **kwargs):
        super(D_inv_AConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.improve = improve
        self.cached = cached

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.lin.reset_parameters()
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

        #if not self.cached:
            #x = self.lin(x)

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.My_norms(edge_index, x.size(0), edge_weight, self.improve
                                            , dtype=x.dtype)
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        if self.cached:
            x = self.lin(self.cached_result)

        return x
    
    def My_norms(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[col] * edge_weight 
        
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)
