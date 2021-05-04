import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl 
from dgl.transform import add_self_loop
from dgl.nn.pytorch.conv import GINConv
import dgl.function as dfunc

from sklearn.preprocessing import normalize

import numpy as np 

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class node_task_wrapper(nn.Module):
    def __init__(self, model_name, params_dict):
        super(node_task_wrapper, self).__init__()

        params_dict['output_dim'] = params_dict['num_classes']
        self.model = model_name(params_dict).to(device)

    def forward(self, batched_graph):
        return self.model(batched_graph)

class graph_task_wrapper(nn.Module):
    def __init__(self, model_name, params_dict):
        super(graph_task_wrapper, self).__init__()

        self.model = model_name(params_dict).to(device)
        self.final_predict = nn.Linear(params_dict['output_dim'], params_dict['num_classes'])

    def forward(self, batched_graph):
        y = self.model(batched_graph)

        batched_graph.ndata['y'] = y
        y = dgl.sum_nodes(batched_graph, 'y')
        y = self.final_predict(F.relu(y))

        return y

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, final_nonlinear = True, bn = False, reset_parameters = False):
        super(MLP, self).__init__()
        dim_list = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(m,n) for m, n in zip(dim_list[:-1], dim_list[1:])])
        self.final_nonlinear = final_nonlinear
        self.bn = bn

        if self.bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hid_dim, track_running_stats = False) for i in range(num_layers - 1)])
            if self.final_nonlinear:
                self.bns.append(nn.BatchNorm1d(out_dim))

        if reset_parameters:
            self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.linears)):
            nn.init.uniform_(self.linears[i].weight, a=-1, b=1)
            nn.init.uniform_(self.linears[i].bias, a=-1, b=1)

    def forward(self, x):
        if not self.bn:
            for linear in self.linears[:-1]:
                x = F.relu(linear(x))

            x = self.linears[-1](x)

            if self.final_nonlinear:
                x = F.relu(x)
        else:
            if self.final_nonlinear:
                for linear, bn in zip(self.linears, self.bns):
                    x = F.relu(bn(linear(x)))
            else:
                for linear, bn in zip(self.linears[:-1], self.bns):
                    x = F.relu(bn(linear(x)))

                x = self.linears[-1](x)

        return x

class GIN_JK(nn.Module):
    def __init__(self, params_dict):
        super(GIN_JK, self).__init__()

        self.input_dim = params_dict['input_dim']
        self.input_channel = params_dict['input_channel']
        hid_dim = params_dict['hid_dim']
        output_dim = params_dict['output_dim']
        self.output_channel = params_dict['output_channel']
        self.num_hops = params_dict['num_hops']

        dim_list = [self.input_dim] + [hid_dim] * self.num_hops

        self.mlps = nn.ModuleList([MLP(in_dim = m, hid_dim = n, out_dim = n, num_layers = 2, final_nonlinear = True) for m, n in zip(dim_list[:-1], dim_list[1:])])

        self.gin_conv_layers = nn.ModuleList([GINConv(apply_func = mlp, aggregator_type = 'sum', init_eps = 0, learn_eps = True) for mlp in self.mlps])
        
        self.final_prediction = MLP(in_dim = sum(dim_list), hid_dim = hid_dim, out_dim = output_dim, num_layers = 2, final_nonlinear = False)

    def forward(self, batched_graph):
        feat = batched_graph.ndata[self.input_channel]
        feat_list = [feat]

        for gin_conv in self.gin_conv_layers:
            feat = gin_conv(batched_graph, feat)
            feat_list.append(feat)

        h = torch.cat(feat_list, dim = 1)

        return self.final_prediction(h)

class GIN(nn.Module):
    def __init__(self, params_dict):
        super(GIN, self).__init__()

        self.input_dim = params_dict['input_dim']
        self.input_channel = params_dict['input_channel']
        hid_dim = params_dict['hid_dim']
        output_dim = params_dict['output_dim']
        self.output_channel = params_dict['output_channel']
        self.num_hops = params_dict['num_hops']

        dim_list = [self.input_dim] + [hid_dim] * (self.num_hops - 1) + [output_dim]

        self.mlps = nn.ModuleList([MLP(in_dim = m, hid_dim = n, out_dim = n, num_layers = 2, final_nonlinear = True, reset_parameters = params_dict.get('reset_parameters', False)) for m, n in zip(dim_list[:-2], dim_list[1:-1])])
        self.mlps.append(MLP(in_dim = dim_list[-2], hid_dim = dim_list[-2], out_dim = dim_list[-1], num_layers = 2, final_nonlinear = False, reset_parameters = params_dict.get('reset_parameters', False)))

        self.gin_conv_layers = nn.ModuleList([GINConv(apply_func = mlp, aggregator_type = 'sum', init_eps = 0, learn_eps = True) for mlp in self.mlps])

    def forward(self, batched_graph):
        feat = batched_graph.ndata[self.input_channel]

        for gin_conv in self.gin_conv_layers:
            feat = gin_conv(batched_graph, feat)

        return feat

class GCN(nn.Module):
    def __init__(self, params_dict):
        super(GCN, self).__init__()

        self.input_dim = params_dict['input_dim']
        self.input_channel = params_dict['input_channel']
        hid_dim = params_dict['hid_dim']
        output_dim = params_dict['output_dim']
        self.output_channel = params_dict['output_channel']
        self.num_hops = params_dict['num_hops']
        
        dim_list = [self.input_dim] + [hid_dim] * (self.num_hops - 1) + [output_dim]

        self.linears = nn.ModuleList([nn.Linear(m,n) for m, n in zip(dim_list[:-1], dim_list[1:])])

    def forward(self, graph):
        degs = graph.out_degrees().float().clamp(min=1)
        graph.ndata['h'] = graph.ndata[self.input_channel]

        norm = torch.pow(degs, -0.5)
        norm = torch.reshape(norm, norm.shape + (1,) * (graph.ndata['feat'].dim() - 1)).to(device)

        for linear in self.linears[:-1]:
            graph.ndata['h'] = graph.ndata['h'] * norm

            graph.update_all(message_func = dfunc.copy_u('h','m'), reduce_func = dfunc.sum('m','h'))

            graph.ndata['h'] = graph.ndata['h'] * norm

            graph.ndata['h'] = F.relu(linear(graph.ndata['h']))

        return self.linears[-1](graph.ndata['h'])
        
class SWL_GNN(nn.Module):
    def __init__(self, params_dict):
        super(SWL_GNN, self).__init__()

        self.params_dict = params_dict
        self.input_dim = params_dict['input_dim']
        self.input_channel = params_dict['input_channel']
        hid_dim = params_dict['hid_dim']
        output_dim = params_dict['output_dim']
        self.output_channel = params_dict['output_channel']

        self.ops = params_dict['ops']

        if params_dict.get('last', False):
            self.first_layer_linear = nn.Linear(self.input_dim, hid_dim)
        else:
            self.first_layer_linear = nn.Linear(self.input_dim * sum([len(self.ops[op]) for op in self.ops]), hid_dim)
        self.second_layer_linear = nn.Linear(hid_dim, output_dim)

        if params_dict.get('precompute', False):
            self.augmented_feature = self.precompute(params_dict['batched_graph'], self.ops)
        else:
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.input_dim, track_running_stats=False) for i in range(sum([len(self.ops[op]) for op in self.ops]))])
            self.bns_used = 0

        if 'bethe' in self.ops:
            self.eps_list = nn.Parameter(torch.zeros(len(self.ops['bethe'])))

        if params_dict.get('reset_parameters', False):
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.first_layer_linear.weight, a=-1, b=1)
        nn.init.uniform_(self.first_layer_linear.bias, a=-1, b=1)
        nn.init.uniform_(self.second_layer_linear.weight, a=-1, b=1)
        nn.init.uniform_(self.second_layer_linear.bias, a=-1, b=1)

    def reset_parameters_0(self):
        # torch.nn.init.xavier_uniform_(self.layer.weight)
        # torch.nn.init.constant_(self.layer.bias, 0)
        torch.nn.init.constant_(self.first_layer_linear_list.weight, 0)
        torch.nn.init.constant_(self.first_layer_linear_list.bias, 0)

    def precompute(self, batched_graph, ops):
        augmented_feature = []

        for op in ops:
            if self.params_dict.get('last', False) and op == 'identity':
                continue
            if op == 'min_triangle':
                # min(A^2,1)
                triangle_graph = dgl.transform.to_simple_graph(dgl.transform.khop_graph(batched_graph, 2)).to(device)

                triangle_graph.ndata['feat'] = batched_graph.ndata['feat']
                a_f = self.calculate_operators_on_graph(triangle_graph, op, ops[op])
            
            elif op == 'motif_triangle':
                # motifnet-triangle, row-wise normalized
                hop2_graph = dgl.transform.khop_graph(batched_graph, 2)
                triangle_adj = normalize(batched_graph.adjacency_matrix_scipy().multiply(hop2_graph.adjacency_matrix_scipy()), norm='l1', axis=1)
                triangle_graph = dgl.DGLGraph()
                triangle_graph.from_scipy_sparse_matrix(triangle_adj)
                triangle_graph = triangle_graph.to(device)

                triangle_graph.ndata['feat'] = batched_graph.ndata['feat']
                a_f = self.calculate_operators_on_graph(triangle_graph, op, ops[op])
                
            else:
                a_f = self.calculate_operators_on_graph(batched_graph, op, ops[op])

            augmented_feature.extend(a_f)

        if self.params_dict.get('last', False):
            return augmented_feature[-1]
        else:
            return torch.cat(augmented_feature, dim = 1)

    def calculate_operators_on_graph(self, graph, op, hop_list):
        a_f = []

        # c: average degree
        c = torch.mean(graph.in_degrees().float()).item()

        graph.ndata['h'] = graph.ndata['feat'].to(torch.float).to(device)

        if op == 'identity':
            augmented_feature = [graph.ndata['h']]
            return augmented_feature

        if op == 'gcn':
            new_g = dgl.transform.add_self_loop(graph)
            new_g.ndata['feat'] = graph.ndata['h']
            new_g.ndata['h'] = graph.ndata['h']
            graph = new_g

        for i in range(max(hop_list)):
            if op == 'gcn':
                degs = graph.out_degrees().float().clamp(min=1)

                norm = torch.pow(degs, -0.5)
                norm = torch.reshape(norm, norm.shape + (1,) * (graph.ndata['feat'].dim() - 1)).to(device)
                graph.ndata['h'] = graph.ndata['h'] * norm

                graph.update_all(message_func = dfunc.copy_u('h','m'), reduce_func = dfunc.sum('m','h'))

                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = torch.reshape(norm, norm.shape + (1,) * (graph.ndata['feat'].dim() - 1)).to(device)

                graph.ndata['h'] = graph.ndata['h'] * norm

            elif op == 'gin' or op == 'min_triangle' or op == 'motif_triangle':
                graph.update_all(message_func = dfunc.copy_u('h','m'), reduce_func = dfunc.sum('m','h'))

            elif op == 'bethe':
                h_prev = graph.ndata['h']
                graph.update_all(message_func = dfunc.copy_u('h','m'), reduce_func = dfunc.sum('m','h'))
                diag_degree = torch.reshape(graph.out_degrees().float(), (graph.number_of_nodes(), 1)).to(device)
                H_pos_r = h_prev * (c - 1) - np.sqrt(c) * graph.ndata['h'] + h_prev * diag_degree
                H_neg_r = h_prev * (c - 1) + np.sqrt(c) * graph.ndata['h'] + h_prev * diag_degree

                graph.ndata['h'] = - H_pos_r + h_prev * 8

            else:
                print("Wrong operator: {}".format(op))
                assert False

            if not self.params_dict['precompute']:
                if i+1 in hop_list:
                    graph.ndata['h'] = self.bns[self.bns_used](graph.ndata['h'])
                    self.bns_used += 1

            a_f.append(graph.ndata['h'])

        augmented_feature = [a_f[i-1] for i in hop_list]

        return augmented_feature

    def forward(self, batched_graph):
        if self.params_dict['precompute']:

            augmented_feature = self.augmented_feature
            
        else:
            augmented_feature = self.precompute(batched_graph, self.ops)
            self.bns_used = 0
            
        y = self.first_layer_linear(augmented_feature)
        y = self.second_layer_linear(F.relu(y))

        return y
