from __future__ import division
import time

import argparse
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# from dgl.data import SBMMixtureDataset
from utils import SBM_dataset

from models import *

from utils import get_config_from_json, process_config, augment_params_list, initialize_model

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model config', default='swl_gnn')
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--n-communities', type=int, help='Number of communities', default=5)
parser.add_argument('--n-epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--hid_dim', type=int, help='Number of hidden dimensions', default=32)
parser.add_argument('--n-graphs', type=int, help='Number of graphs', default=400)
parser.add_argument('--n-layers', type=int, help='Number of layers', default=30)
parser.add_argument('--n-nodes', type=int, help='Number of nodes', default=400)
parser.add_argument('--radius', type=int, help='Radius', default=3)
parser.add_argument('--p', type=float, default=0)
parser.add_argument('--q', type=float, default=18)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--clip_grad_norm', type=float, default=40.0)
args = parser.parse_args()

dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
K = args.n_communities

train_dataset = SBM_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes, n_communities=args.n_communities, p=args.p, q=args.q)
test_dataset = SBM_dataset(n_graphs=200, n_nodes=args.n_nodes, n_communities=args.n_communities, p=args.p, q=args.q)

# print(train_dataset.output_overlap())
# print(test_dataset.output_overlap())

ones = torch.ones(args.n_nodes // K)
y_list = [torch.cat([x * ones for x in p]).long().to(dev) for p in permutations(range(K))]

config = process_config(args.model)

params_dict = {
    'model': config.model,
    'input_dim': 1,
    'input_channel': config.input_channel,
    'hid_dim': config.hid_dim,
    'output_dim': config.hid_dim,
    'num_classes': args.n_communities,
    'output_channel': config.output_channel,
    'num_hops': config.num_hops, # for models other than swl-gnn
    "nhop_gcn": config.nhop_gcn, # for swl
    "nhop_gin": config.nhop_gin, # for swl
    "nhop_min_triangle": config.nhop_min_triangle, # for swl
    "nhop_motif_triangle": config.nhop_motif_triangle, # for swl
    "stack_op": config.stack_op, # for swl
    "no_identity": config.no_identity, # for swl-gnn
    "role": "t",
    "nhop_bethe": config.nhop_bethe,
    "last": False
}

params_dict = augment_params_list(params_dict)

params_dict['precompute'] = False
# model = node_task_wrapper(SWL_GNN, params_dict).to(dev)
model = initialize_model(params_dict)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def compute_overlap(z_list):
    ybar_list = [torch.max(z, 1)[1] for z in z_list]
    overlap_list = []
    for y_bar in ybar_list:
        accuracy = max(torch.sum(y_bar == y).item() for y in y_list) / args.n_nodes
        overlap = (accuracy - 1 / K) / (1 - 1 / K)
        overlap_list.append(overlap)
    return sum(overlap_list) / len(overlap_list)

def from_np(f, *args):
    def wrap(*args):
        new = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args]
        return f(*new)
    return wrap

def train(model, graph):
    """ One step of training. """
    model.train()
    z = model(graph.to(dev))

    z_list = torch.chunk(z, args.batch_size, 0)
    loss = sum(min(F.cross_entropy(z, y) for y in y_list) for z in z_list) / args.batch_size
    overlap = compute_overlap(z_list)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optimizer.step()

    return loss.detach().cpu().item(), overlap

def eval(model, dataset):
    """ One step of training. """
    model.eval()

    overlap_list = []

    for graph in dataset.graphs:
        z = model(graph.to(dev))

        z_list = torch.chunk(z, args.batch_size, 0)
        overlap_list.append(compute_overlap(z_list))

    return np.mean(overlap_list), np.std(overlap_list)

@from_np
def inference(g, lg, deg_g, deg_lg, pm_pd):
    g = g.to(dev)
    lg = lg.to(dev)
    deg_g = deg_g.to(dev).unsqueeze(1)
    deg_lg = deg_lg.to(dev).unsqueeze(1)
    pm_pd = pm_pd.to(dev)

    z = model(g, lg, deg_g, deg_lg, pm_pd)

    return z
def test():
    p_list =[6, 5.5, 5, 4.5, 1.5, 1, 0.5, 0]
    q_list =[0, 0.5, 1, 1.5, 4.5, 5, 5.5, 6]
    N = 1
    overlap_list = []
    for p, q in zip(p_list, q_list):
        dataset = SBMMixtureDataset(N, args.n_nodes, K, pq=[[p, q]] * N)
        loader = DataLoader(dataset, N, collate_fn=dataset.collate_fn)
        g, lg, deg_g, deg_lg, pm_pd = next(iter(loader))
        z = inference(g, lg, deg_g, deg_lg, pm_pd)
        overlap_list.append(compute_overlap(torch.chunk(z, N, 0)))
    return overlap_list

n_iterations = args.n_graphs // args.batch_size
for i in range(args.n_epochs):
    total_loss, total_overlap = [], []
    for j, graph in enumerate(train_dataset.graphs):
        loss, overlap = train(model, graph)
        total_loss.append(loss)
        total_overlap.append(overlap)

        epoch = '0' * (len(str(args.n_epochs)) - len(str(i)))
        iteration = '0' * (len(str(n_iterations)) - len(str(j)))
        if args.verbose:
            if (j+1) % 50 == 0:
                print('[epoch %s%d iteration %s%d]loss %.3f | overlap %.3f'
                    % (epoch, i, iteration, j, np.mean(total_loss[-50:]), np.mean(total_overlap[-50:])))

    epoch = '0' * (len(str(args.n_epochs)) - len(str(i)))
    loss = np.mean(total_loss)
    overlap = np.mean(total_overlap)
    print('[epoch %s%d]loss %.3f | overlap %.3f'
          % (epoch, i, loss, overlap))

    overlap_mean, overlap_std = eval(model, test_dataset)
    print('[epoch %s%d] | test mean overlap %.3f +- %.3f'
        % (epoch, i, overlap_mean, overlap_std))


    # overlap_list = test(model, test_dataset)
    # overlap_str = ' - '.join(['%.3f' % overlap for overlap in overlap_list])
    # print('[epoch %s%d]overlap: %s' % (epoch, i, overlap_str))
