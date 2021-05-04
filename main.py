import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl 
from dgl.transform import add_self_loop
from dgl.data import CoraDataset, PPIDataset, GINDataset, CitationGraphDataset
from dgl.data.utils import Subset
from dgl.nn.pytorch.conv import GINConv
import dgl.function as dfunc
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
from numpy.random import permutation
from tqdm import tqdm

from utils import Cora_Count_Path, RegularGraph_Count_Path

from models import *

import argparse

import sys

cls_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.MSELoss()
# reg_criterion = torch.nn.L1Loss()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def train_with_loader(model, loader, optimizer):
    model.train()
    epoch_loss = []

    for i, (batched_graph, labels) in tqdm(enumerate(loader)):
        prediction = model(batched_graph)
        loss = cls_criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.detach().item())

    return np.mean(epoch_loss)

def train_with_mask_batch(model, graph, mask, optimizer):
    model.train()

    n_train = int(torch.sum(mask).item())
    batch_size = 32
    random_perm = permutation(n_train)
    batch_mask = [torch.LongTensor(random_perm[i * batch_size: ((i+1) * batch_size) % n_train]) for i in range((n_train-1)//batch_size + 1)]

    loss_rec = []

    for b_m in batch_mask:
        prediction = model(graph)

        if args.dataset == 'cora_path' or args.dataset == 'regular':
            loss = reg_criterion(prediction[mask][b_m], graph.ndata['label'][mask][b_m].to(torch.float).to(device))
        else:
            loss = cls_criterion(prediction[mask], graph.ndata['label'][mask].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_rec.append(loss.detach().cpu().item())

    return np.mean(loss_rec)

def train_with_mask_full(model, graph, mask, optimizer):
    model.train()

    prediction = model(graph)

    if args.dataset == 'cora_path' or args.dataset == 'regular':
        loss = reg_criterion(prediction[mask], graph.ndata['label'][mask].to(torch.float).to(device))
    else:
        loss = cls_criterion(prediction[mask], graph.ndata['label'][mask].to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.detach().cpu().item()

@torch.no_grad()
def eval_with_loader(model, loader, total_num):
    model.eval()
    correct = 0

    for _, (batched_graph, labels) in tqdm(enumerate(loader)):
        prediction = model(batched_graph)
        prediction = prediction.max(1, keepdim = True)[1]
        # print(prediction)
        correct += prediction.eq(labels.view_as(prediction).to(device)).sum().cpu().item()

    return correct / total_num

@torch.no_grad()
def eval_with_mask_no_val(model, graph, train_mask, test_mask):
# def eval_with_mask(model, graph, train_mask, val_mask, test_mask):
    model.eval()

    if args.dataset == 'cora_path' or args.dataset == 'regular':
        prediction = model(graph)
        # train_prediction, val_prediction, test_prediction = prediction[train_mask], prediction[val_mask], prediction[test_mask]
        train_prediction, test_prediction = prediction[train_mask], prediction[test_mask]

        train_loss = reg_criterion(train_prediction, graph.ndata['label'][train_mask].to(torch.float).to(device))
        # val_loss = reg_criterion(val_prediction, graph.ndata['label'][val_mask].to(torch.float))
        test_loss = reg_criterion(test_prediction, graph.ndata['label'][test_mask].to(torch.float).to(device))

        # return train_loss.detach().item(), val_loss.detach().item(), test_loss.detach().item()
        return train_loss.detach().cpu().item(), test_loss.detach().cpu().item()

    else: 
        print("Wrong eval function!")
        assert False

@torch.no_grad()
def eval_with_mask_with_val(model, graph, train_mask, val_mask, test_mask):
    model.eval()

    prediction = model(graph).max(1, keepdim = True)[1]
    train_prediction, val_prediction, test_prediction = prediction[train_mask], prediction[val_mask], prediction[test_mask]

    train_correct = train_prediction.eq(graph.ndata['label'][train_mask].view_as(train_prediction).to(device)).sum().cpu().item()
    val_correct = val_prediction.eq(graph.ndata['label'][val_mask].view_as(val_prediction).to(device)).sum().cpu().item()
    test_correct = test_prediction.eq(graph.ndata['label'][test_mask].view_as(test_prediction).to(device)).sum().cpu().item()

    return train_correct/train_mask.sum().item(), val_correct/val_mask.sum().item(), test_correct/test_mask.sum().item()

def convert_params_dict(params_dict):
    ops = {}
    if params_dict['op_base'] == 'adj':
        ops['gin'] = list(range(1, 1 + params_dict['num_hops']))
    elif params_dict['op_base'] == 'laplacian':
        ops['gcn'] = list(range(1, 1 + params_dict['num_hops']))

    params_dict['ops'] = ops 
    params_dict['precompute'] = True

    return params_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--graph_level', action='store_true', default=False)
parser.add_argument('--self_loop', action='store_true', default=False)
parser.add_argument('--num_hops', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--op_base', type=str, default='laplacian')
parser.add_argument('--hid_dim', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--model', type=str, default='placeholder')
args = parser.parse_args()

# dataset and dataloader
if args.dataset.lower() in ['cora', 'cora_path', 'citeseer', 'pubmed', 'regular'] and not args.graph_level:
    if args.dataset.lower() == 'cora':
        dataset = CoraDataset()
    elif args.dataset.lower() == 'cora_path':
        dataset = Cora_Count_Path()
    elif args.dataset.lower() == 'regular':
        dataset = RegularGraph_Count_Path(1000, 6, length_path = 3)
    else:
        dataset = CitationGraphDataset(args.dataset.lower())

    train_mask = torch.BoolTensor(dataset.train_mask)
    # val_mask = torch.BoolTensor(dataset.val_mask)
    test_mask = torch.BoolTensor(dataset.test_mask)
    graph = dataset[0].to(device)
elif args.dataset.lower() in ['imdbbinary', 'imdbmulti', 'redditbinary', 'redditmulti5k' , 'collab'] and args.graph_level:
    dataset = GINDataset(args.dataset.upper(), self_loop = args.self_loop)
    random_permutation = list(permutation(len(dataset)))
    train_mask, test_mask = random_permutation[:int(0.9 * len(dataset))], random_permutation[int(0.9 * len(dataset)):]
    train_loader = DataLoader(Subset(dataset, train_mask), batch_size = 32, shuffle  = True, collate_fn = collate)
    test_loader = DataLoader(Subset(dataset, test_mask), batch_size = 32, shuffle  = False, collate_fn = collate)
else:
    print('Either dataset or task is wrong!', 'Dataset:', args.dataset.lower(), 'Graph-level:', args.graph_level)
    assert False

# model
if args.op_base not in ['adj', 'laplacian', 'chebyshev']:
    print("Wrong operator base!")
    assert False

if args.graph_level:
    params_dict = {'input_dim': 1,
                   'input_channel': 'attr',
                   'hid_dim': args.hid_dim,
                   'output_dim': args.hid_dim,
                   'num_classes': dataset.gclasses,
                   'output_channel': 'out',
                   'num_hops': args.num_hops,
                   'op_base': args.op_base
    }

    model = graph_task_wrapper(SWL_GNN, params_dict)
elif args.dataset == 'cora_path' or args.dataset == 'regular':
    params_dict = {'input_dim': dataset[0].ndata['feat'].size()[-1],
                   'input_channel': 'feat',
                   'hid_dim': args.hid_dim,
                   'output_dim': args.hid_dim,
                   'num_classes': dataset.labels.shape[-1],
                   'output_channel': 'out',
                   'num_hops': args.num_hops,
                   'op_base': args.op_base
    }

    if args.model.lower() == 'gin':
        model = node_task_wrapper(GIN, params_dict).to(device)
    elif args.model.lower() == 'gin_jk': 
        model = node_task_wrapper(GIN_JK, params_dict).to(device)
    elif args.model.lower() == 'swl_gnn':
        params_dict['batched_graph'] = graph.to(device)
        params_dict = convert_params_dict(params_dict)
        model = node_task_wrapper(SWL_GNN, params_dict).to(device)
    else:
        print("Undefined model!")
        assert False
else:
    params_dict = {'input_dim': dataset[0].ndata['feat'].size()[-1],
                   'input_channel': 'feat',
                   'hid_dim': args.hid_dim,
                   'output_dim': args.hid_dim,
                   'num_classes': graph.ndata['label'].max().item() + 1,
                   'output_channel': 'out',
                   'num_hops': args.num_hops,
                   'op_base': args.op_base
    }

    model = node_task_wrapper(SWL_GNN, params_dict).to(device)

# optimizer
params_dict['lr'] = args.lr
print(params_dict)
optimizer = optim.Adam(model.parameters(), lr = args.lr)

if args.dataset == 'regular' and args.model == 'gin_jk':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5000, gamma = 0.7)

# epochs

train_rec = []
val_rec = []
test_rec = []

for epoch in range(args.epochs):
    print("=====Epoch {}".format(epoch))
    print("Training...")

    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        train_loss = train_with_mask_full(model, graph, train_mask, optimizer)
    elif args.dataset.lower() in ['cora_path', 'regular']:
        train_loss = train_with_mask_batch(model, graph, train_mask, optimizer)
    else:
        train_loss = train_with_loader(model, train_loader, optimizer)

    if args.dataset == 'regular' and args.model == 'gin_jk':
        scheduler.step()

    print('Evluating...')

    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        train_acc, val_acc, test_acc = eval_with_mask_with_val(model, graph, train_mask, val_mask, test_mask)
        print("Loss: {} ===== Acc: {}, {}, {}".format(train_loss, train_acc, val_acc, test_acc))
        train_rec.append(train_acc)
        val_rec.append(val_acc)
        test_rec.append(test_acc)

    elif args.dataset.lower() in ['cora_path', 'regular']:
        # train_loss, val_loss, test_loss = eval_with_mask(model, graph, train_mask, val_mask, test_mask)
        # print("Loss: {} ===== Acc: {}, {}, {}".format(train_loss, train_loss, val_loss, test_loss))

        # train_loss, val_loss, test_loss = eval_with_mask(model, graph, train_mask, val_mask, test_mask)
        train_loss, test_loss = eval_with_mask_no_val(model, graph, train_mask, test_mask)
        print("Loss: {}, {}".format(train_loss, test_loss))

        train_rec.append(train_loss)
        # val_rec.append(val_loss)
        test_rec.append(test_loss)
    else:
        train_acc = eval_with_loader(model, train_loader, len(train_mask))
        test_acc = eval_with_loader(model, test_loader, len(test_mask))

        print("Loss: {} ===== Acc: {}, {}".format(train_loss, train_acc, test_acc))

        train_rec.append(train_acc)
        test_rec.append(test_acc)

if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
    print("Best Validation Perf: {} ===== Test perf: {} ===== Train Perf: {}".format(np.max(val_rec), test_rec[np.argmax(val_rec)], train_rec[np.argmax(val_rec)]))
elif args.dataset.lower() in ['cora_path', 'regular']:
    print("Best Train ===== Train Perf: {} ===== Test perf: {}".format(np.min(train_rec), test_rec[np.argmin(train_rec)]))
    print("Best Test  ===== Train Perf: {} ===== Test perf: {}".format(train_rec[np.argmin(test_rec)], np.min(test_rec)))
