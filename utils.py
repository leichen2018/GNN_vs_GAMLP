import numpy as np
import random
import json
import sys
import os
import scipy as sp

import networkx as nx
from networkx.readwrite import json_graph

import torch
import dgl
from dgl.data import CoraDataset
from dgl.data.utils import Subset

import scipy.sparse.linalg as SLA
import numpy.linalg as LA

from easydict import EasyDict
import datetime

from models import *

TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

# version_info = list(map(int, nx.__version__.split('.')))
# major = version_info[0]
# minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    # Remove all nodes that do not have val/test annotations
    # (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        # train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])    # network 1.11
        train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]) # network 2.3
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

class RegularGraph_Count_Path(object):
    def __init__(self, num_nodes, degree, length_path = 3):
        self.num_nodes = num_nodes
        self.degree = degree
        self.length_path = length_path

        self.generate_graph()

        self.init_node_feat()

        self.count_path()
        
    def generate_graph(self):
        random.seed(0)
        np.random.seed(0)

        graph = nx.generators.random_graphs.random_regular_graph(self.degree, self.num_nodes)

        self.graph = dgl.DGLGraph()
        self.graph.from_networkx(graph)

    def init_node_feat(self):
        self.graph = dgl.DGLGraph(self.graph)

        self.features = torch.FloatTensor([[1,0] if i%2 == 0 else [0,1] for i in range(self.graph.number_of_nodes())])

        self.graph.ndata['feat'] = self.features

        num_train, num_test = int(0.3 * self.num_nodes), self.num_nodes - int(0.3 * self.num_nodes)

        self.train_mask = np.array([1] * num_train + [0] * num_test)
        self.test_mask = np.array([0] * num_train + [1] * num_test)

    def count_path(self):
        store_prev_path = []
        store_cur_path = []

        for i in range(self.graph.number_of_nodes()):
            this_node_dict = {}
            if self.graph.nodes[i].data['feat'][0][0].item() == 1:
                this_node_dict['0'] = 1
            else:
                this_node_dict['1'] = 1

            store_prev_path.append(this_node_dict)

        for _ in range(self.length_path):
            for j in range(self.graph.number_of_nodes()):
                predecessors = [int(p.item()) for p in self.graph.predecessors(j)]
                this_node_dict = {}

                if self.graph.nodes[j].data['feat'][0][0].item() == 1:
                    prefix = '0<-'
                else:
                    prefix = '1<-'

                for p in predecessors:
                    for path in store_prev_path[p]:
                        this_node_dict[prefix + path] = this_node_dict.get(prefix + path, 0) + store_prev_path[p][path]

                store_cur_path.append(this_node_dict)

            store_prev_path = store_cur_path
            store_cur_path = []

        path_to_class_dict = {}

        for i in range(self.graph.number_of_nodes()):
            for path in store_prev_path[i]:
                if path not in path_to_class_dict:
                    path_to_class_dict[path] = len(path_to_class_dict)

        labels = np.zeros((self.graph.number_of_nodes(), len(path_to_class_dict)))

        for i in range(self.graph.number_of_nodes()):
            for path in store_prev_path[i]:
                labels[i, path_to_class_dict[path]] = store_prev_path[i][path]

        self.labels = labels[:,:1]
        print(path_to_class_dict)
        # self.labels = labels
        self.graph.ndata['label'] = torch.FloatTensor(self.labels)

        print("Label mean: {} ===== Label std: {} ===== #label: {}".format(np.mean(self.labels), np.std(self.labels), self.labels.shape[-1]))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = dgl.DGLGraph(self.graph)
        g.ndata['train_mask'] = self.train_mask
        # g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        return g


class Cora_Count_Path(CoraDataset):
    def __init__(self):
        super(Cora_Count_Path, self).__init__()

        self.init_node_feat()

        self.count_path()

    def init_node_feat(self):
        self.graph = dgl.DGLGraph(self.graph)

        self.features = torch.FloatTensor([[1,0] if i%2 == 0 else [0,1] for i in range(self.graph.number_of_nodes())])

        self.graph.ndata['feat'] = self.features

        random.seed(0)
        np.random.seed(0)

        # self.train_mask = np.array([1] * 1000 + [0] * 500 + [0] * 1000 + [0] * (self.graph.number_of_nodes() - 2500))
        # self.val_mask = np.array([0] * 1000 + [1] * 500 + [0] * 1000 + [0] * (self.graph.number_of_nodes() - 2500))
        # self.test_mask = np.array([0] * 1000 + [0] * 500 + [1] * 1000 + [0] * (self.graph.number_of_nodes() - 2500))

        random_perm_even = np.random.permutation(self.graph.number_of_nodes()//2)
        random_perm_odd = np.random.permutation(self.graph.number_of_nodes()//2)
        train_set = np.concatenate([random_perm_even[:500] * 2, random_perm_odd[:500] * 2 + 1])
        test_set = np.concatenate([random_perm_even[500:] * 2, random_perm_odd[500:] * 2 + 1])
        train_mask = np.zeros((self.graph.number_of_nodes(),))
        test_mask = np.zeros((self.graph.number_of_nodes(),))
        train_mask[train_set] = 1
        test_mask[test_set] = 1
        self.train_mask = train_mask
        self.test_mask = test_mask

    def count_path(self):
        store_prev_path = []
        store_cur_path = []

        for i in range(self.graph.number_of_nodes()):
            this_node_dict = {}
            if self.graph.nodes[i].data['feat'][0][0].item() == 1:
                this_node_dict['0'] = 1
            else:
                this_node_dict['1'] = 1

            store_prev_path.append(this_node_dict)

        for _ in range(2):
            for j in range(self.graph.number_of_nodes()):
                predecessors = [int(p.item()) for p in self.graph.predecessors(j)]
                this_node_dict = {}

                if self.graph.nodes[j].data['feat'][0][0].item() == 1:
                    prefix = '0<-'
                else:
                    prefix = '1<-'

                for p in predecessors:
                    for path in store_prev_path[p]:
                        this_node_dict[prefix + path] = this_node_dict.get(prefix + path, 0) + store_prev_path[p][path]

                store_cur_path.append(this_node_dict)

            store_prev_path = store_cur_path
            store_cur_path = []

        path_to_class_dict = {}

        for i in range(self.graph.number_of_nodes()):
            for path in store_prev_path[i]:
                if path not in path_to_class_dict:
                    path_to_class_dict[path] = len(path_to_class_dict)

        labels = np.zeros((self.graph.number_of_nodes(), len(path_to_class_dict)))

        for i in range(self.graph.number_of_nodes()):
            for path in store_prev_path[i]:
                labels[i, path_to_class_dict[path]] = store_prev_path[i][path]

        self.labels = labels[:,:1]
        print(path_to_class_dict)
        self.graph.ndata['label'] = torch.FloatTensor(self.labels)

        print("Label mean: {} ===== Label std: {} ===== #label: {}".format(np.mean(self.labels), np.std(self.labels), self.labels.shape[-1]))

def sbm(n_blocks, block_size, p, q, rng=None):
    """ (Symmetric) Stochastic Block Model
    Parameters
    ----------
    n_blocks : int
        Number of blocks.
    block_size : int
        Block size.
    p : float
        Probability for intra-community edge.
    q : float
        Probability for inter-community edge.
    rng : numpy.random.RandomState, optional
        Random number generator.
    Returns
    -------
    scipy sparse matrix
        The adjacency matrix of generated graph.
    """
    n = n_blocks * block_size
    p /= n
    q /= n
    rng = np.random.RandomState() if rng is None else rng

    rows = []
    cols = []
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            density = p if i == j else q
            block = sp.sparse.random(block_size, block_size, density,
                                     random_state=rng, data_rvs=lambda n: np.ones(n))
            rows.append(block.row + i * block_size)
            cols.append(block.col + j * block_size)

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    a = sp.sparse.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    adj = sp.sparse.triu(a) + sp.sparse.triu(a, 1).transpose()
    
    # adj_dense = adj.todense()
    # H = 2 * np.eye(250) - np.sqrt(3) * adj_dense + np.diag(np.sum(np.array(adj_dense), axis=1))
    # pred = np.array(np.squeeze(np.sign(LA.eigh(H)[1][:,1])))[0]
    # label1 = np.concatenate([np.ones(125), np.ones(125) * -1])
    # label2 = -label1
    # overlap = max(sum(pred==label1), sum(pred==label2))
    # overlap = (overlap/250-1/2)*2

    # return adj, overlap
    return adj

class SBM_dataset(object):
    def __init__(self,
                 n_graphs=300,
                 n_nodes=1000,
                 n_communities=5,
                 p=0,
                 q=18,
                 rng=None):

        self.n_graphs = n_graphs
        self.n_nodes = n_nodes 
        self.n_communities = n_communities
        self.p = p
        self.q = q

        self.rng = rng

        self.generate_graphs()


    def generate_graphs(self):
        self.graphs = []
        # self.overlap = []
        
        for i in range(self.n_graphs):
            g = dgl.DGLGraph()
            # adj, overlap = sbm(self.n_communities, self.n_nodes // self.n_communities, self.p, self.q)
            adj = sbm(self.n_communities, self.n_nodes // self.n_communities, self.p, self.q)
            g.from_scipy_sparse_matrix(adj)
            g.ndata['feat'] = torch.ones((self.n_nodes, 1))
            # g.ndata['feat'] = g.out_degrees().float().clamp(min=1).unsqueeze(1)
            self.graphs.append(g)
            # self.overlap.append(overlap)

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.graphs[idx]
        elif isinstance(idx, list):
            return [self.graphs[i] for i in idx]
        else:
            print("Wrong idx.")
            assert False

    # def output_overlap(self):
    #     return np.mean(self.overlap), np.std(self.overlap)

class ErdosRenyi_Random_Graph(object):
    def __init__(self, num_graphs, num_nodes, p, seed = 0, random_feature = False):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.p = p
        self.seed = seed

        self.generate_graph()

        if not random_feature:
            self.init_node_feat()
        else:
            self.init_node_feat_random()
        
    def generate_graph(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        if isinstance(self.num_nodes, list):
            num_nodes_list = self.num_nodes
        else:
            num_nodes_list = [self.num_nodes]

        if isinstance(self.p, list):
            p_list = self.p
        else:
            p_list = [self.p]

        iterator_num_node_vs_p = zip(random.choices(num_nodes_list, k = self.num_graphs), random.choices(p_list, k = self.num_graphs))

        self.graphs = []
        for nn, pp in iterator_num_node_vs_p:
            graph = nx.generators.erdos_renyi_graph(nn, pp)
            graph_dgl = dgl.DGLGraph()
            graph_dgl.from_networkx(graph)
            self.graphs.append(graph_dgl)

    def init_node_feat(self):
        for i in range(len(self.graphs)):
            self.graphs[i].ndata['feat'] = torch.ones((self.graphs[i].number_of_nodes(), 1))

    def init_node_feat_random(self):
        for i in range(len(self.graphs)):
            # self.graphs[i].ndata['feat'] = torch.ones((self.graphs[i].number_of_nodes(), 1))
            self.graphs[i].ndata['feat'] = torch.randn(self.graphs[i].number_of_nodes(), 2)

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

def collate_graphs(samples):
    # graphs = map(list, zip(*samples))
    batched_graph = dgl.batch(samples)

    return batched_graph



def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    if json_file.startswith('gin_jk'):
        folder = 'gin_jk'
    elif json_file.startswith('gin'):
        folder = 'gin'
    elif json_file.startswith('gcn'):
        folder = 'gcn'
    elif json_file.startswith('swl_gnn'):
        folder = 'swl_gnn'
    else:
        print("Wrong config name!")
        assert False
    
    with open('configs/' + folder + '/' + json_file + '.json', 'r') as config_file:
        config_dict = json.load(config_file)

    if "nhop_bethe" not in config_dict:
        config_dict["nhop_bethe"] = 0

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config

def process_config(json_file):
    config = get_config_from_json(json_file)
    config.timestamp = TIME

    return config

def augment_params_list(params):
    if params['model'] != 'swl_gnn':
        return params

    dict_ops = {}

    no_identity, nhop_gcn, nhop_gin, nhop_min_triangle, nhop_motif_triangle, stack_op = params['no_identity'], params['nhop_gcn'], params['nhop_gin'], params['nhop_min_triangle'], params['nhop_motif_triangle'], params['stack_op']

    nhop_bethe = params['nhop_bethe']
    
    if not no_identity:
        dict_ops['identity'] = [1]
    
    if nhop_gcn > 0:
        if stack_op:
            dict_ops['gcn'] = list(range(1, nhop_gcn + 1))
        else:
            dict_ops['gcn'] = [nhop_gcn]

    if nhop_gin > 0:
        if stack_op:
            dict_ops['gin'] = list(range(1,nhop_gin + 1))
        else:
            dict_ops['gin'] = [nhop_gin]
    
    # if nhop_chebyshev > 0:
    #     if stack_op:
    #         dict_ops['chebyshev'] = list(range(1,nhop_chebyshev + 1))
    #     else:
    #         dict_ops['chebyshev'] = [nhop_chebyshev]

    if nhop_min_triangle > 0:
        if stack_op:
            dict_ops['min_triangle'] = list(range(1,nhop_min_triangle + 1))
        else:  
            dict_ops['min_triangle'] = [nhop_min_triangle]

    if nhop_motif_triangle > 0:
        if stack_op:
            dict_ops['motif_triangle'] = list(range(1,nhop_motif_triangle + 1))
        else:
            dict_ops['motif_triangle'] = [nhop_min_triangle]

    if nhop_bethe > 0:
        if stack_op:
            dict_ops['bethe'] = list(range(1,nhop_bethe + 1))
        else:
            dict_ops['bethe'] = [nhop_bethe]

    params['ops'] = dict_ops

    print("SWL-GNN operators: {}".format(dict_ops))

    return params

def initialize_model(params_dict, graph = None):
    if params_dict['model'].lower() == 'gin':
        model = node_task_wrapper(GIN, params_dict).to(device)
    elif params_dict['model'].lower() == 'gcn':
        model = node_task_wrapper(GCN, params_dict).to(device)
    elif params_dict['model'].lower() == 'gin_jk':
        model = node_task_wrapper(GIN_JK, params_dict).to(device)
    elif params_dict['model'].lower() == 'swl_gnn':
        params_dict['batched_graph'] = graph
        model = node_task_wrapper(SWL_GNN, params_dict).to(device)
    else:
        print("Undefined model!")
        assert False

    return model

