import dgl
from dgl.data import CoraDataset, PPIDataset, GINDataset
import dgl.function as dfunc
import torch
from collections import Counter

from utils import Cora_Count_Path, RegularGraph_Count_Path

def wl_dictionary(encoding):
    dictionary = {}

    rec = []
    for e in encoding:
        if e not in dictionary:
            dictionary[e] = len(dictionary)
        
        rec.append(dictionary[e])

    print(len(dictionary))
    return rec, len(dictionary)

def equivalence_class(encoding):
    dictionary = {}

    eq_cls = []
    for i, e in enumerate(encoding):
        if e not in dictionary:
            dictionary[e] = len(dictionary)
            eq_cls.append([])

        temp_list = eq_cls[dictionary[e]]
        temp_list.append(i)
        eq_cls[dictionary[e]] = temp_list
    
    return eq_cls

def wl_encoding(graph, rest_depth, encoding):
    if len(encoding) == 0:
        encoding = [0 for i in range(graph.number_of_nodes())]
    
    assert len(encoding) == graph.number_of_nodes()

    if rest_depth == 0:
        return encoding

    new_encoding = []

    for i in range(graph.number_of_nodes()):
        adj_nodes = graph.successors(i)

        this_encoding = []
        for an in adj_nodes:
            this_encoding.append(encoding[an])

        this_encoding.sort()

        new_encoding.append(str(encoding[i]) + str(this_encoding))
    
    new_encoding, _ = wl_dictionary(new_encoding)

    return wl_encoding(graph, rest_depth-1, new_encoding)

def linear_swl_encoding(graph, rest_depth, encoding):
    # 
    # use ndata['h'] to store node hidden states
    # 

    if not 'h' in graph.ndata:
        graph.ndata['h'] = torch.ones(graph.number_of_nodes())
    
    if len(encoding) == 0:
        encoding = [[0] for i in range(graph.number_of_nodes())]

    if rest_depth == 0:
        encoding = [str(ee) for ee in encoding]
        encoding, _ = wl_dictionary(encoding)
        return encoding

    graph.update_all(message_func = dfunc.copy_u('h','m'), reduce_func = dfunc.sum('m','h'))
    # graph.update_all(message_func = dfunc.copy_u('h','m'), reduce_func = dfunc.mean('m','h'))

    new_encoding = [str(hh.tolist()) for hh in graph.ndata['h']]
    new_encoding, _ = wl_dictionary(new_encoding)

    temp_encoding = []
    for ee, nee in zip(encoding, new_encoding):
        ee.append(nee)
        temp_encoding.append(ee)
    
    encoding = temp_encoding

    return linear_swl_encoding(graph, rest_depth - 1, encoding)

def wl_with_node_feature(graph, total_depth, feature_channel):
    feat = graph.ndata[feature_channel]
    encoding = [str(ff.tolist()) for ff in feat]
    encoding, _ = wl_dictionary(encoding)
    return wl_encoding(graph, total_depth, encoding)

def wl_without_node_feature(graph, total_depth):
    return wl_encoding(graph, total_depth, [])

def linear_swl_with_node_feature(graph, total_depth, feature_channel):
    feat = graph.ndata[feature_channel]
    encoding = [str(ff.tolist()) for ff in feat]
    encoding, num_diff_feats = wl_dictionary(encoding)
    feat = torch.zeros((graph.number_of_nodes(), num_diff_feats))

    for i, ee in enumerate(encoding):
        feat[i][ee] = 1

    graph.ndata['h'] = feat
    encoding = [[ee] for ee in encoding]
    return linear_swl_encoding(graph, total_depth, encoding)

def linear_swl_without_node_feature(graph, total_depth):
    return linear_swl_encoding(graph, total_depth, [])

def eq_cls_classification_error(eq_cls, labels):
    sum_correct = 0
    
    for e_c in eq_cls:
        ll = [labels[eecc] for eecc in e_c]
        counter = Counter(ll)
        most_common_element, most_common_frequency = counter.most_common(1)[0]
        sum_correct += most_common_frequency
    
    return float(sum_correct) / len(labels)

def node_to_graph_encoding(batched_graph, node_encoding):
    each_graph_num_nodes = batched_graph.batch_num_nodes

    cur = 0
    graphs_encoding = []
    for egnn in each_graph_num_nodes:
        ng_encoding = [node_encoding[cur + ee] for ee in range(egnn)]
        cur += egnn
        ng_encoding.sort()
        graphs_encoding.append(str(ng_encoding))
    
    graphs_encoding, _ = wl_dictionary(graphs_encoding)

    return graphs_encoding

def cora():
    # data = CoraDataset()
    data = Cora_Count_Path()
    graph = data[0]
    feature_channel = 'feat'
    # final_encoding = wl_with_node_feature(graph, 2, feature_channel)
    # final_encoding = wl_without_node_feature(graph, 4)
    final_encoding = linear_swl_with_node_feature(graph, 2, feature_channel)
    # final_encoding = linear_swl_without_node_feature(graph, 4)
    
    eq_cls = equivalence_class(final_encoding)
    labels = graph.ndata['label'].tolist()

    print(eq_cls_classification_error(eq_cls, labels))

def regular_graph():
    # data = CoraDataset()
    data = RegularGraph_Count_Path(1000, 6)
    graph = data[0]
    feature_channel = 'feat'
    final_encoding = wl_with_node_feature(graph, 2, feature_channel)
    # final_encoding = wl_without_node_feature(graph, 4)
    final_encoding = linear_swl_with_node_feature(graph, 2, feature_channel)
    # final_encoding = linear_swl_without_node_feature(graph, 4)
    
    eq_cls = equivalence_class(final_encoding)
    labels = graph.ndata['label'].tolist()

    print(eq_cls_classification_error(eq_cls, labels))

def gin_reddit():
    # data = GINDataset('REDDITBINARY', self_loop=False)
    data = GINDataset('REDDITMULTI5K', self_loop=False)

    graphs = data.graphs
    batched_graph = dgl.batch(graphs)
    
    # node_encoding = wl_without_node_feature(big_graph, 4)
    node_encoding = linear_swl_without_node_feature(batched_graph, 3)
    graphs_encoding = node_to_graph_encoding(batched_graph, node_encoding)
    eq_cls = equivalence_class(graphs_encoding) 
    print(eq_cls_classification_error(eq_cls, data.labels))
    exit()

if __name__ == "__main__":
    # gin_reddit()
    # cora()
    regular_graph()