import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from ogb.nodeproppred import DglNodePropPredDataset

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()
    elif name == 'arxiv':
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']
    ogbgraph = ['arxiv']

    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
   
    if name in ogbgraph:
        splitted_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    
    #import ipdb;ipdb.set_trace()
    num_class = dataset.num_classes
    if name in ogbgraph:
        labels = graph[1].squeeze()
        feat = graph[0].ndata.pop('feat')
        graph = graph[0]
    else:
        feat = graph.ndata.pop('feat')
        labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx
