import torch_geometric.utils
from torch_geometric.datasets import TUDataset
import torch
from itertools import repeat, product
import numpy as np
import networkx as nx

BIONIC_NAMES = ['Costanzo-2016', 'Hein-2015', 'Hu-2007', 'Huttlin-2015', 'Huttlin-2017', 'Krogan-2006', 'Rolland-2014', 'BIONIC']

class TUDatasetExt(TUDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 processed_filename='data.pt',
                 aug="none", aug_ratio=None, mcl_iters=None):
        self.processed_filename = processed_filename
        self.aug = "none"
        self.aug_ratio = None
        self.mcl_iters = None
        self.name = name
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform,
                                           pre_filter, use_node_attr)

    @property
    def raw_file_names(self):
        if self.name in BIONIC_NAMES:
            return [f'{self.name}.txt']
        else:
            return super(TUDatasetExt, self).raw_file_names

    def download(self):
        if self.name in BIONIC_NAMES:
            return
        else:
            return super(TUDatasetExt, self).download()

    @property
    def processed_file_names(self):
        return self.processed_filename

    def get(self, idx):
        data = self.data.__class__()


        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        # do a quick and dirty fix for one graph:
        data.edge_index[data.edge_index >= data.num_nodes] = 1
        data.edge_index[data.edge_index <= 0] = 1

        if self.aug == 'dropN':
            data = drop_nodes(data, self.aug_ratio)
        elif self.aug == 'wdropN':
            data = weighted_drop_nodes(data, self.aug_ratio, self.npower)
        elif self.aug == 'permE':
            data = permute_edges(data, self.aug_ratio)
        elif self.aug == 'mcl':
            data = mcl_aug(data, self.aug_ratio, self.mcl_iters)
        elif self.aug == 'subgraph':
            data = subgraph(data, self.aug_ratio)
        elif self.aug == 'maskN':
            data = mask_nodes(data, self.aug_ratio)
        elif self.aug == 'none':
            data = data
        elif self.aug == 'random4':
            ri = np.random.randint(4)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = permute_edges(data, self.aug_ratio)
            elif ri == 3:
                data = mask_nodes(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False
        elif self.aug == 'random3':
            ri = np.random.randint(3)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = permute_edges(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False
        elif self.aug == 'random2':
            ri = np.random.randint(2)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False
        else:
            print('augmentation error')
            assert False

        # print(data)
        # print(self.aug)
        # assert False

        return data


def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))

    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


def weighted_drop_nodes(data, aug_ratio, npower):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg==0] = 0.1
    # print(deg)
    # deg = deg ** (-1)
    deg = deg ** (npower)
    # print(deg)
    # print(deg / deg.sum())
    # assert False

    idx_drop = np.random.choice(node_num, drop_num, replace=False, p=deg / deg.sum())

    # idx_perm = np.random.permutation(node_num)
    # idx_drop = idx_perm[:drop_num]
    # idx_nondrop = idx_perm[drop_num:]

    idx_nondrop = np.array([n for n in range(node_num) if not n in idx_drop])

    # idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ###
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    ###
    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data

from torch_geometric.utils.subgraph import subgraph as geo_subgraph
import torch_geometric.data
from mcl_augs import preproc_graph, MCL_raw

def mcl_aug(data, aug_ratio, mcl_iters=None): # TODO - ratio
    if mcl_iters is None:
        mcl_iters = 10
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    affected_num = int(node_num * aug_ratio)
    subset = torch.tensor(np.random.choice(node_num, affected_num, replace=False), dtype=torch.long)
    sub_edge_index, sub_edge_attr, sub_edge_mask = geo_subgraph(subset, data.edge_index, data.edge_attr, return_edge_mask=True)
    sub_graph = data.clone()
    sub_graph.x = sub_graph.x[subset]
    sub_graph.edge_index = data.edge_index[..., sub_edge_mask]
    if sub_graph.edge_attr is not None:
        sub_graph.edge_attr = sub_graph.edge_attr[sub_edge_mask]

    nx_data = torch_geometric.utils.to_networkx(sub_graph)
    preproced = preproc_graph(nx_data)

    if len(preproced):
        mcl_post = MCL_raw(preproced, nodes=None, r=2, steps=mcl_iters)
        inds = mcl_post.data<=0.5
        mcl_post.data[inds] = 0
        mcl_post.data[~inds] = 1
        mcl_post_nonsparse = nx.from_scipy_sparse_array(mcl_post)
        geo_out = torch_geometric.utils.from_networkx(mcl_post_nonsparse)
    else:
        geo_out = torch_geometric.utils.from_networkx(preproced)
    geo_out_final = data.clone()
    out_edge_index_after_mapping_to_original_inds = (subset[geo_out.edge_index[0]], subset[geo_out.edge_index[1]])
    geo_out_edge_index = torch.vstack(out_edge_index_after_mapping_to_original_inds)
    non_subgraph_edges = data.edge_index[..., ~sub_edge_mask]
    final_edge_index = torch.hstack((non_subgraph_edges, geo_out_edge_index))
    geo_out_final.edge_index = final_edge_index
    return geo_out_final


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data

def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[list(range(node_num)), list(range(node_num))] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    data.edge_index = edge_index

    return data


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data





