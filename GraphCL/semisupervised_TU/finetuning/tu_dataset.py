from torch_geometric.datasets import TUDataset
import torch
from itertools import repeat, product
from copy import deepcopy


BIONIC_NAMES = ['Costanzo-2016', 'Hein-2015', 'Hu-2007', 'Huttlin-2015', 'Huttlin-2017', 'Krogan-2006', 'Rolland-2014', 'BIONIC']
BIONIC_LABEL_PATH = []
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
                 use_node_attr=True,
                 processed_filename='data.pt'):
        self.processed_filename = processed_filename
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform,
                                           pre_filter, False)

    @property
    def processed_file_names(self):
        return self.processed_filename

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

    def get(self, idx):
        data = super().get(idx)
        # do a quick and dirty fix for one graph:
        data.edge_index[data.edge_index >= data.num_nodes] = 1
        data.edge_index[data.edge_index <= 0] = 1
        return data