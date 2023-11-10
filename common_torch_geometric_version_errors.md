## Common torch_geometric version failures
torch_geometric has multiple versions with a lot of breaking changes. The GrapCL repo that we use in our project uses
multiple datasets, each one saved in a different torch_geometric version which causes errors. \
Also, it is increasingly difficult to maintain an old conda environment with these old versions and make it work. \
Thus, the solution achieved was using an up to date envinroment and changing the code accordingly to make it compatible.

Below is a list of common errors that we get for using a wrong torch_geometric version and how to solve them

### super(FeatureExpander, self).__init__('add', 'source_to_target') TypeError: __init__() takes from 1 to 2 positional arguments but 3 were given
Change the FeatureExpander __init__ from `super(FeatureExpander, self).__init__('add', 'source_to_target')`
to `super(FeatureExpander, self).__init__('add', flow='source_to_target')`

### RuntimeError: expand(torch.cuda.LongTensor{[2, 3584]}, size=[3584]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)
Change in GCNConv
from `        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)`
to `edge_index, edge_attr = add_self_loops(edge_index, num_nodes=num_nodes)`

### ValueError: Found input variables with inconsistent numbers of samples: [3699, 4110]
from `for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset[idx_train].data.y):`
to `for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):`