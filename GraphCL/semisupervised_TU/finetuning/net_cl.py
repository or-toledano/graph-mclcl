from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool
from gcn_conv import GCNConv


class BatchNorm1dSafe(torch.nn.Module):
    def __init__(self, dim):
        super(BatchNorm1dSafe, self).__init__()
        self.bn = BatchNorm1d(dim)

    def forward(self, x):
        if x.shape[0] == 1:
            return x
        else:
            return self.bn(x)


class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1dSafe(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1dSafe(hidden)
            self.lin2_xg = Linear(hidden, hidden)
        else:
            self.use_xg = False

        hidden_in = dataset.num_features
        self.num_node_labels = dataset.num_node_labels
        hidden_in -= self.num_node_labels
        if collapse:
            self.bn_feat = BatchNorm1dSafe(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_in, hidden_in),
                    torch.nn.ReLU(),
                    Linear(hidden_in, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1dSafe(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, dataset.num_classes)
        else:
            self.bn_feat = BatchNorm1dSafe(hidden_in)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = GCNConv(hidden_in, hidden, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            if self.res_branch == "resnet":
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1dSafe(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
                    self.bns_conv.append(BatchNorm1dSafe(hidden))
                    self.convs.append(GConv(hidden, hidden))
                    self.bns_conv.append(BatchNorm1dSafe(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
            else:
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1dSafe(hidden))
                    self.convs.append(GConv(hidden, hidden))
            self.bn_hidden = BatchNorm1dSafe(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1dSafe(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, dataset.num_classes)

            self.bn_hidden_nodes = BatchNorm1dSafe(hidden)
            self.bns_fc_nodes = torch.nn.ModuleList()
            self.lins_nodes = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc_nodes.append(BatchNorm1dSafe(hidden))
                self.lins_nodes.append(Linear(hidden, hidden))
            self.lin_class_nodes = torch.nn.Sequential(Linear(hidden, hidden), torch.nn.ReLU(), Linear(hidden, self.num_node_labels - 1))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (BatchNorm1dSafe)):
                torch.nn.init.constant_(m.bn.weight, 1)
                torch.nn.init.constant_(m.bn.bias, 0.0001)

        self.proj_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x[:, self.num_node_labels:]
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.collapse:
            return self.forward_collapse(x, edge_index, batch, xg)
        elif self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        elif self.res_branch == "BNReLUConv":
            return self.forward_BNReLUConv(x, edge_index, batch, xg)
        elif self.res_branch == "ConvReLUBN":
            return self.forward_ConvReLUBN(x, edge_index, batch, xg)
        elif self.res_branch == "resnet":
            return self.forward_resnet(x, edge_index, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_collapse(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    ### This one!!!
    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        with torch.no_grad():
            x = self.bn_feat(x)
            x = F.relu(self.conv_feat(x, edge_index))
            for i, conv in enumerate(self.convs):
                x_ = self.bns_conv[i](x)
                x_ = F.relu(conv(x_, edge_index))
                x = x + x_ if self.conv_residual else x_
            node_features = x
            gate = 1 if self.gating is None else self.gating(x)
            x = self.global_pool(x * gate, batch)
            x = x if xg is None else x + xg
            for i, lin in enumerate(self.lins):
                x_ = self.bns_fc[i](x)
                x_ = F.relu(lin(x_))
                x = x + x_ if self.fc_residual else x_
            x = self.bn_hidden(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin_class(x)

        x_n = node_features
        #x_n = F.dropout(x_n, p=0.2, training=self.training)
        # for i, lin in enumerate(self.lins_nodes):
        #     x_ = self.bns_fc_nodes[i](x_n)
        #     x_ = F.relu(lin(x_))
        #     x_n = x_n + x_ if self.fc_residual else x_
        # x_n = self.bn_hidden_nodes(x_n)
        # if self.dropout > 0:
        #     x_n = F.dropout(x_n, p=self.dropout, training=self.training)
        x_n = self.lin_class_nodes(x_n)
        return F.log_softmax(x, dim=-1), F.log_softmax(x_n, dim=-1)

    def forward_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        return self.forward_BNConvReLU_cl(x, edge_index, batch, xg)

    def forward_BNConvReLU_cl(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x

    def forward_BNReLUConv(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(self.bns_conv[i](x))
            x_ = conv(x_, edge_index)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_ if self.fc_residual else x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def forward_ConvReLUBN(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        x = self.bn_hidden(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(conv(x, edge_index))
            x_ = self.bns_conv[i](x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(lin(x))
            x_ = self.bns_fc[i](x_)
            x = x + x_ if self.fc_residual else x_
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_resnet(self, x, edge_index, batch, xg=None):
        # this mimics resnet architecture in cv.
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i in range(len(self.convs) // 3):
            x_ = x
            x_ = F.relu(self.bns_conv[i*3+0](x_))
            x_ = self.convs[i*3+0](x_, edge_index)
            x_ = F.relu(self.bns_conv[i*3+1](x_))
            x_ = self.convs[i*3+1](x_, edge_index)
            x_ = F.relu(self.bns_conv[i*3+2](x_))
            x_ = self.convs[i*3+2](x_, edge_index)
            x = x + x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
