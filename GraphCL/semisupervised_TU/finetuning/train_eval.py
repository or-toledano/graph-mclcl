import sys
import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

from utils import print_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_train_test(train_dataset,
                      test_dataset,
                      model_func,
                      epochs,
                      batch_size,
                      lr,
                      lr_decay_factor,
                      lr_decay_step_size,
                      weight_decay,
                      epoch_select,
                      with_eval_mode=True):
    assert epoch_select in ['test_last', 'test_max'], epoch_select

    model = model_func(train_dataset).to(device)
    print_weights(model)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_accs, test_accs = [], []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_loss, train_acc = train(
            model, optimizer, train_loader, device)
        train_accs.append(train_acc)
        test_accs.append(eval_acc(model, test_loader, device, with_eval_mode))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print('Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
            epoch, train_accs[-1], test_accs[-1]))
        sys.stdout.flush()

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    t_end = time.perf_counter()
    duration = t_end - t_start

    if epoch_select == 'test_max':
        train_acc = max(train_accs)
        test_acc = max(test_accs)
    else:
        train_acc = train_accs[-1]
        test_acc = test_accs[-1]

    return train_acc, test_acc, duration


def cross_validation_with_val_set(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None, model_PATH=None, semi_split=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, semi_split))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = model_func(dataset).to(device)
        model_dict = torch.load(model_PATH)
        # new_model_dict = dict()
        # for key, value in model_dict.items():
        #     if 'bn_feat' in key:
        #         new_key = key.replace('bn_feat', 'bn_feat.bn')
        #     elif 'bns_conv' in key:
        #         new_key = key.replace('bns_conv', 'bns_conv.bn')
        #     elif 'bns_fc' in key:
        #         new_key = key.replace('bns_fc', 'bns_fc.bn')
        #     elif 'bn_hidden' in key:
        #         new_key = key.replace('bn_hidden', 'bn_hidden.bn')
        #     else:
        #         new_key = key
        #     new_model_dict[new_key] = value
        model.load_state_dict(model_dict, strict=False)

        if fold == 0:
            print_weights(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                model, optimizer, train_loader, device)
            train_accs.append(train_acc)
            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode))
            test_accs.append(eval_acc(
                model, test_loader, device, with_eval_mode))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    train_acc = train_acc[train_acc <= 1.0]
    if train_acc.ndim == 1:
        train_acc = train_acc[None, :]
    test_acc = test_acc[test_acc <= 1.0]
    if test_acc.ndim == 1:
        test_acc = test_acc[None, :]
    val_loss = val_loss[val_loss >= 0.0]
    if val_loss.ndim == 1:
        val_loss = val_loss[None, :]
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(test_acc.shape[0], dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))
    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def k_fold(dataset, folds, epoch_select, semi_split):
    if dataset.name == "BIONIC":
        skf = KFold(folds, shuffle=True, random_state=44)
    else:
        skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for train_idx, test_idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(test_idx))
        train_indices.append(torch.from_numpy(train_idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    # skf_semi = StratifiedKFold(semi_split, shuffle=True, random_state=12345)
    # for i in range(folds):
    #     train_mask = torch.ones(len(dataset), dtype=torch.uint8)
    #     train_mask[test_indices[i].long()] = 0
    #     train_mask[val_indices[i].long()] = 0
    #     idx_train = train_mask.nonzero().view(-1)
    #
    #     for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):
    #         idx_train = idx_train[idx]
    #         break
    #
    #     train_indices.append(idx_train)

    test_indices = [x.type(torch.long) for x in test_indices]
    train_indices = [x.type(torch.long) for x in train_indices]
    val_indices = [x.type(torch.long) for x in val_indices]

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def get_node_labels(data, num_node_labels):
    labels_one_hot = data.x[:, :num_node_labels]
    labels = torch.argmax(labels_one_hot, dim=1)
    labels_mask = labels != num_node_labels - 1
    return labels, labels_mask, labels_one_hot[:, :num_node_labels - 1]

def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    correct = 0
    total_node_labels = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        labels, mask, labels_one_hot = get_node_labels(data, model.num_node_labels)
        out_g, out_n = model(data)
        if loader.dataset.name == 'BIONIC':
            out_n = out_n[mask, :]
            labels = labels[mask]
            labels_one_hot = labels_one_hot[mask]
            #loss_n = torch.mean(torch.clamp(1 - labels_one_hot * out_n, min=0))
            loss_n = F.nll_loss(out_n, labels)
            loss_n.backward()
            pred = out_n.max(1)[1]
            correct += pred.eq(labels).sum().item()
            total_loss += loss_n
            total_node_labels += out_n.shape[0]
        else:
            loss_g = F.nll_loss(out_g, data.y.view(-1))
            pred = out_g.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            loss_g.backward()
            total_loss += loss_g.item() * num_graphs(data)
        optimizer.step()
    if loader.dataset.name == 'BIONIC':
        if loader.dataset.name == 'BIONIC':
            print(f"total_node_labels train: {total_node_labels}")
            if total_node_labels > 0:
                return total_loss / total_node_labels, correct / total_node_labels
            else:
                return -1, 2
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    total_node_labels = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred_g, pred_n = model(data)
        if loader.dataset.name == 'BIONIC':
            labels, mask, _ = get_node_labels(data, model.num_node_labels)
            pred_n = pred_n[mask, :]
            labels = labels[mask]
            pred_n = pred_n.max(1)[1]
            correct += pred_n.eq(labels).sum().item()
            total_node_labels += pred_n.shape[0]
        else:
            pred_g = pred_g.max(1)[1]
            correct += pred_g.eq(data.y.view(-1)).sum().item()
    if loader.dataset.name == 'BIONIC':
        print(f"total_node_labels eval acc: {total_node_labels}")
        if total_node_labels > 0:
            return correct / total_node_labels
        else:
            return 2
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    total_node_labels = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out_g, out_n = model(data)
        if loader.dataset.name == 'BIONIC':
            labels, mask, labels_one_hot = get_node_labels(data, model.num_node_labels)
            out_n = out_n[mask, :]
            labels = labels[mask]
            labels_one_hot = labels_one_hot[mask]
            #loss += torch.mean(torch.clamp(1 - labels_one_hot * out_n, min=0))
            loss += F.nll_loss(out_n, labels, reduction='sum').item()
            total_node_labels += out_n.shape[0]
        else:
            loss += F.nll_loss(out_g, data.y.view(-1), reduction='sum').item()
    if loader.dataset.name == 'BIONIC':
        print(f"total_node_labels eval loss: {total_node_labels}")
        if total_node_labels > 0:
            return loss / total_node_labels
        else:
            return -1
    else:
        return loss / len(loader.dataset)