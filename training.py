import torch
import torch.nn as nn
import torch.nn.functional as F


def general_one_epoch(net, data_loader, optimizer=None, device='cpu'):
    """
    General one epoch function for training and validation
    [Important] if optimizer is provided, it will train the model.
    Make sure to call net.train() and net.eval() accordingly before calling this function.
    """
    loss = 0
    acc = 0
    data_num = 0
    for i, (data, target) in enumerate(data_loader):
        data_num += len(target)
        data, target = data.to(device), target.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        h, y_hat = net(data)
        loss = F.cross_entropy(y_hat, target)
        acc = (y_hat.argmax(dim=1) == target).sum().item()
        loss += loss.item() * len(target)
        acc += acc
        if optimizer is not None:
            loss.backward()
            optimizer.step()
    loss /= data_num
    acc /= data_num
    return loss, acc