import torch
import torch.nn as nn
import torch.nn.functional as F


def general_one_epoch(net, data_loader, optimizer=None, device='cpu'):
    """
    General one epoch function for training and validation
    [Note] if optimizer is provided, it will train the model.
    Make sure to call net.train() and net.eval() accordingly before calling this function.
    """
    total_loss = 0
    total_acc = 0
    data_num = 0
    for i, (data, target) in enumerate(data_loader):
        data_num += len(target)
        data, target = data.to(device), target.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        h, y_hat = net(data)
        loss = F.cross_entropy(y_hat, target)
        acc = (y_hat.argmax(dim=1) == target).sum().item()
        total_loss += loss.item() * len(target)
        total_acc += acc
        if optimizer is not None:
            loss.backward()
            optimizer.step()
    total_loss /= data_num
    total_acc /= data_num
    return total_loss, total_acc