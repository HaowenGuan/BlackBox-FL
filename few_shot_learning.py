import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def few_shot_prototype(net, x_sup, y_sup, x_qry, y_qry):
    """
    Perform [prototype calculation] + [cosine similarity] for few-shot learning.
    https://arxiv.org/pdf/2003.04390
    Based on paper, meta training with this approach will overfit training base classes.
    Specifically, meta testing acc on base classes will increase but novel classes will decrease.
    [Warning]
    - Make sure to call net.train() and net.eval() accordingly before this function.
    - Make sure net and data are on the same device before this function.
    """
    embedding_sup, y_hat_sup = net(x_sup)
    embedding_qry, y_hat_qry = net(x_qry)
    classes = sorted(set(y_sup.tolist()))
    prototype = torch.zeros(len(classes), embedding_sup.size(1), device=embedding_sup.device)
    new_y_qry = torch.zeros_like(y_qry)
    for i, c in enumerate(classes):
        prototype[i] = embedding_sup[y_sup == c].mean(0)
        new_y_qry[y_qry == c] = i
    y_qry = new_y_qry
    prototype = F.normalize(prototype) # C x Z
    embedding_qry = F.normalize(embedding_qry) # Q x Z
    logits = net.tau * torch.mm(embedding_qry, prototype.t()) # Q x C
    loss = F.cross_entropy(logits, y_qry)
    acc = (logits.argmax(1) == y_qry).float().mean().item()
    return loss, acc


def few_shot_logistic_regression(net, x_sup, y_sup, x_qry, y_qry):
    """
    Perform logistic regression on the support set and predict the query set.
    """
    embedding_sup, _, _ = net(x_sup)
    embedding_qry, _, _ = net(x_qry)

    classes = sorted(set(y_sup.tolist()))
    new_y_sup = torch.zeros_like(y_sup)
    new_y_qry = torch.zeros_like(y_qry)
    for i, c in enumerate(classes):
        new_y_sup[y_sup == c] = i
        new_y_qry[y_qry == c] = i
    y_sup = new_y_sup
    y_qry = new_y_qry

    def l2_normalize(x):
        norm = (x.pow(2).sum(1, keepdim=True) + 1e-9).pow(1. / 2)
        out = x.div(norm + 1e-9)
        return out

    embedding_sup = l2_normalize(embedding_sup.detach().cpu()).numpy()
    embedding_qry = l2_normalize(embedding_qry.detach().cpu()).numpy()

    clf = LogisticRegression(penalty='l2',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(embedding_sup, y_sup.detach().cpu().numpy())

    # query_y_hat = clf.predict(embedding_qry)
    query_y_hat_prob = torch.tensor(clf.predict_proba(embedding_qry)).to(y_qry.device)

    acc = (torch.argmax(query_y_hat_prob, -1) == y_qry).float().mean().item()
    return acc

