import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import wandb

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

resize = transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)

def generate_labels(number, class_num):
    labels = np.arange(number)
    proportions = class_num / class_num.sum()
    proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
    labels_split = np.split(labels, proportions)
    for i in range(len(labels_split)):
        labels_split[i].fill(i)
    labels = np.concatenate(labels_split)
    np.random.shuffle(labels)
    return labels.astype(int)


def get_batch_weight(labels, class_client_weight):
    bs = labels.size
    num_clients = class_client_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    batch_weight[np.arange(bs), :] = class_client_weight[labels, :]
    return batch_weight


def compute_backward_flow_G_dis(z, y_onehot, labels,
                                generator, student, teacher,
                                weight, num_clients, train_fedgen_feature=False,
                                device='cuda'):
    lambda_cls = 1.0
    lambda_dis = 1.0
    # cls_criterion = nn.CrossEntropyLoss().to(device)
    cls_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    diversity_criterion = DiversityLoss(metric='l1').to(device)

    y = torch.Tensor(labels).long().to(device)

    fake = generator(z, y_onehot)

    _, _, t_logit = teacher(fake)
    _, _, s_logit = student(resize(fake))

    loss_md = - torch.mean(torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1) * weight)

    # loss_cls = cls_criterion(t_logit, y)
    loss_cls = torch.mean(cls_criterion(t_logit, y) * weight.squeeze())

    loss_ap = diversity_criterion(z.view(z.shape[0],-1), fake)
    loss = loss_md + lambda_cls * loss_cls + lambda_dis * loss_ap / num_clients
    loss.backward()
    return loss, loss_md, loss_cls, loss_ap


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


# def set_client_from_params(mdl, params):
#     dict_param = copy.deepcopy(dict(mdl.state_dict()))
#     idx = 0
#     for name, param in dict(mdl.state_dict()).items():
#         weights = param.data
#         length = len(weights.reshape(-1))
#         dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
#         idx += length
#
#     mdl.load_state_dict(dict_param)
#     return mdl


# # --- Evaluate a NN model
# def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
#     acc_overall = 0
#     loss_overall = 0
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
#
#     # batch_size = min(6000, data_x.shape[0])
#     batch_size = min(2000, data_x.shape[0])
#     n_tst = data_x.shape[0]
#     tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
#     model.eval()
#     model = model.to(device)
#     with torch.no_grad():
#         tst_gen_iter = tst_gen.__iter__()
#         for i in range(int(np.ceil(n_tst / batch_size))):
#             batch_x, batch_y = tst_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             y_pred = model(batch_x)
#
#             loss = loss_fn(y_pred, batch_y.reshape(-1).long())
#
#             loss_overall += loss.item()
#
#             # Accuracy calculation
#             y_pred = y_pred.cpu().numpy()
#             y_pred = np.argmax(y_pred, axis=1).reshape(-1)
#             batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
#             batch_correct = np.sum(y_pred == batch_y)
#             acc_overall += batch_correct
#
#     loss_overall /= n_tst
#     if w_decay is not None:
#         # Add L2 loss
#         params = get_mdl_params([model], n_par=None)
#         loss_overall += w_decay / 2 * np.sum(params * params)
#
#     model.train()
#     return loss_overall, acc_overall / n_tst


def warmup_generator(
        args,
        student,
        generator,
        clients,
        client_class_num,
        gen_model_lr,
        batch_size,
        iterations,
        n_cls,
        device='cuda'):
    """
    Warmup generator.
    """
    generator.to(device)
    num_clients, num_classes = client_class_num.shape
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=gen_model_lr)

    class_num = np.sum(client_class_num, axis=0)
    class_client_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = class_client_weight.transpose()
    labels_all = generate_labels(iterations * batch_size, class_num)
    print('Start training generator and server model')
    loss_G_meta = []
    loss_md_total_meta = []
    loss_cls_total_meta = []
    loss_ap_total_meta = []
    for e in range(iterations):
        labels = labels_all[e * batch_size:(e * batch_size + batch_size)]
        batch_weight = torch.Tensor(get_batch_weight(labels, class_client_weight)).to(device)
        onehot = np.zeros((batch_size, num_classes))
        onehot[np.arange(batch_size), labels] = 1
        y_onehot = torch.Tensor(onehot).to(device)
        z = torch.randn((batch_size, n_cls, 1, 1)).to(device)

        ############## train generator ##############
        student.eval()
        generator.train()
        loss_G = 0
        loss_md_total = 0
        loss_cls_total = 0
        loss_ap_total = 0
        for client_i, c_model in enumerate(clients):
            optimizer_G.zero_grad()
            loss, loss_md, loss_cls, loss_ap = compute_backward_flow_G_dis(z, y_onehot, labels,
                                                                           generator, student, c_model,
                                                                           batch_weight[:, client_i], num_clients,
                                                                           device=device)
            loss_G += loss
            loss_md_total += loss_md
            loss_cls_total += loss_cls
            loss_ap_total += loss_ap
            optimizer_G.step()

        loss_G_meta.append(loss_G.item())
        loss_md_total_meta.append(loss_md_total.item())
        loss_cls_total_meta.append(loss_cls_total.item())
        loss_ap_total_meta.append(loss_ap_total.item())

    print('[WarmUp] KD Generator loss', loss_G_meta)
    print('[WarmUp] KD Generator model discrepancy loss', loss_md_total_meta)
    print('[WarmUp] KD Generator classification loss', loss_cls_total_meta)
    print('[WarmUp] KD Generator diversity loss', loss_ap_total_meta)


def local_to_global_knowledge_distillation(
        args,
        epoch,
        s_model,
        g_model,
        c_models,
        client_class_num,
        glb_model_lr,
        gen_model_lr,
        batch_size,
        weight_decay,
        iterations,
        g_inner_round,
        d_inner_round,
        n_cls,
        device='cuda',
        **kwargs):
    """
    Local to global knowledge distillation using FedFTG.
    https://arxiv.org/abs/2203.09249
    """
    s_model.to(device)
    g_model.to(device)
    num_clients, num_classes = client_class_num.shape
    for params in s_model.parameters():
        params.requires_grad = True
    optimizer_D = torch.optim.SGD(s_model.parameters(), lr=glb_model_lr, momentum=0.9, weight_decay=weight_decay)
    optimizer_G = torch.optim.Adam(g_model.parameters(), lr=gen_model_lr)

    class_num = np.sum(client_class_num, axis=0)
    class_client_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = class_client_weight.transpose()
    labels_all = generate_labels(iterations * batch_size, class_num)
    print('Start training generator and server model')
    # fake_data = []
    # fake_data_labels = []
    # server_embedding = []
    for e in range(iterations):
        labels = labels_all[e * batch_size:(e * batch_size + batch_size)]
        batch_weight = torch.Tensor(get_batch_weight(labels, class_client_weight)).to(device)
        onehot = np.zeros((batch_size, num_classes))
        onehot[np.arange(batch_size), labels] = 1
        y_onehot = torch.Tensor(onehot).to(device)
        z = torch.randn((batch_size, n_cls, 1, 1)).to(device)

        ############## train generator ##############
        s_model.eval()
        g_model.train()
        loss_G = 0
        loss_md_total = 0
        loss_cls_total = 0
        loss_ap_total = 0
        for i in range(g_inner_round):
            for client_i, c_model in enumerate(c_models):
                optimizer_G.zero_grad()
                loss, loss_md, loss_cls, loss_ap = compute_backward_flow_G_dis(z, y_onehot, labels,
                                                                               g_model, s_model, c_model,
                                                                               batch_weight[:, client_i], num_clients,
                                                                               device=device)
                loss_G += loss
                loss_md_total += loss_md
                loss_cls_total += loss_cls
                loss_ap_total += loss_ap
                optimizer_G.step()
        if args['log_wandb']:
            wandb.log({'KD Generator model discrepancy loss': loss_md_total}, step=epoch)
            wandb.log({'KD Generator classification loss': loss_cls_total}, step=epoch)
            wandb.log({'KD Generator diversity loss': loss_ap_total}, step=epoch)

        # fake = g_model(z, y_onehot)
        # import matplotlib
        # matplotlib.use('Agg')  # Use the 'Agg' backend which does not require a GUI
        # import matplotlib.pyplot as plt
        # for i in range(10):
        #     image = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
        #     plt.imshow(image)
        #     plt.title(f'fake{i}_class{labels[i]}')
        #     plt.show()
        #     plt.savefig(f'fake{i}_class{labels[i]}.png')
        # exit(0)

        ############## train student model ##############
        s_model.train()
        g_model.eval()
        MSEloss = nn.MSELoss()
        loss_D_total = 0
        with torch.no_grad():
            fake = g_model(z, y_onehot).detach()
        fake_224 = resize(fake)
        # fake_data.append(fake)
        # fake_data_labels.append(labels)
        # server_embedding.append(emb)
        for i in range(d_inner_round):
            optimizer_D.zero_grad()
            _, emb, s_logit = s_model(fake_224, all_classify=True)
            c_logit_merge = 0
            for client_i, c_model in enumerate(c_models):
                c_logit = c_model(fake)[2].detach()
                c_logit_merge += F.softmax(c_logit, dim=1) * batch_weight[:, client_i][:, np.newaxis].repeat(1, num_classes)
            loss_D = torch.mean(-F.log_softmax(s_logit, dim=1) * c_logit_merge)
            loss_D_total += loss_D
            # loss_D = MSEloss(s_logit, c_logit_merge)
            # print(f'Round {i} Student loss:', loss_D)
            loss_D.backward()
            optimizer_D.step()

        if args['log_wandb']:
            wandb.log({'KD Server loss': loss_D_total}, step=epoch)

        # fake_data = torch.cat(fake_data, dim=0)
        # fake_data_labels = np.concatenate(fake_data_labels, axis=0)
        # server_embedding = torch.cat(server_embedding, dim=0)
        # if (e + 1) % print_per == 0:
        #     loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, s_model, dataset_name)
        #     loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, s_model, dataset_name)
        #     print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
        #           % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, optimizer_D.param_groups[0]['lr']))
        #     print("Epoch %3d, Loss D: %.4f, Loss G: %.4f, Loss md: %.4f, Loss cls: %.4f, Loss ap: %.4f"
        #           % (e + 1, loss_D.item(), loss_G.item(), loss_md_total.item(), loss_cls_total.item(), loss_ap_total.item()))
        #     s_model.train()

    # Freeze model
    # for params in s_model.parameters():
    #     params.requires_grad = False
    s_model.eval()
    fake_data, fake_data_labels, server_embedding = None, None, None

    return s_model, fake_data, fake_data_labels, server_embedding