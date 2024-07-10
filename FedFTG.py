import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import wandb
from tqdm import tqdm

# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
#
# resize = transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)

def generate_labels(n, class_num):
    """
    Generate labels for generating data.
    The labels are generated according to the proportion of existence of each class in the training set.

    @param n: int, number of samples to generate.
    @param class_num: np.array (num_classes,), number of samples in each class.
    @return: np.array (n,), generated labels.
    """
    labels = np.arange(n)
    proportions = class_num / class_num.sum()
    proportions = (np.cumsum(proportions) * n).astype(int)[:-1]
    labels_split = np.split(labels, proportions)
    for i in range(len(labels_split)):
        labels_split[i].fill(i)
    labels = np.concatenate(labels_split)
    np.random.shuffle(labels)
    return labels.astype(int)


def generate_soft_labels(n, class_num):
    """
    Generate soft labels for generating data.

    @param n: int, number of samples to generate.
    @param class_num: np.array (num_classes,), number of samples in each class.
    @return: np.array (n, num_classes), generated soft labels.
    """
    classes = class_num.shape[0]
    soft_labels = np.zeros((n, classes))
    # Get non-zero sample classes from clients
    non_zero_classes = np.where(class_num > 0)[0]
    for i in range(n):
        # Randomly shuffle the training classes
        sequence = np.random.permutation(non_zero_classes)
        remaining = 1.0
        # Randomly assign the proportion of remaining logits to each class
        for j in range(len(sequence)):
            if remaining < 0.001:
                soft_labels[i, sequence[j]] = remaining
                break
            v = np.random.uniform(0.9, 1.0) * remaining
            soft_labels[i, sequence[j]] = v
            remaining -= v

        # if np.random.rand() < 0.5:
        #     # Randomly shuffle the training classes
        #     sequence = np.random.permutation(non_zero_classes)
        #     remaining = 1.0
        #     # Randomly assign the proportion of remaining logits to each class
        #     for j in range(len(sequence) - 1):
        #         v = np.random.uniform(0, remaining)
        #         soft_labels[i, sequence[j]] = v
        #         remaining -= v
        #     soft_labels[i, sequence[-1]] = remaining
        # else:
        #     # Assign a more confident label
        #     soft_labels[i, np.random.choice(non_zero_classes)] = 1.0
    return soft_labels


def get_batch_weight(labels, class_client_weight):
    """

    @param labels: np.array (bs,)
        each value [0, num_classes) is a integer class number.
    @param class_client_weight: np.array (num_classes, num_clients)
        each value [0, 1] is the proportion of the class in the client.
    @return: np.array (bs, num_clients)
    each value [0, 1] is the weight of clients contribute to the specific label.
    """
    bs = labels.size
    num_clients = class_client_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    batch_weight[np.arange(bs), :] = class_client_weight[labels, :]
    return batch_weight


def get_soft_batch_weight(soft_labels, class_client_weight):
    """
    Input:
    soft_labels shape: (bs, num_classes)
        each value [0, 1] is a likelihood of sample belonging to the class.
    class_client_weight shape: (num_classes, num_clients)
        each value [0, 1] is the proportion of the class in the client.

    return:
    batch_weight shape: (bs, num_clients)
        each value [0, 1] is the weight of each client contributing to the sample.
    """
    bs = soft_labels.shape[0]
    num_clients = class_client_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    for i in range(num_clients):
        batch_weight[:, i] = np.sum(soft_labels * class_client_weight[:, i].reshape(1, -1), axis=1)
    return batch_weight


def compute_backward_flow_G_dis(z, y_one_hot, labels,
                                generator, student, teacher,
                                weight, num_clients, device='cuda', calc_md=True):
    lambda_cls = 1.0
    lambda_dis = 1.0
    cls_criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    diversity_criterion = DiversityLoss(metric='l1').to(device)

    fake = generator(z, y_one_hot)
    _, t_logit = teacher(fake)

    if calc_md:
        _, s_logit = student(fake)
        md_loss = - torch.mean(torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1) * weight)
    else:
        md_loss = torch.tensor(0.0).to(device)

    y = torch.Tensor(labels).long().to(device)
    cls_loss = torch.mean(cls_criterion(t_logit, y) * weight.squeeze())
    acc = (torch.argmax(t_logit, dim=1) == y).float().mean()

    # loss_ap = diversity_criterion(z.view(z.shape[0],-1), fake)
    loss_ap = torch.tensor(0.0).to(device)
    loss = md_loss + lambda_cls * cls_loss + lambda_dis * loss_ap / num_clients
    loss.backward()
    return loss, md_loss, cls_loss, loss_ap, acc


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

def global_to_local_knowledge_distillation(args, step, clients, client_optimizers, client_class_num, generator, batch_size, device='cpu'):
    num_clients, num_classes = client_class_num.shape
    class_num = np.sum(client_class_num, axis=0)
    client_class_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = client_class_weight.transpose()

    generator.eval()
    for c_id, client in clients.items():
        client.train()
    soft_label = generate_soft_labels(batch_size, class_num)
    batch_weight = torch.Tensor(get_soft_batch_weight(soft_label, class_client_weight)).to(device)
    soft_label = torch.Tensor(soft_label).to(device)
    z = torch.randn((batch_size, args['num_classes'], 1, 1)).to(device)
    fake_data = generator(z, soft_label).detach()
    c_logits = dict()
    c_logit_merge = 0
    for c_id, client in clients.items():
        _, c_logit = client(fake_data)
        c_logits[c_id] = F.softmax(c_logit, dim=1)
        c_logit_merge += F.softmax(c_logit.detach(), dim=1) * batch_weight[:, c_id][:, None]

    fl_loss_total = []
    for c_id, client in clients.items():
        client_optimizers[c_id].zero_grad()
        fl_loss = calculate_loss(c_logits[c_id], c_logit_merge, 'KLDiv')
        fl_loss_total.append(fl_loss.item())
        fl_loss.backward()
        client_optimizers[c_id].step()
    fl_loss_total = np.mean(fl_loss_total)
    if args['log_wandb']:
        wandb.log({'BlackBox FL Clients Loss': fl_loss_total}, step=step)


def train_generator(
        args,
        server,
        generators,
        clients,
        generator_optimizers,
        client_class_num,
        batch_size,
        iterations,
        n_cls,
        get_data=False,
        early_stop=10,
        device='cuda'):
    """
    Train the generator model for a certain iteration.
    """
    num_clients, num_classes = client_class_num.shape
    class_num = np.sum(client_class_num, axis=0)
    # print('class_num:', class_num)
    class_client_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = class_client_weight.transpose()
    labels_all = generate_labels(iterations * batch_size, class_num)
    e = 0
    loss_G_total = []
    md_loss_total = []
    cls_loss_total = []
    loss_ap_total = []
    acc_total = []

    for g in generators.values():
        g.to(device)
        g.train()

    for e in range(iterations):
        if e > early_stop and sum(acc_total[-early_stop:])/early_stop > 0.95:
            break
        labels = labels_all[e * batch_size:(e * batch_size + batch_size)]
        batch_weight = torch.Tensor(get_batch_weight(labels, class_client_weight)).to(device)
        # batch_weight = torch.ones(batch_size).to(device)
        one_hot = np.zeros((batch_size, num_classes))
        one_hot[np.arange(batch_size), labels] = 1
        y_one_hot = torch.Tensor(one_hot).to(device)
        z = torch.randn((batch_size, n_cls, 1, 1)).to(device)

        ############## train generator ##############
        server.eval()
        for i, (client_id, client) in enumerate(list(sorted(clients.items(), key=lambda x: x[0]))):
            generator_optimizers[client_id].zero_grad()
            loss_G, md_loss, cls_loss, loss_ap, acc = compute_backward_flow_G_dis(z, y_one_hot, labels,
                                                                               generators[client_id], server, client,
                                                                               batch_weight[:, i], num_clients,
                                                                               device=device, calc_md=False)
            if args['log_wandb'] and args['log_generator']:
                wandb.log({f'Generator_{client_id}_md_loss': md_loss.item(), f'Generator_{client_id}_cls_loss': cls_loss.item(),
                           f'Generator_{client_id}_loss_ap': loss_ap.item(), f'Generator_{client_id}_Accuracy': acc})
            generator_optimizers[client_id].step()

            loss_G_total.append(loss_G.item())
            md_loss_total.append(md_loss.item())
            cls_loss_total.append(cls_loss.item())
            loss_ap_total.append(loss_ap.item())
            acc_total.append(acc)

    if get_data:

        for client_id, client in clients.items():
            generators[client_id].eval()
            with torch.no_grad():
                z = torch.randn((batch_size, n_cls, 1, 1)).to(device)
                one_hot = np.zeros((batch_size, num_classes))
                labels = labels_all[:batch_size]
                one_hot[np.arange(batch_size), labels] = 1
                y_one_hot = torch.Tensor(one_hot).to(device)
                fake = generators[client_id](z, y_one_hot).detach()
                # _, c_logit = client(fake)
                # acc = (torch.argmax(c_logit, dim=1) == torch.tensor(labels).to(device)).float().mean().item()
                # print(f'Client {client_id} verify accuracy: {acc}')
            return fake, labels

    print(f'Warmup generator done at step {e}')
    return e == (iterations - 1)


def calculate_loss(output, target, loss_function):
    """
    Calculate the loss between the output and target.

    @param output: Must be probabilistic form (e.g. softmax)
    @param target: Must be probabilistic form (e.g. softmax), and must be detached from gradient diagram
    @param loss_function: Loss function to use
    @return: Loss value
    """
    MSELoss = nn.MSELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    if loss_function == 'MSE':
        loss = MSELoss(output, target)
    elif loss_function == 'KLDiv':
        loss = KLDivLoss(torch.log(output), target)
    else:
        raise ValueError(f'Loss function {loss_function} not implemented')

    return loss


def blackbox_federated_learning_with_generator(
        args,
        server,
        server_optimizer,
        clients,
        client_optimizers,
        generator,
        generator_optimizer,
        client_class_num,
        batch_size,
        iterations,
        loss_function,
        warmup=False,
        use_md_loss=False,
        device='cuda',
        **kwargs):
    """
    Local to global knowledge distillation using FedFTG.
    https://arxiv.org/abs/2203.09249
    """
    num_clients, num_classes = client_class_num.shape
    class_num = np.sum(client_class_num, axis=0)
    client_class_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = client_class_weight.transpose()

    wandb_step = wandb.run.step if args['log_wandb'] else 0
    cls_loss = torch.Tensor([0]).to(device)
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    for s in tqdm(range(iterations)):
        soft_label = generate_soft_labels(batch_size, class_num)
        batch_weight = torch.Tensor(get_soft_batch_weight(soft_label, class_client_weight)).to(device)
        soft_label = torch.Tensor(soft_label).to(device)
        z = torch.randn((batch_size, num_classes, 1, 1)).to(device)

        ############## Local to Global Knowledge Distillation ##############
        server.eval()
        generator.train()
        generator_optimizer.zero_grad()
        c_logit_merge = torch.zeros((batch_size, num_classes)).to(device)
        fake_data = generator(z, soft_label)
        
        # Merge all client logits according to weights
        for c_id, client in clients.items():
            _, c_logit = client(fake_data)
            c_logit_merge += F.softmax(c_logit, dim=1) * batch_weight[:, c_id][:, None]

        # Classification Loss
        cls_loss = calculate_loss(c_logit_merge, soft_label, loss_function)
        
        # Model Discrepancy Loss
        if use_md_loss:
            _, s_logit = server(fake_data)
            s_logit = F.softmax(s_logit, dim=1)
            md_loss = - calculate_loss(s_logit, c_logit_merge.detach(), loss_function)
        else:
            md_loss = torch.Tensor([0]).to(device)

        (cls_loss + md_loss).backward()
        generator_optimizer.step()

        if args['log_wandb']:
            clients_l1_loss = L1Loss(c_logit_merge.detach(), soft_label)
            clients_l2_loss = MSELoss(c_logit_merge.detach(), soft_label)
            wandb.log({'Clients merge logit verify L1 Loss': clients_l1_loss}, step=wandb_step + s)
            wandb.log({'Clients merge logit verify L2 Loss': clients_l2_loss}, step=wandb_step + s)
        
        if args['log_wandb']:
            if md_loss.item() != 0:
                wandb.log({'Fake data Model Discrepancy Loss': md_loss.item()}, step=wandb_step + s)
            wandb.log({'KD Generator classification loss': cls_loss.item()}, step=wandb_step + s)

        ############## Global to Local Knowledge Distillation ##############
        if not warmup:
            global_to_local_knowledge_distillation(
                args=args,
                step=wandb_step + s,
                clients=clients,
                client_optimizers=client_optimizers,
                client_class_num=client_class_num,
                generator=generator,
                batch_size=args['KD_config']['batch_size'],
                device=device
            )
        ############## Train Server Model ##############
        # if not warmup:
        #     server.train()
        #     generator.eval()
        #
        #     fake_data = generator(z, soft_label).detach()
        #     c_logit_merge = 0
        #     for c_id, client in clients.items():
        #         c_logit = client(fake_data)[1].detach()
        #         c_logit_merge += F.softmax(c_logit, dim=1) * batch_weight[:, c_id][:, None]
        #
        #     kd_loss_total = []
        #     for i in range(d_inner_round):
        #         server_optimizer.zero_grad()
        #         _, s_logit = server(fake_data)
        #         s_logit = F.softmax(s_logit, dim=1)
        #         kd_loss = calculate_loss(s_logit, c_logit_merge, loss_function)
        #         kd_loss_total.append(kd_loss.item())
        #         kd_loss.backward()
        #         server_optimizer.step()
        #     kd_loss_total = np.mean(kd_loss_total)
        #
        #     if args['log_wandb']:
        #         wandb.log({'Client to Server KD L2 Loss': kd_loss_total}, step=wandb_step + s)

    if warmup:
        return cls_loss.item()


def local_to_global_knowledge_distillation_independent(
        args,
        epoch,
        server,
        server_optimizer,
        clients,
        generators,
        generator_optimizers,
        client_class_num,
        batch_size,
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
    server.to(device)
    num_clients, num_classes = client_class_num.shape

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
        one_hot = np.zeros((batch_size, num_classes))
        one_hot[np.arange(batch_size), labels] = 1
        y_one_hot = torch.Tensor(one_hot).to(device)
        z = torch.randn((batch_size, n_cls, 1, 1)).to(device)

        ############## train generator ##############
        fakes = {}
        labels = {}
        for i, client in clients.items():
            fake, label = train_generator(
                args,
                server,
                generators,
                {i: client},
                generator_optimizers,
                np.array(client_class_num)[i:i + 1],
                batch_size,
                1,
                n_cls,
                get_data=True,
                device='cuda')
            fakes[i] = fake
            labels[i] = label

        ############## train student model ##############
        server.train()
        MSEloss = nn.MSELoss()
        loss_D_total = 0
        # fake_data.append(fake)
        # fake_data_labels.append(labels)
        # server_embedding.append(emb)
        c_logits = {}
        c_acc = []
        for i, client in clients.items():
            fake_32 = fakes[i]
            c_logits[i] = F.log_softmax(client(fake_32)[1].detach(), dim=1)

            c_acc.append((torch.argmax(c_logits[i], dim=1) == torch.tensor(labels[i]).to(device)).float().mean().item())
        print(f'Epoch {e}, Checking Client Accuracy toward its own data: {np.mean(c_acc)}')
        acc = []
        loss_total = []
        for i in range(d_inner_round):
            for c_id in clients:
                server_optimizer.zero_grad()
                h, s_logit = server(fakes[c_id])
                # loss_D = torch.mean(-F.log_softmax(s_logit, dim=1) * c_logits[c_id])
                loss_D = MSEloss(s_logit, c_logits[c_id])
                loss_total.append(loss_D.item())
                # print(torch.argmax(s_logit, dim=1).size())
                # print(torch.tensor(labels[c_id]).size())
                # print(torch.argmax(s_logit, dim=1) == torch.tensor(labels[c_id]).to(device))
                acc.append((torch.argmax(s_logit, dim=1) == torch.tensor(labels[c_id]).to(device)).float().mean().item())
                loss_D.backward()
                server_optimizer.step()
        print(f'Epoch {e}, loss: {np.mean(loss_total)}')
        print(f'Epoch {e}, acc: {np.mean(acc)}')
        acc = np.mean(acc)

        if args['log_wandb']:
            wandb.log({'KD Server acc': acc})
            # wandb.log({'KD Server acc': acc}, step=epoch)

        # fake_data = torch.cat(fake_data, dim=0)
        # fake_data_labels = np.concatenate(fake_data_labels, axis=0)
        # server_embedding = torch.cat(server_embedding, dim=0)
        # if (e + 1) % print_per == 0:
        #     loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, server, dataset_name)
        #     loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, server, dataset_name)
        #     print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
        #           % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, optimizer_S.param_groups[0]['lr']))
        #     print("Epoch %3d, Loss D: %.4f, Loss G: %.4f, Loss md: %.4f, Loss cls: %.4f, Loss ap: %.4f"
        #           % (e + 1, loss_D.item(), loss_G.item(), md_loss_total.item(), cls_loss_total.item(), loss_ap_total.item()))
        #     server.train()

    # Freeze model
    # for params in server.parameters():
    #     params.requires_grad = False
    server.eval()
    fake_data, fake_data_labels, server_embedding = None, None, None

    return server, fake_data, fake_data_labels, server_embedding


def local_to_global_knowledge_dataset_distillation(
        args,
        server,
        server_optimizer,
        clients,
        fake_data,
        fake_data_optimizer,
        fake_target,
        client_class_num,
        iterations,
        d_inner_round=5,
        device='cuda',
        warmup=False,
        **kwargs):
    server.to(device)
    num_clients, num_classes = client_class_num.shape
    class_num = np.sum(client_class_num, axis=0)
    client_class_weight = client_class_num / (np.tile(class_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    class_client_weight = client_class_weight.transpose()

    for c_id, client in clients.items():
        for p in client.parameters():
            p.requires_grad = False
        client.eval()

    class_weight = torch.Tensor(get_batch_weight(fake_target.cpu().numpy(), class_client_weight)).to(device)
    server_acc = 0
    client_acc = 0
    start = 0
    if args['log_wandb']:
        start = wandb.run.step

    for s in range(start, start + iterations):
        if warmup and client_acc > 0.95:
            print('Warmup Early Stopped')
            break
        ############## Train Fake Data ##############
        server.eval()
        cls_loss_total = []
        md_loss_total = []
        client_acc = []
        for c_id, client in clients.items():
            fake_data_optimizer.zero_grad()

            cls_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

            _, c_logit = client(fake_data)

            loss = torch.mean(cls_criterion(c_logit, fake_target) * class_weight[:, c_id])
            cls_loss_total.append(loss.item())

            if server_acc > 0.95:
                _, s_logit = server(fake_data)
                md_loss = - torch.mean(torch.mean(torch.abs(s_logit - c_logit.detach()), dim=1) * class_weight[:, c_id])
                md_loss_total.append(md_loss.item())
                loss += md_loss

            acc = (torch.argmax(c_logit, dim=1) == fake_target).float().mean()
            client_acc.append(acc.item())

            loss.backward()
            fake_data_optimizer.step()

        cls_loss_total = np.mean(cls_loss_total)
        client_acc = np.mean(client_acc)

        if args['log_wandb']:
            if len(md_loss_total) > 0:
                md_loss_total = np.mean(md_loss_total)
                wandb.log({'Fake data Model Discrepancy Loss': md_loss_total}, step=s)
            wandb.log({'Fake data Classification Loss': cls_loss_total}, step=s)
            wandb.log({'Fake data Client Acc': client_acc}, step=s)

        if not warmup:
            ############## Train Server Model ##############
            c_logit_merge = 0
            for c_id, client in clients.items():
                c_logit = client(fake_data)[1].detach()
                c_logit_merge += F.softmax(c_logit, dim=1) * class_weight[:, c_id][:, np.newaxis].repeat(1, num_classes)

            server.train()
            MSELoss = nn.MSELoss()
            kd_loss_total = []
            server_acc = []
            for i in range(d_inner_round):
                server_optimizer.zero_grad()
                _, s_logit = server(fake_data)
                server_acc.append((torch.argmax(s_logit, dim=1) == fake_target).float().mean().item())
                loss = MSELoss(s_logit, c_logit_merge)
                kd_loss_total.append(loss.item())
                loss.backward()
                server_optimizer.step()
            kd_loss_total = np.mean(kd_loss_total)
            server_acc = np.mean(server_acc)

            if args['log_wandb']:
                wandb.log({'KD Server Loss': kd_loss_total}, step=s)
                wandb.log({'KD Server Acc': server_acc}, step=s)

    for c_id, client in clients.items():
        for p in client.parameters():
            p.requires_grad = True
        client.train()
    server.eval()