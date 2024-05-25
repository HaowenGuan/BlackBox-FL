import json
import torch.optim as optim
import argparse
import copy
import datetime
from sklearn.linear_model import LogisticRegression
import yaml

from PIL import Image

from models.model import *
from utils import *
import warnings
from models.model_factory_fn import *

warnings.filterwarnings('ignore')

fine_to_coarse = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7,
                  15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15,
                  28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5,
                  41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4,
                  54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12,
                  67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
                  80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2,
                  93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}

coarse_to_fine = {0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92], 3: [9, 10, 16, 28, 61],
                  4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87], 6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24],
                  8: [3, 42, 43, 88, 97], 9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71],
                  11: [15, 19, 21, 31, 38], 12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99],
                  14: [2, 11, 35, 46, 98], 15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80],
                  17: [47, 52, 56, 59, 96], 18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89]}

coarse_split = {'train': [1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19], 'valid': [8, 11, 13, 16], 'test': [0, 7, 12, 14]}

from collections import defaultdict

fine_split = defaultdict(list)

for fine_id, sparse_id in fine_to_coarse.items():
    if sparse_id in coarse_split['train']:
        fine_split['train'].append(fine_id)
    elif sparse_id in coarse_split['valid']:
        fine_split['valid'].append(fine_id)
    else:
        fine_split['test'].append(fine_id)

    # fine_split_train_map={class_:i for i,class_ in enumerate(fine_split['train'])}

# train_class2id={class_id: i for i, class_id in enumerate(fine_split['train'])}


import torchvision.transforms as transforms

# FC100
normalize_fc100 = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                       std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

# miniImageNet
mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
normalize_mini = transforms.Normalize(mean=mean_pix,
                                      std=std_pix)


# transform_train = transforms.Compose([
#     transforms.RandomCrop(32),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ])

def transform_train(normalize, crop_size=None, padding=None):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(crop_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])


# data prep for test set
def transform_test(normalize):
    return transforms.Compose([
        transforms.ToTensor(),
        normalize])


# transform_train=transform_test
def l2_normalize(x):
    norm = (x.pow(2).sum(1, keepdim=True) + 1e-9).pow(1. / 2)
    out = x.div(norm + 1e-9)
    return out


def InforNCE_Loss(anchor, sample, tau, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float).to(anchor.device)
    neg_mask = 1. - pos_mask
    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix != None else sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return -loss.mean(), sim


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='path to the config file')
    parser.add_argument('--model', type=str, default='resnet12', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='FC100', help='dataset used for training')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes in the dataset')
    parser.add_argument('--net_config', type=dict, help='network configuration')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01, 0.0005, 0.005)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')

    parser.add_argument('--method', type=str, default='new',
                        help='few-shot or normal')
    parser.add_argument('--mode', type=str, default='few-shot', help='few-shot or normal')
    parser.add_argument('--N', type=int, default=5, help='number of ways')
    parser.add_argument('--K', type=int, default=5, help='number of shots')
    parser.add_argument('--Q', type=int, default=5, help='number of queries')
    parser.add_argument('--num_train_tasks', type=int, default=1, help='number of meta-training tasks (5)')
    parser.add_argument('--num_test_tasks', type=int, default=10, help='number of meta-test tasks')
    parser.add_argument('--num_true_test_ratio', type=int, default=10, help='number of meta-test tasks (10)')
    parser.add_argument('--fine_tune_steps', type=int, default=5, help='number of meta-learning steps (5)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1, help='number of meta-learning lr (0.05)')
    parser.add_argument('--meta_lr', type=float, default=0.1 / 100, help='number of meta-learning lr (0.05)')
    parser.add_argument('--comm_round', type=int, default=5000, help='number of maximum communication roun')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')

    parser.add_argument("--bert_cache_dir", default=None, type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))
    parser.add_argument("--wv_path", type=str,
                        default="./",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", type=bool, default=False)
    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))

    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default="./checkpoints/",
                        help='Model checkpoint directory path')
    parser.add_argument('--beta', type=float, default=1,  # 0.5
                        help='The parameter for the dirichlet distribution for data partitioning')

    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--gpus', type=str, default='0', help='Visible GPUs for this task')

    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--server_pretrained', type=bool, default=False, help='whether the server is pretrained')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.config_file = "configs/image_FC100.yaml"
    if args.config_file is not None:
        # Read the configuration from a yaml file
        with open(args.config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        # Load the dictionary to an into args
        for key, value in config_dict.items():
            setattr(args, key, value)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(args.device)

    return args


    return nets, model_meta_data, layer_type


def train_net_few_shot_new(net_id, net, n_epoch, lr, args_optimizer, args, x_train_client, y_train_client, x_test,
                           y_test, device='cpu', test_only=False, test_only_k=0):
    """
    Train a network on a given dataset with meta learning
    :param net_id: the id of the network
    :param net: the network to train
    :param n_epoch: the number of epochs to train
    :param lr: the learning rate
    :param args_optimizer: the optimizer to use in string name
    :param args: the arguments
    :param x_train_client: the training data for this client
    :param y_train_client: the training labels for this client
    :param x_test: the test data
    :param y_test: the test labels
    :param device: the device to use
    :param test_only: whether to test only
    :param test_only_k: only used when test_only is True, the number of shots for testing
    """
    # net = nn.DataParallel(net)
    # net=nn.parallel.DistributedDataParallel(net)
    # net.cuda()

    # logger.info('Training network %s' % str(net_id))
    # logger.info('n_training: %d' % X_train_client.shape[0])
    # logger.info('n_test: %d' % X_test.shape[0])

    if args_optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.05, momentum=0.9,
                              weight_decay=args.reg)
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    def train_epoch(epoch, mode='train'):

        if mode == 'train':

            if args.dataset == 'fewrel':
                N = args.N * 3
                K = 2
                Q = 2
            elif args.dataset == 'huffpost':
                N = args.N
                K = 5  # args.K
                Q = args.Q
            elif args.dataset == 'FC100':
                N = args.N * 4
                K = 2
                Q = 2
            elif args.dataset == 'miniImageNet':
                N = args.N * 4
                K = 2
                Q = 2
            else:
                N = args.N
                K = 5  # args.K
                Q = args.Q
            net.train()
            optimizer.zero_grad()
            if args.dataset == 'FC100':
                # X_transform = transform_train(normalize=normalize_fc100, crop_size=32, padding=4)
                X_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize_fc100
                ])
            else:
                # X_transform = transform_train(normalize=normalize_mini, crop_size=84)
                X_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    # transforms.ToPILImage(),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize_mini
                ])

        else:
            N = args.N
            K = args.K
            Q = args.Q
            # N=args.N*2
            net.eval()
            if args.dataset == 'FC100':
                X_transform = transform_test(normalize=normalize_fc100)
            else:
                X_transform = transform_test(normalize=normalize_mini)

        if test_only:
            K = test_only_k

        support_labels = torch.zeros(N * K, dtype=torch.long)
        for i in range(N):
            support_labels[i * K:(i + 1) * K] = i
        query_labels = torch.zeros(N * Q, dtype=torch.long)
        for i in range(N):
            query_labels[i * Q:(i + 1) * Q] = i
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)

        if mode == 'train':
            if args.dataset == 'FC100':
                class_dict = fine_split['train']
            elif args.dataset == 'miniImageNet':
                class_dict = list(range(64))
            elif args.dataset == '20newsgroup':
                class_dict = [1, 5, 10, 11, 13, 14, 16, 18]
            elif args.dataset == 'fewrel':
                class_dict = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                              22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                              39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                              59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                              76, 77, 78]
            elif args.dataset == 'huffpost':
                class_dict = list(range(20))

            X = x_train_client
            y = y_train_client
            # for i in class_dict:
            # class_dict[i] = class_dict[i][:avail_train_num_per_class]
        elif mode == 'test':
            if args.dataset == 'FC100':
                class_dict = fine_split['test']
            elif args.dataset == 'miniImageNet':
                class_dict = list(range(20))
            elif args.dataset == '20newsgroup':
                class_dict = [0, 2, 3, 8, 9, 15, 19]
            elif args.dataset == 'fewrel':
                class_dict = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]
            elif args.dataset == 'huffpost':
                class_dict = list(range(25, 41))

            X = x_test
            y = y_test

        # prepare training data
        min_size = 0
        # Make sure that there are at least K + Q samples for each class
        while min_size < K + Q:
            X_class = []
            classes = np.random.choice(class_dict, N, replace=False).tolist()
            for i in classes:
                X_class.append(X[y == i])
                if X_class[-1].shape[0] < K + Q:
                    class_dict.remove(i)
            min_size = min([one.shape[0] for one in X_class])
            print("", sum([one.shape[0] for one in X_class]), X.shape[0])

        x_sup = []
        y_sup = []
        # Following labels are never used actually
        x_query = []
        y_query = []
        transformed_class_index_list = []
        for class_index, class_data in zip(classes, X_class):
            sample_idx = np.random.choice(list(range(class_data.shape[0])), K + Q, replace=False).tolist()
            x_sup.append(class_data[sample_idx[:K]])
            x_query.append(class_data[sample_idx[K:]])
            if mode == 'train':
                if args.dataset == 'FC100' or args.dataset == '20newsgroup' or args.dataset == 'fewrel' or args.dataset == 'huffpost':
                    transformed_class_index_list.append(fine_split_train_map[class_index])
                    y_sup.append(torch.ones(K) * fine_split_train_map[class_index])
                    y_query.append(torch.ones(Q) * fine_split_train_map[class_index])
                elif args.dataset == 'miniImageNet':
                    transformed_class_index_list.append(class_index)
                    y_sup.append(torch.ones(K) * class_index)
                    y_query.append(torch.ones(Q) * class_index)

        x_sup = np.concatenate(x_sup, 0)
        x_query = np.concatenate(x_query, 0)
        if mode == 'train':
            y_total = torch.cat([torch.cat(y_sup, 0), torch.cat(y_query, 0)], 0).long().to(args.device)

        # Apply the same transformation to the support set and the query set if its image dataset
        if args.dataset == 'FC100' or args.dataset == 'miniImageNet':
            X_total_transformed_sup = []
            X_total_transformed_query = []
            for i in range(x_sup.shape[0]):
                X_total_transformed_sup.append(X_transform(x_sup[i]))
            x_sup = torch.stack(X_total_transformed_sup, 0).to(args.device)

            for i in range(x_query.shape[0]):
                X_total_transformed_query.append(X_transform(x_query[i]))
            x_query = torch.stack(X_total_transformed_query, 0).to(args.device)
        else:
            x_sup = torch.tensor(x_sup).to(args.device)
            x_query = torch.tensor(x_query).to(args.device)

        # net.load_state_dict(net_para_ori)
        # _,_,y_hat=net_new(torch.cat([x_sup, x_query],0), all_classify=True)

        # print(ft_sup_y_hat[:3])
        if mode == 'train':
            loss_all = 0
            # all_classify update
            latent_embedding, latent_feature, y_hat = net(torch.cat([x_sup, x_query], 0), all_classify=True)
            sup_embedding = latent_embedding[:N * K].reshape([N, K, -1]).transpose(0, 1)
            query_embedding = latent_embedding[N * K:].reshape([N, Q, -1]).transpose(0, 1)

            # _, _, y_hat = net(x_sup, all_classify=True)

            if args.fine_tune_steps > 0:
                net_new = copy.deepcopy(net)
                print('Fine-tuning the model')

                for j in range(args.fine_tune_steps):
                    print('support shape', x_sup.shape, N, K)
                    ft_sup_embedding, ft_sup_transformer_feature, ft_sup_y_hat = net_new(x_sup)
                    loss = loss_ce(ft_sup_y_hat, support_labels)

                    net_para = net_new.state_dict()
                    param_require_grad = {}
                    for key, param in net_new.named_parameters():
                        if key == 'few_classify.weight' or key == 'few_classify.bias':
                            if param.requires_grad:
                                param_require_grad[key] = param
                    grads = torch.autograd.grad(loss, param_require_grad.values(), allow_unused=True)
                    for key, grad in zip(param_require_grad.keys(), grads):
                        if grad is None: continue
                        net_para[key] -= args.fine_tune_lr * grad
                    net_new.load_state_dict(net_para)

                X_out_query, _, ft_sup_y_hat = net_new(x_query)
                ft_sup_embedding, ft_sup_transformer_feature, _ = net_new(x_sup)

                ft_sup_transformer_feature = ft_sup_transformer_feature.reshape([N, K, -1]).transpose(0, 1)
                #############################
                # Q=K here update for all-model
                for j in range(Q):
                    contras_loss, similarity = InforNCE_Loss(ft_sup_transformer_feature[j], sup_embedding[(j + 1) % Q],
                                                             tau=0.5)
                    loss_all += contras_loss / Q * 0.1
                loss_all += loss_ce(y_hat, y_total)
                loss_all.backward()
                optimizer.step()
                ############################

                latent_embedding, latent_feature, y_hat = net(torch.cat([x_sup, x_query], 0), all_classify=True)
                ###################################
                # few_classify update
                net_para_ori = net.state_dict()

                param_require_grad = {}
                for key, param in net_new.named_parameters():
                    if key == 'few_classify.weight' or key == 'few_classify.bias' or 'transformer' in key:
                        # if key != 'module.all_classify.weight' and key != 'module.all_classify.bias':
                        param_require_grad[key] = param

                # meta-update few-classifier on query
                loss = loss_ce(ft_sup_y_hat, query_labels)
                out_sup_on_N_class = y_hat[N * K:, transformed_class_index_list]
                out_sup_on_N_class /= out_sup_on_N_class.sum(-1, keepdim=True)
                loss += loss_ce(ft_sup_y_hat, out_sup_on_N_class) * 0.1
                grads = torch.autograd.grad(loss, param_require_grad.values())
                for key, grad in zip(param_require_grad.keys(), grads):
                    net_para_ori[key] = net_para_ori[key] - args.meta_lr * grad
                net.load_state_dict(net_para_ori)
                ##################################
                del net_new, X_out_query, ft_sup_y_hat

            if np.random.rand() < 0.005:
                print('loss: {:.4f}'.format(loss_all.item()))

            acc_train = (torch.argmax(y_hat, -1) == y_total).float().mean().item()

            del latent_embedding, y_hat
            return acc_train

        else:
            use_logistic = True

            if use_logistic:
                with torch.no_grad():
                    latent_embedding, latent_feature, y_hat = net(torch.cat([x_sup, x_query], 0))
                    ft_sup_embedding = latent_embedding[:N * K]
                    X_out_query = latent_embedding[N * K:]

                    support_features = l2_normalize(ft_sup_embedding.detach().cpu()).numpy()
                    query_features = l2_normalize(X_out_query.detach().cpu()).numpy()

                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(support_features, support_labels.detach().cpu().numpy())

                    query_ys_pred = clf.predict(query_features)

                    ft_sup_y_hat = torch.tensor(clf.predict_proba(query_features)).to(args.device)

                    acc_train = (torch.argmax(ft_sup_y_hat, -1) == query_labels).float().mean().item()
                    max_value, index = torch.max(ft_sup_y_hat, -1)

                    # del net_new, ft_sup_embedding, X_out_query, ft_sup_y_hat, param_require_grad, grads
                    if test_only:
                        return acc_train, max_value, index
                    else:
                        return acc_train

                # return metrics.accuracy_score(query_labels.detach().cpu().numpy(), query_ys_pred)

            else:

                acc_train = (torch.argmax(ft_sup_y_hat, -1) == query_labels).float().mean().item()
                with torch.no_grad():
                    max_value, index = torch.max(ft_sup_y_hat, -1)

                del net_new, ft_sup_embedding, X_out_query, ft_sup_y_hat, net_para, param_require_grad, grads, x_query, x_sup
                if test_only:
                    return acc_train, max_value, index
                else:
                    return acc_train

    if not test_only:
        best_acc = 0
        accs_train = []
        for epoch in range(args.num_train_tasks):
            accs_train.append(train_epoch(epoch))
            # Print the training accuracy
            # logger.info("Meta-train_Accuracy: {:.4f}".format(np.mean(accs_train)))
            # print("Meta-train_Accuracy: {:.4f}".format(np.mean(accs_train)))

        accs = []
        for epoch_test in range(args.num_test_tasks):
            accs.append(train_epoch(epoch_test, mode='test'))
    else:
        accs = []
        max_values = []
        indices = []
        accs_train = []

        #########################################
        # train before test
        # for epoch in range(args.num_train_tasks//5):
        #    accs_train.append(train_epoch(epoch))
        #########################################

        for epoch_test in range(args.num_test_tasks * args.num_true_test_ratio):
            acc, max_value, index = train_epoch(epoch_test, mode='test')
            accs.append(acc)
            max_values.append(max_value)
            indices.append(index)
            del acc, max_value, index

        return np.mean(accs), torch.cat(max_values, 0), torch.cat(indices, 0)

    print(f"Client {net_id} Meta-test_Accuracy: {np.mean(accs):.4f}")
    # logger.info("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))

    return np.mean(accs)


def local_train_net_few_shot(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, device="cpu",
                             test_only=False, test_only_k=0):
    avg_acc = 0.0
    acc_list = []
    max_value_all_clients = []
    indices_all_clients = []

    for net_id, net in nets.items():

        # Each client has a list of data samples assigned to it in the format of indices
        dataidxs = net_dataidx_map[net_id]

        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        n_epoch = args.epochs

        # _,_, train_ds, test_ds = get_dataloader(args.dataset, args.datadir, args.batch_size, len(dataidxs), dataidxs)

        # X_train_client=train_ds.data
        # y_train_client=train_ds.target

        # get the private data for the client
        X_train_client = X_train[dataidxs]
        y_train_client = y_train[dataidxs]

        # X_test=test_ds.data
        # y_test=test_ds.target

        if test_only is False:
            testacc = train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client,
                                             y_train_client, X_test, y_test,
                                             device=device, test_only=False)
        else:
            # np.random.seed(1)
            testacc, max_values, indices = train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args,
                                                                  X_train_client, y_train_client, X_test, y_test,
                                                                  device=device, test_only=True,
                                                                  test_only_k=test_only_k)
            max_value_all_clients.append(max_values)
            indices_all_clients.append(indices)
            # np.random.seed(int(time.time()))

            acc_list.append(testacc)
            logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
            print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

            max_value_all_clients = torch.stack(max_value_all_clients, 0)
            indices_all_clients = torch.stack(indices_all_clients, 0)
            return acc_list, max_value_all_clients, indices_all_clients

        # logger.info("net {} final test acc {:.4f}" .format(net_id, testacc))

        avg_acc += testacc
        acc_list.append(testacc)

        # net.cpu()

    logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
    print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

    if test_only:
        max_value_all_clients = torch.stack(max_value_all_clients, 0)
        indices_all_clients = torch.stack(indices_all_clients, 0)
        return acc_list, max_value_all_clients, indices_all_clients

    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))

    return nets


if __name__ == '__main__':
    args = init_args()
    print(args)

    if args.dataset == 'FC100':
        fine_split_train_map = {class_: i for i, class_ in enumerate(fine_split['train'])}
    elif args.dataset == '20newsgroup':
        fine_split_train_map = {class_: i for i, class_ in enumerate([1, 5, 10, 11, 13, 14, 16, 18])}
    elif args.dataset == 'fewrel':
        fine_split_train_map = {class_: i for i, class_ in enumerate([0, 1, 2, 3,
                                                      4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21, 22, 24, 25,
                                                      26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                                      43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58, 59, 61, 62,
                                                      63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78])}
    elif args.dataset == 'huffpost':
        fine_split_train_map = {class_: i for i, class_ in enumerate(list(range(20)))}

    mkdirs(args.logdir)
    mkdirs(args.checkpoint_dir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    test_task_sample_seed = 1
    np.random.seed(test_task_sample_seed)
    test_classes = []
    test_index = []
    for i in range(args.num_test_tasks):
        test_classes.append(np.random.choice(fine_split['test'], args.N, replace=False).tolist())
        test_index.append(np.random.rand(args.N, args.K + args.Q))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    if args.dataset == '20newsgroup':
        seed = 13
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_data_idx_map, client_class_cnt = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    # for k, v in net_data_idx_map.items():
    #     print(k, len(v))
    # for i, l in enumerate(client_class_cnt):
    #     print(i, l)
    print(X_train.shape)
    print(X_test.shape)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args)

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    # TODO: set up the generator========================================================
    out_channel = 100
    in_channel = 3
    g_model_arch = 'CGeneratorA'
    nz = 100
    nc = 3
    img_size = 32
    from models import model_factory_fn

    g_model_func = lambda: model_factory_fn.get_generator(g_model_arch, nz=nz, nc=nc,
                                                          img_size=img_size, n_cls=out_channel)
    init_g_model = g_model_func()
    # TODO: ==============================================================================

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'fedavg':
        use_minus = False
        best_acc = 0
        best_acc_5 = 0
        best_confident_acc = 0

        for round in range(n_comm_rounds):
            # logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}

            total_data_points = sum([len(net_data_idx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_data_idx_map[r]) / total_data_points for r in range(args.n_parties)]

            # Distribute global model to all clients
            for net_id, net in nets_this_round.items():
                if use_minus:
                    net_para = net.state_dict()
                    for key in net_para:
                        net_para[key] = (global_w[key] * total_data_points - net_para[key] * len(
                            net_data_idx_map[net_id])) / (total_data_points + 1e-9 - len(net_data_idx_map[net_id]))
                    net.load_state_dict(net_para)
                else:
                    net_para = net.state_dict()
                    for key in net_para:
                        if key != 'few_classify.weight' and key != 'few_classify.bias' and 'transformer' not in key:
                            net_para[key] = global_w[key]
                    net.load_state_dict(net_para)

            for k in [1, 5]:
                print(k, len(nets_this_round))
                global_acc, max_value_all_clients, indices_all_clients = local_train_net_few_shot(nets_this_round, args,
                                                                                                  net_data_idx_map,
                                                                                                  X_train, y_train,
                                                                                                  X_test, y_test,
                                                                                                  device=device,
                                                                                                  test_only=True,
                                                                                                  test_only_k=k)
                global_acc = max(global_acc)
                if k == 1:
                    if global_acc > best_acc:
                        best_acc = global_acc
                    print('>> Global 1 Model Test accuracy: {:.4f} Best Acc: {:.4f}'.format(global_acc, best_acc))
                    logger.info(
                        '>> Global 1 Model Test accuracy: {:.4f} Best Acc: {:.4f} '.format(global_acc, best_acc))
                elif k == 5:
                    if global_acc > best_acc_5:
                        best_acc_5 = global_acc
                    print('>> Global 5 Model Test accuracy: {:.4f} Best Acc: {:.4f}'.format(global_acc, best_acc_5))
                    logger.info(
                        '>> Global 5 Model Test accuracy: {:.4f} Best Acc: {:.4f} '.format(global_acc, best_acc_5))

            local_train_net_few_shot(nets_this_round, args, net_data_idx_map, X_train, y_train, X_test, y_test,
                                     device=device)

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            # TODO: Use knowledge distillation for the global model===========================
            # from FedFTG import *
            #
            # # client_class_num = np.ones((n_partyCurrent Round_per_round, 100))
            # glb_model_lr = 0.1
            # gen_model_lr = 0.01
            # lr_decay_per_round = 0.99
            # global_model.load_state_dict(global_w)
            # batch_size = 50
            # print_per = 200
            # weight_decay = 1e-3
            #
            # avg_model_ft = local_to_global_knowledge_distillation(global_model, init_g_model, nets_this_round.values(),
            #                                                       np.array(client_class_cnt),
            #                                                       glb_model_lr * (lr_decay_per_round ** round),
            #                                                       gen_model_lr * (lr_decay_per_round ** round),
            #                                                       batch_size, print_per, args.reg,
            #                                                       'CIFAR10',
            #                                                       None, None, None, None, device=args.device)
            # TODO: ===========================================================================

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            # global_model = avg_model_ft

            # global_model.cuda()

            print('>> Current Round: {}'.format(round))
            logger.info('>> Current Round: {}'.format(round))

            mkdirs(args.checkpoint_dir + 'fedavg/')

            if global_acc > best_acc:
                torch.save(global_model.state_dict(),
                           args.checkpoint_dir + 'fedavg/' + 'globalmodel' + args.log_file_name + '.pth')
                torch.save(nets[0].state_dict(),
                           args.checkpoint_dir + 'fedavg/' + 'localmodel0' + args.log_file_name + '.pth')
