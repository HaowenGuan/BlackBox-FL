import torch.optim as optim
import argparse
import copy
import datetime
import random
import yaml
import sys

import wandb as wandb
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from few_shot_learning import few_shot_prototype, few_shot_logistic_regression
from training import general_one_epoch


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

resize = transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)


from PIL import Image

from models.model import *
from utils import *
import warnings
from models.model_factory_fn import get_generator, init_client_nets, init_server_net
from FedFTG import local_to_global_knowledge_distillation, warmup_generator
from pseudocode import main_pseudocode
from dataset.transforms import *
from dataset.custom_datasets import CustomDataset

from collections import defaultdict, Counter

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


def InforNCE_Loss(anchor, sample, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float).to(anchor.device)
    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix else sample)
    exp_sim = torch.exp(sim)
    target = (exp_sim * pos_mask).sum(dim=1)
    prob = target / (exp_sim.sum(dim=1) + 1e-9)
    log_prob = torch.log(prob)
    loss = -log_prob.mean()
    return loss, sim


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='path to the config file')
    parser.add_argument('--model', type=str, default='resnet12', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='FC100', help='dataset used for training')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes in the dataset')
    parser.add_argument('--net_config', type=dict, help='network configuration')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01, 0.0005, 0.005)')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')

    parser.add_argument('--mode', type=str, default='few-shot', help='few-shot or normal')
    parser.add_argument('--num_train_tasks', type=int, default=1, help='number of meta-training tasks (5)')
    parser.add_argument('--num_test_tasks', type=int, default=10, help='number of meta-test tasks')
    parser.add_argument('--num_true_test_ratio', type=int, default=10, help='number of meta-test tasks (10)')
    parser.add_argument('--fine_tune_steps', type=int, default=5, help='number of meta-learning steps (5)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1, help='number of meta-learning lr (0.05)')
    parser.add_argument('--meta_lr', type=float, default=0.1 / 100, help='number of meta-learning lr (0.05)')
    parser.add_argument('--meta_steps', type=int, default=5000, help='number of maximum communication round')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')

    parser.add_argument("--wv_path", type=str,
                        default="./",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec", help="Name of pretrained word embeddings.")
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

    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default="./checkpoints/",
                        help='Model checkpoint directory path')
    parser.add_argument('--beta', type=float, default=1,  # 0.5
                        help='The parameter for the dirichlet distribution for data partitioning')

    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--gpus', type=str, default='0', help='Visible GPUs for this task')

    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.config_file = "configs/FC100.yaml"
    if args.config_file is not None:
        # Read the configuration from a yaml file
        with open(args.config_file, 'r') as file:
            config_dict = yaml.safe_load(file)

        # Load the dictionary to an into args
        for key, value in config_dict.items():
            setattr(args, key, value)

        return config_dict

    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(args.device)

    return vars(args)


def meta_train_net(args, epoch, net, optimizer, x, y, transform, global_model, fake_data, fake_data_labels, server_embedding, device='cpu'):
    """
    Train a network on a given dataset with meta learning
    :param args: the arguments
    :param epoch: the current epoch
    :param net: the network to train
    :param x: the training data for this client
    :param y: the training labels for this client
    :param transform: the transformation to apply to the data
    :param device: the device to use
    """
    net.train()
    # for params in net.parameters():
    #     params.requires_grad = True
    # if args['optimizer'] == 'adam':
    #     optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['reg'])
    # elif args['optimizer'] == 'amsgrad':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['reg'],
    #                            amsgrad=True)
    # elif args['optimizer'] == 'sgd':
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.05, momentum=0.9,
    #                           weight_decay=args['reg'])
    optimizer.zero_grad()
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    N = args['meta_config']['train_client_class']
    K = args['meta_config']['train_support_num']
    Q = args['meta_config']['train_query_num']

    if args['dataset'] == 'FC100':
        class_dict = fine_split['train']
    elif args['dataset'] == 'miniImageNet':
        class_dict = list(range(64))
    elif args['dataset'] == '20newsgroup':
        class_dict = [1, 5, 10, 11, 13, 14, 16, 18]
    elif args['dataset'] == 'fewrel':
        class_dict = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                      22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                      39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                      59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                      76, 77, 78]
    elif args['dataset'] == 'huffpost':
        class_dict = list(range(20))

    x_sup, y_sup, x_qry, y_qry = sample_few_shot_data(x, y, class_dict, transform, N, K, Q, device=device)
    x_total = torch.cat([x_sup, x_qry], 0).to(device)
    y_total = torch.cat([y_sup, y_qry], 0).long().to(device)

    # Start training
    ############################
    # Fast Adaptation
    ft_net = copy.deepcopy(net)
    ft_optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.05, momentum=0.9, weight_decay=args['reg'])
    for j in range(args['meta_config']['fine_tune_steps']):
        ft_optimizer.zero_grad()
        ft_sup_embedding, ft_sup_transformer_feature, ft_sup_y_hat = ft_net(x_sup)
        loss = loss_ce(ft_sup_y_hat, y_sup)
        loss.backward()
        ft_optimizer.step()
    ############################
    # Meta-Update
    ft_optimizer.zero_grad()
    _, _, meta_query_y_hat = ft_net(x_qry)
    loss = loss_ce(meta_query_y_hat, y_qry)
    loss.backward()

    if args['log_wandb']:
        wandb.log({'Client Query Loss': loss.item()}, step=epoch)

    for param1, param2 in zip(ft_net.parameters(), net.parameters()):
        if param2.grad is not None:
            param2.grad.zero_()  # zero out gradients in model2 to avoid accumulating with existing ones
        if param1.grad is not None:
            param2.grad = param1.grad.clone()

    optimizer.step()
    optimizer.zero_grad()

    # Global to Local Distillation Through Logits Matching
    # out_sup_on_N_class = y_hat[N * K:, transformed_class_index_list]
    # out_sup_on_N_class /= out_sup_on_N_class.sum(-1, keepdim=True)
    # mse = loss_mse(meta_query_y_hat, out_sup_on_N_class) * 0.1
    # loss += mse
    # if args['log_wandb']:
    #     wandb.log({'Global to Local Partial KD Loss': mse.item()}, step=epoch)

    ############################
    # Local to Global Knowledge Distillation Through Feature Matching

    _, client_embedding, y_hat = net(x_total)
    client_embedding = client_embedding.reshape([N, K + Q, -1]).transpose(0, 1)
    #
    # # Maximize the similarity of local and global H
    # # Update the parameter of global model (Main Feature + Full Classifier)
    loss = 0
    global_model.eval()
    with torch.no_grad():
        server_embedding, _, y_hat = global_model(resize(torch.cat([x_sup, x_qry], 0)))
    server_embedding = server_embedding.reshape([N, K + Q, -1]).transpose(0, 1)
    for j in range(K + Q):
        contras_loss, similarity = InforNCE_Loss(server_embedding[j], client_embedding[j])
        loss += contras_loss / K
    if args['log_wandb']:
        wandb.log({'Global to Local Contrastive Loss': loss.item()}, step=epoch)
    loss.backward()
    optimizer.step()

    acc_train = (torch.argmax(y_hat, -1) == y_total).float().mean().item()
    #
    # del latent_embedding, y_hat
    return acc_train


def meta_test_net(args, net, x_test, y_test, transform, device='cpu', test_k=None):
    """
        test a net on a given dataset with meta learning
        :param args: the arguments
        :param net: the network to test
        :param x_test: the test data
        :param y_test: the test labels
        :param transform: the transform to apply to the data
        :param test_k: specific k for the test, if None, use the default test_k
        :param device: the device to use
        """
    N = args['meta_config']['test_client_class']
    K = test_k if test_k else args['meta_config']['test_support_num']
    Q = args['meta_config']['test_query_num']

    if args['dataset'] == 'FC100':
        class_dict = fine_split['test']
    elif args['dataset'] == 'miniImageNet':
        class_dict = list(range(20))
    elif args['dataset'] == '20newsgroup':
        class_dict = [0, 2, 3, 8, 9, 15, 19]
    elif args['dataset'] == 'fewrel':
        class_dict = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]
    elif args['dataset'] == 'huffpost':
        class_dict = list(range(25, 41))
    else:
        raise ValueError('Unknown dataset')

    x_sup, y_sup, x_qry, y_qry = sample_few_shot_data(x_test, y_test, class_dict, transform, N, K, Q, device=device)

    # Testing
    net.eval()
    with torch.no_grad():
        acc = few_shot_logistic_regression(net, x_sup, y_sup, x_qry, y_qry)

    return acc


def sample_few_shot_data(x, y, class_dict, transform, N, K, Q, device='cpu'):
    # Pick N classes
    # Make sure that there are at least K + Q samples for each class
    class_dict = class_dict.copy()
    current_min_size = 0
    while current_min_size < K + Q:
        X_class = []
        classes = np.random.choice(class_dict, N, replace=False).tolist()
        for i in classes:
            X_class.append(x[y == i])
            if X_class[-1].shape[0] < K + Q:
                class_dict.remove(i)
        current_min_size = min([one.shape[0] for one in X_class])

    x_sup = []
    y_sup = []
    # Following labels are never used actually
    x_qry = []
    y_qry = []
    # sample K + Q samples for each class
    for class_index, class_data in zip(classes, X_class):
        sample_idx = np.random.choice(list(range(class_data.shape[0])), K + Q, replace=False).tolist()
        x_sup.append(class_data[sample_idx[:K]])
        x_qry.append(class_data[sample_idx[K:]])
        y_sup.append(torch.ones(K) * class_index)
        y_qry.append(torch.ones(Q) * class_index)

    x_sup = np.concatenate(x_sup, 0)
    x_qry = np.concatenate(x_qry, 0)
    y_sup = torch.cat(y_sup, 0).long()
    y_qry = torch.cat(y_qry, 0).long()

    # Apply the same transformation to the support set and the query set if its image dataset
    if args['dataset'] == 'FC100' or args['dataset'] == 'miniImageNet':
        X_total_transformed_sup = []
        X_total_transformed_query = []
        for i in range(x_sup.shape[0]):
            X_total_transformed_sup.append(transform(x_sup[i]))
        x_sup = torch.stack(X_total_transformed_sup, 0)
        for i in range(x_qry.shape[0]):
            X_total_transformed_query.append(transform(x_qry[i]))
        x_qry = torch.stack(X_total_transformed_query, 0)

    # Finalized meta-learning data
    x_sup = torch.tensor(x_sup).to(device)
    x_qry = torch.tensor(x_qry).to(device)

    return x_sup, y_sup, x_qry, y_qry


def clients_meta_train(nets, opts, args, epoch, net_data_idx_map, x_train, y_train, x_test, y_test, train_transform, test_transform, global_model, fake_data, fake_data_labels, server_embedding, device="cpu"):
    acc_list = []

    for net_id, net in nets.items():

        # Each client has a list of data samples assigned to it in the format of indices
        data_idxs = net_data_idx_map[net_id]

        # get the private data for the client
        X_train_client = x_train[data_idxs]
        y_train_client = y_train[data_idxs]

        # Training
        accs = []
        for _ in range(args['meta_config']['num_train_tasks']):
            accs.append(meta_train_net(args, epoch, net, opts[net_id], X_train_client, y_train_client, train_transform, global_model, fake_data, fake_data_labels, server_embedding, device=device))

        # Local Testing
        accs_test = []
        for _ in range(args['num_test_tasks']):
            acc_test = meta_test_net(args, net, x_test, y_test, test_transform, device)
            accs_test.append(acc_test)

        acc_list.append(np.mean(accs_test))

    logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
    if args['log_wandb']:
        wandb.log({f'{args["n_parties"]} clients Train Accuracy': np.mean(acc_list)}, step=epoch)
        wandb.log({f'{args["n_parties"]} clients Best Train Accuracy': np.max(acc_list)}, step=epoch)

    if args['alg'] == 'local_training':
        logger.info("avg test acc %f" % np.mean(acc_list))
        logger.info("std acc %f" % np.std(acc_list))


def init_wandb(args):
    wandb.init(sync_tensorboard=False,
               project="BlackBoxFL",
               config=args,
               job_type="CleanRepo",
               name=args['wandb_name'] if args['wandb_name'] else None,
               )


def run_experiment(args):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['cuda_gpu'])
    # Set random seed
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    mkdirs(args['log_dir'])
    mkdirs(args['checkpoint_dir'])
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up the logger
    if args['log_file_name'] is None:
        args['log_file_name'] = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args['log_file_name'] + '.log'
    logging.basicConfig(
        filename=os.path.join(args['log_dir'], log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    if args['log_wandb']:
        init_wandb(args)

    logger.info(args)
    device = args['device']
    logger.info(f'Using device {args["device"]}')

    # Print Algorithm Pseudocode
    main_pseudocode(args, logger)

    # torch.backends.cudnn.deterministic = True

    logger.info("Partitioning data")
    x_train, y_train, x_test, y_test, net_data_idx_map, client_class_cnt = partition_data(
        args['dataset'], args['data_dir'], args['partition'], args['n_parties'], beta=args['beta'])
    # Initialize FL client pattern
    n_party_per_round = int(args['n_parties'] * args['sample_fraction'])
    party_list = [i for i in range(args['n_parties'])]
    party_list_rounds = []
    if n_party_per_round != args['n_parties']:
        for i in range(args['meta_steps']):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args['meta_steps']):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    logger.info("Initializing clients")
    clients, local_model_meta_data, layer_type = init_client_nets(args['n_parties'], args)
    global_model, global_model_meta_data, global_layer_type = init_server_net(args)

    optimizers = {}
    for client_id, client in clients.items():
        for param in client.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(client.parameters(), lr=args['lr'], weight_decay=args['reg'])
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, client.parameters()), lr=0.05, momentum=0.9,
        #                       weight_decay=args['reg'])
        # if args['optimizer'] == 'adam':
        #     optimizer = optim.Adam(client.parameters(), lr=args['lr'], weight_decay=args['reg'])
        # elif args['optimizer'] == 'amsgrad':
        #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, client.parameters()), lr=args['lr'], weight_decay=args['reg'],
        #                            amsgrad=True)
        # elif args['optimizer'] == 'sgd':
        #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, client.parameters()), lr=0.05, momentum=0.9,
        #                           weight_decay=args['reg'])
        optimizers[client_id] = optimizer

    server_train_transform = server_test_transform = client_train_transform = client_test_transform = None
    if args['net_config']['encoder'] == 'resnet18':
        if args['dataset'] == 'FC100':
            transform = get_fc100_transform()
            server_train_transform = client_train_transform = transform['train_transform']
            server_test_transform = client_test_transform = transform['test_transform']
        elif args['dataset'] == 'miniImageNet':
            transform = get_mini_image_transform()
            server_train_transform = client_train_transform = transform['train_transform']
            server_test_transform = client_test_transform = transform['test_transform']
    elif 'clip' in args['net_config']['encoder']:
        transform = get_vit_224_size_transform(global_model.encoder)
        server_train_transform = transform['train_transform']
        server_test_transform = transform['test_transform']
        transform = get_vit_original_size_transform()
        client_train_transform = transform['train_transform']
        client_test_transform = transform['test_transform']
    else:
        raise ValueError('Unknown encoder')

    train_loaders = {}
    for client_id, client in clients.items():
        # Each client has a list of data samples assigned to it in the format of indices
        data_idxs = net_data_idx_map[client_id]

        # Get the private data for the client
        X_train_client = x_train[data_idxs]
        y_train_client = y_train[data_idxs]

        # Create the private data loader for each client
        train_dataset = CustomDataset(X_train_client, y_train_client, transform=client_train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args['warmup_bs'], shuffle=True, num_workers=4)

        train_loaders[client_id] = train_loader

    test_data_num = len(x_test)
    test_dataset = CustomDataset(x_test, y_test, transform=client_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args['warmup_bs'], shuffle=False, num_workers=4)

    logger.info(">> Warmup Each Clients:")
    client_best_acc = {}
    for client_id, client in clients.items():
        client_best_acc[client_id] = 0
    epoch = 0
    while min(client_best_acc.values()) < args['warmup_acc']:
        epoch += 1
        cur_clients = []
        all_loss = []
        all_acc = []
        for client_id, client in clients.items():
            if client_best_acc[client_id] > args['warmup_acc']:
                continue
            client.train()
            cur_clients.append(client_id)
            client_loss, client_acc = general_one_epoch(client, train_loaders[client_id], optimizers[client_id], device)
            all_loss.append(client_loss)
            all_acc.append(client_acc)
            client_best_acc[client_id] = max(client_best_acc[client_id], client_acc)
        all_loss = [f'{loss:.2f}' for loss in all_loss]
        all_acc = [f'{acc:.2f}' for acc in all_acc]
        results = ' | '.join(f'{c}:({loss},{acc})' for c, loss, acc in zip(cur_clients, all_loss, all_acc))
        logger.info(f">> Epoch {epoch}, Client Warmup Train (Loss,Acc): {results}")
        if args['log_wandb']:
            wandb.log({'Warmup Train Acc': sum(client_best_acc.values()) / args['n_parties']}, step=epoch)

        # Evaluate
        all_loss = []
        all_acc = []
        for client_id, client in clients.items():
            client.eval()
            client_loss, client_acc = general_one_epoch(client, train_loaders[client_id], optimizers[client_id], device)
            all_loss.append(client_loss)
            all_acc.append(client_acc)
        logger.info(f">> Epoch {epoch}, Client Warmup test Loss: {' | '.join(f'{loss:.4f}' for loss in all_loss)}")
        logger.info(f">> Epoch {epoch}, Client Warmup test Acc: {' | '.join(f'{acc:.4f}' for acc in all_acc)}")
        logger.info("-------------------------------------------------------------------------------------------------")
        if args['log_wandb']:
            wandb.log({f'Warmup Test Loss': sum(all_loss) / args['n_parties']}, step=epoch)
            wandb.log({f'Warmup Test Accuracy': sum(all_acc) / args['n_parties']}, step=epoch)


    # set up the generator
    init_g_model = get_generator(**args['generator_config'])

    mkdirs(args['checkpoint_dir'] + 'blackbox/')
    best_acc_dict = Counter()
    best_confident_acc = 0

    for step in range(epoch, epoch + args['meta_steps']):
        logger.info(f'>> Current Round: {step}')
        party_list_this_round = party_list_rounds[step]

        global_w = global_model.state_dict()

        clients_this_round = {k: clients[k] for k in party_list_this_round}

        # The total number of data points in the current round of selected clients
        total_data_points = sum([len(net_data_idx_map[r]) for r in range(args['n_parties'])])
        # Calculate the percentage of data points for each client
        fed_avg_weight = [len(net_data_idx_map[r]) / total_data_points for r in range(args['n_parties'])]

        # Test the global model
        for k in args['meta_config']['test_k']:
            accs = []
            for epoch_test in range(args['num_test_tasks'] * args['num_true_test_ratio']):
                acc, max_value, index = meta_test_net(args, global_model, x_test, y_test, server_test_transform, device, k)
                accs.append(acc)

            global_acc = np.mean(accs)

            if global_acc > best_acc_dict[k]:
                best_acc_dict[k] = global_acc
            logger.info(f'>> Global Model few-shot {k} Meta-Test accuracy: {global_acc:.4f} Best Acc: {best_acc_dict[k]:.4f}')
            if args['log_wandb']:
                wandb.log({f'Global K={k} Test Accuracy': global_acc}, step=step)

        if args['use_KD_Generator']:
            if step == args['warmup_epoch']:
                print('Warmup the generator')
                KD_config = args['KD_config']
                warmup_generator(
                    args,
                    global_model,
                    init_g_model,
                    clients_this_round.values(),
                    np.array(client_class_cnt),
                    KD_config['gen_model_lr'],
                    KD_config['batch_size'],
                    100,
                    KD_config['n_cls'],
                    device=args['device']
                )

            # Local to Global: generator knowledge distillation from local to global model
            KD_config = args['KD_config']
            KD_config['glb_model_lr'] *= KD_config['lr_decay_per_epoch']
            KD_config['gen_model_lr'] *= KD_config['lr_decay_per_epoch']

            avg_model_ft, fake_data, fake_data_labels, server_embedding = \
                local_to_global_knowledge_distillation(args, step, global_model, init_g_model, clients_this_round.values(),
                                                       np.array(client_class_cnt), device=args['device'], **KD_config)
            global_model = avg_model_ft
        else:
            fake_data, fake_data_labels, server_embedding = None, None, None

        ###############################################
        # Meta-training on each client's end
        clients_meta_train(
            clients_this_round,
            optimizers,
            args,
            step,
            net_data_idx_map,
            x_train,
            y_train,
            x_test,
            y_test,
            client_train_transform,
            client_test_transform,
            global_model, fake_data, fake_data_labels, server_embedding,
            device=device)

        # save model example
        # torch.save(global_model.state_dict(),
        #            args['checkpoint_dir'] + 'blackbox/' + 'globalmodel' + args['log_file_name'] + '.pth')
        # torch.save(clients[0].state_dict(),
        #            args['checkpoint_dir'] + 'blackbox/' + 'localmodel0' + args['log_file_name'] + '.pth')


if __name__ == '__main__':
    args = init_args()
    run_experiment(args)