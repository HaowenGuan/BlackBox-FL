import shutil

import torch
import torch.optim as optim
import argparse
import copy
import datetime
import random
import yaml
import sys
from tqdm import tqdm

import wandb as wandb
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from few_shot_learning import few_shot_prototype, few_shot_logistic_regression
from training import general_one_epoch

from models.feature_extractor.fe_utils import DiffAugment


# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
#
# resize = transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)


from PIL import Image

from models.model import *
from utils import *
import warnings
from models.model_factory_fn import get_generator, init_client_nets, init_server_net
from FedFTG import *
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


def meta_train_net(args, net, optimizer, x, y, transform, device='cpu'):
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
    # loss_ce = nn.CrossEntropyLoss()
    # loss_mse = nn.MSELoss()

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
    else:
        raise ValueError('Unknown dataset')

    x_sup, y_sup, x_qry, y_qry = sample_few_shot_data(x, y, class_dict, transform, N, K, Q, device=device)
    # x_total = torch.cat([x_sup, x_qry], 0).to(device)
    # y_total = torch.cat([y_sup, y_qry], 0).long().to(device)

    # Meta Training
    ############################
    net.train()
    optimizer.zero_grad()
    loss, acc = few_shot_prototype(net, x_sup, y_sup, x_qry, y_qry)
    loss.backward()
    optimizer.step()

    return acc


def meta_test_net(args, net, x_test, y_test, transform, ft_approach='prototype', test_k=None, device='cpu'):

    """
        test a net on a given dataset with meta learning
        :param args: the arguments
        :param net: the network to test
        :param x_test: the test data
        :param y_test: the test labels
        :param transform: the transform to apply to the data
        :param device: the device to use
        :param test_k: specific k for the test, if None, use the default test_k
        @param args: arguments
        @param net: network to test
        @param x_test: test data
        @param y_test: test label
        @param transform: transform to apply to the data
        @param ft_approach: few shot testing approach, 'prototype' or 'classic'
        @param test_k: specific k for the test, if None, use the default test_k
        @param device:
        @return:
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

    # Fine-tune with meta-test
    CELoss = nn.CrossEntropyLoss()
    test_net = copy.deepcopy(net).to(device)
    test_net.train()
    meta_config = args['meta_config']
    optimizer = optim.Adam(test_net.parameters(), lr=meta_config['test_ft_lr'], weight_decay=args['reg'])
    test_accs = {}
    for step in range(meta_config['test_ft_steps']):
        if step % 10 == 0:
            test_net.eval()
            with torch.no_grad():
                loss, acc = few_shot_prototype(test_net, x_sup, y_sup, x_qry, y_qry)
            test_accs[step] = acc
            test_net.train()
        optimizer.zero_grad()
        if ft_approach == 'prototype':
            aug_x_sup = DiffAugment(x_sup, meta_config['aug_types'], meta_config['aug_prob'], detach=True)
            loss, acc = few_shot_prototype(test_net, x_sup, y_sup, aug_x_sup, y_sup)
        else:  # classic
            _, logits = test_net(x_sup)
            loss = CELoss(F.softmax(logits, dim=1), y_sup)
        loss.backward()
        optimizer.step()
    # Final meta-test
    test_net.eval()
    with torch.no_grad():
        loss, acc = few_shot_prototype(test_net, x_sup, y_sup, x_qry, y_qry)
    test_accs[meta_config['test_ft_steps']] = acc
    # acc = few_shot_logistic_regression(net, x_sup, y_sup, x_qry, y_qry)

    return test_accs


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
    y_sup = torch.cat(y_sup, 0).long().to(device)
    y_qry = torch.cat(y_qry, 0).long().to(device)

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


def clients_meta_train(args, clients, opts, x_train_clients, y_train_clients, x_test, y_test, train_transform, test_transform, device="cpu"):
    meta_train_acc = []
    meta_test_acc = defaultdict(list)

    for net_id, client in clients.items():
        logger.info(f'Meta Training and Testing: Client {net_id}')

        # Meta Training
        acc_train = []
        for _ in tqdm(range(args['meta_config']['num_train_tasks'])):
            acc_train.append(meta_train_net(args, client, opts[net_id], x_train_clients[net_id], y_train_clients[net_id], train_transform, device=device))
        meta_train_acc.append(np.mean(acc_train))

        # Meta Testing
        acc_test = defaultdict(list)
        for _ in tqdm(range(args['meta_config']['num_test_tasks'] // len(clients))):
            acc_dict = meta_test_net(
                args=args,
                net=client,
                x_test=x_test,
                y_test=y_test,
                transform=test_transform,
                ft_approach=args['meta_config']['test_ft_approach'],
                device=device,
            )
            for step, acc in acc_dict.items():
                acc_test[step].append(acc)
        for step, acc_list in acc_test.items():
            meta_test_acc[step].append(np.mean(acc_list))

    logger.info('Meta Train Accuracy: ' + ' | '.join(['{:.4f}'.format(acc) for acc in meta_train_acc]))
    for step, acc_list in meta_test_acc.items():
        logger.info(f'Meta Test Accuracy at FT step {step}: ' + ' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

    if args['log_wandb']:
        wandb.log({f'Clients Meta Train Accuracy': np.mean(meta_train_acc)})
        for step, acc_list in meta_test_acc.items():
            wandb.log({f'Clients Meta Test Accuracy at FT {step}': np.mean(acc_list)})


def copy_and_rename(src_path, dest_path, new_name):
    # Copy the file
    shutil.copy(src_path, dest_path)
    old_name = src_path.split('/')[-1]

    # Rename the copied file
    shutil.move(f"{dest_path}{old_name}", f"{dest_path}{new_name}")


def init_wandb(args):
    wandb.init(
        sync_tensorboard=False,
        project="BlackBoxFL",
        config=args,
        job_type="CleanRepo",
        name=args['wandb_name'] if args['wandb_name'] else None,
    )
    for _ in range(args['wandb_start_step']):
        wandb.log({})


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
    x_train, y_train, x_test, y_test, net_data_idx_map, client_class_num = partition_data(
        args['dataset'], args['data_dir'], args['partition'], args['n_parties'], beta=args['beta'], seed=args['seed'])
    client_class_num = np.array(client_class_num)

    logger.info("Initializing clients and server models")
    clients, local_model_meta_data, layer_type = init_client_nets(args['n_parties'], args)
    server, global_model_meta_data, global_layer_type = init_server_net(args)

    client_optimizers = {}
    for client_id, client in clients.items():
        for param in client.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(client.parameters(), lr=args['lr'], weight_decay=args['reg'])
        client_optimizers[client_id] = optimizer
    for param in server.parameters():
        if not param.requires_grad:
            print(param)
    server_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, server.parameters()), lr=args['server_lr'], weight_decay=args['reg'])


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
        if args['net_config']['encoder'] == 'vit_base_patch16_clip_224.openai':
            transform = get_vit_224_size_transform(server.encoder)
            server_train_transform = transform['train_transform']
            server_test_transform = transform['test_transform']
        elif args['net_config']['encoder'] == 'clip_vit_tiny':
            transform = get_vit_original_size_transform()
            server_train_transform = transform['train_transform']
            server_test_transform = transform['test_transform']
        transform = get_vit_original_size_transform()
        client_train_transform = transform['train_transform']
        client_test_transform = transform['test_transform']
    else:
        raise ValueError('Unknown encoder')

    x_train_clients = {}
    y_train_clients = {}
    train_loaders = {}
    for client_id, client in clients.items():
        # Each client has a list of data samples assigned to it in the format of indices
        data_idxs = net_data_idx_map[client_id]

        # Get the private data for the client
        x_train_client = x_train[data_idxs]
        y_train_client = y_train[data_idxs]
        logger.info(f'>> Client {client_id} owns {len(x_train_client)} training samples.')
        x_train_clients[client_id] = x_train_client
        y_train_clients[client_id] = y_train_client

        # Create the private data loader for each client
        train_dataset = CustomDataset(x_train_client, y_train_client, transform=client_train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args['warmup_bs'], shuffle=True, num_workers=4)

        train_loaders[client_id] = train_loader

    test_data_num = len(x_test)
    test_dataset = CustomDataset(x_test, y_test, transform=client_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args['warmup_bs'], shuffle=False, num_workers=4)


    ######################################## Warmup Clients Model ########################################
    epoch = 0
    if args['load_clients'] is not None:
        logger.info(f'>> Loading clients checkpoint from {args["load_clients"]}')
        for client_id, client in clients.items():
            client.load_state_dict(torch.load(args['load_clients'] + f'{client_id}.pth'))
    if args['warmup_clients']:
        logger.info(">> Warmup Each Clients:")
        client_best_acc = {}
        for client_id, client in clients.items():
            client_best_acc[client_id] = 0

        while min(client_best_acc.values()) < args['warmup_acc']:
            cur_clients = []
            all_loss = []
            all_acc = []
            for client_id, client in clients.items():
                if client_best_acc[client_id] > args['warmup_acc']:
                    continue
                client.train()
                cur_clients.append(client_id)
                client_loss, client_acc = general_one_epoch(client, train_loaders[client_id], client_optimizers[client_id], device)
                all_loss.append(client_loss)
                all_acc.append(client_acc)
                client_best_acc[client_id] = max(client_best_acc[client_id], client_acc)
            all_loss = [f'{loss:.2f}' for loss in all_loss]
            all_acc = [f'{acc:.2f}' for acc in all_acc]
            results = ' | '.join(f'{c}:({loss},{acc})' for c, loss, acc in zip(cur_clients, all_loss, all_acc))
            logger.info(f">> Epoch {epoch}, Client Warmup Train (Loss,Acc): {results}")
            if args['log_wandb']:
                wandb.log({'Warmup Train Acc': sum(client_best_acc.values()) / args['n_parties']}, step=epoch)
            epoch += 1

        if args['save_clients']:
            warmup_folder = args['checkpoint_dir'] + 'warmup/'
            mkdirs(warmup_folder)
            for client_id, client in clients.items():
                logger.info(f'>> Saving warmup checkpoint for client {client_id}')
                torch.save(client.state_dict(), warmup_folder + f'{client_id}.pth')
            shutil.copy(os.path.join(args['log_dir'], log_path), warmup_folder)

    # Evaluation After Warm Up
    all_loss = []
    all_acc = []
    cur_clients = []
    for client_id, client in clients.items():
        cur_clients.append(client_id)
        client.eval()
        client_loss, client_acc = general_one_epoch(client, train_loaders[client_id], None, device)
        all_loss.append(client_loss)
        all_acc.append(client_acc)
    all_loss = [f'{loss:.2f}' for loss in all_loss]
    all_acc = [f'{acc:.2f}' for acc in all_acc]
    results = ' | '.join(f'{c}:({loss},{acc})' for c, loss, acc in zip(cur_clients, all_loss, all_acc))
    logger.info(f">> Epoch {epoch}, Client Warmup Train (Loss,Acc): {results}")
    logger.info("-------------------------------------------------------------------------------------------------")


    ######################################## Set Up global Generators ########################################
    if args['use_KD_Generator_one']:
        KD_config = args['KD_config']
        generator = get_generator(**args['generator_config'])
        generator.to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=KD_config['gen_model_lr'])
        best_loss = 1e9

        # Load pre-trained generator if specified
        if args['load_generator'] is not None:
            checkpoint = torch.load(args['load_generator'])
            generator.load_state_dict(checkpoint['model_state_dict'])
            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=KD_config['gen_model_lr'])
            generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # for g in generator_optimizer.param_groups:
            #     g['lr'] = KD_config['gen_model_lr']
            # best_loss = checkpoint['best_loss']
            logger.info(f'>> Loaded generator from {args["load_generator"]}')
        if args['warmup_generator']:
            logger.info(f'>> Start Warmup generator')
            for _ in range(6):
                loss = local_to_global_knowledge_distillation(
                    args=args,
                    server=server,
                    server_optimizer=server_optimizer,
                    clients=clients,
                    generator=generator,
                    generator_optimizer=generator_optimizer,
                    client_class_num=client_class_num,
                    batch_size=KD_config['batch_size'],
                    iterations=5000,
                    loss_function=KD_config['loss_function'],
                    warmup=True,
                    device='cuda',
                )

                # Save the generator
                if args['save_generator'] is not None and loss < best_loss:
                    best_loss = loss
                    generator_folder = args['checkpoint_dir'] + 'generator/'
                    mkdirs(generator_folder)
                    logger.info(f'>> Saving warmup global generator, current loss {loss}')
                    torch.save({
                        'model_state_dict': generator.state_dict(),
                        'optimizer_state_dict': generator_optimizer.state_dict(),
                        'best_loss': best_loss,
                        }, generator_folder + args['save_generator'])
                    copy_and_rename(os.path.join(args['log_dir'], log_path), generator_folder, 'global_generator.log')
                    logger.info(f'>> Generator checkpoint saved to {generator_folder + args["save_generator"]}')
    ######################################## Set Up independent Generators ########################################
    elif args['use_KD_Generator']:
        KD_config = args['KD_config']
        generators = {i: get_generator(**args['generator_config']) for i in range(args['n_parties'])}
        for generator in generators.values():
            generator.to(device)
        generator_optimizers = {i: torch.optim.Adam(generator.parameters(), lr=KD_config['gen_model_lr'])
                                 for i, generator in generators.items()}

        # Load pre-trained generator if specified
        if args['load_generator'] is not None:
            logger.info(f'>> Loading generator from {args["load_generator"]}')
            for i, generator in generators.items():
                checkpoint = torch.load(args['load_generator'] + f'{i}.pth')
                generator.load_state_dict(checkpoint['model_state_dict'])
                generator_optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])

        removing_list = []
        for i, generator in generators.items():
            logger.info(f'Warmup client {i}\'s generator')
            removing = train_generator(
                args,
                server,
                generators,
                {i: clients[i]},
                generator_optimizers,
                client_class_num[i:i + 1],
                KD_config['batch_size'],
                iterations=500,
                n_cls=KD_config['n_cls'],
                early_stop=KD_config['warmup_early_stop'],
                device=args['device']
            )
            if removing:
                removing_list.append(i)

        for i in removing_list:
            del generators[i]
            del clients[i]
            logger.info(f'Client {i} is removed due to bad performance of generator')

        # Save the generator
        if args['save_generator']:
            generator_folder = args['checkpoint_dir'] + 'generator/'
            mkdirs(generator_folder)
            for g_id, generator in generators.items():
                logger.info(f'>> Saving warmup generator for client {g_id}')
                torch.save({
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': generator_optimizers[g_id].state_dict(),
                }, generator_folder + f'{g_id}.pth')
            shutil.copy(os.path.join(args['log_dir'], log_path), generator_folder)
    ######################################## Set Up Fake Distill Dataset ########################################
    elif args['use_KD_dataset']:
        logger.info(f'>> Warmup Distill Dataset')
        fake_data = torch.randn((len(fine_split['train']), 3, 32, 32), device=device, requires_grad=True)
        fake_target = torch.Tensor(fine_split['train']).long().to(device)
        fake_data_optimizer = torch.optim.Adam([fake_data], lr=0.001)

        # Load fake data checkpoint if specified
        if args['load_fake_data'] is not None:
            logger.info(f'>> Loading fake data from {args["load_fake_data"]}')
            checkpoint = torch.load(args['load_fake_data'] + 'fake_data.pth', map_location=device)
            fake_data = checkpoint['fake_data'].to(device).requires_grad_(True)
            fake_data_optimizer = torch.optim.Adam([fake_data], lr=0.001)
            fake_data_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            local_to_global_knowledge_dataset_distillation(
                args,
                server,
                server_optimizer,
                clients,
                fake_data,
                fake_data_optimizer,
                fake_target,
                client_class_num,
                iterations=1000,
                device='cuda',
                warmup=True,
            )

        # Save fake data
        if args['save_fake_data']:
            fake_data_folder = args['checkpoint_dir'] + 'fake_data/'
            mkdirs(fake_data_folder)
            torch.save({
                'fake_data': fake_data.detach().cpu(),  # Detach and move to CPU to save
                'optimizer_state_dict': fake_data_optimizer.state_dict(),
            }, fake_data_folder + 'fake_data.pth')
            shutil.copy(os.path.join(args['log_dir'], log_path), fake_data_folder)
            logger.info(f'>> Successfully Saved Fake Data')
    ######################################## --------------------------- ########################################

    best_acc_dict = Counter()

    for step in range(args['meta_steps']):
        logger.info(f'>> Current Round: {step}')

        # The total number of data points in the current round of selected clients
        total_data_points = sum([len(net_data_idx_map[r]) for r in range(args['n_parties'])])
        # Calculate the percentage of data points for each client
        fed_avg_weight = [len(net_data_idx_map[r]) / total_data_points for r in range(args['n_parties'])]

        if args['use_KD_Generator_one']:
            logger.info(f'>> Blackbox Federated Learning with One Generator')
            KD_config = args['KD_config']

            blackbox_federated_learning_with_generator(
                args=args,
                server=server,
                server_optimizer=server_optimizer,
                clients=clients,
                client_optimizers=client_optimizers,
                generator=generator,
                generator_optimizer=generator_optimizer,
                client_class_num=client_class_num,
                batch_size=KD_config['batch_size'],
                iterations=KD_config['num_of_kd_steps'],
                loss_function=KD_config['loss_function'],
                warmup=False,
                use_md_loss=KD_config['use_md_loss'],
                device='cuda',
            )
        elif args['use_KD_dataset']:
            logger.info(f'>> Local to Global Using Distill Dataset')
            local_to_global_knowledge_dataset_distillation(
                args,
                server,
                server_optimizer,
                clients,
                fake_data,
                fake_data_optimizer,
                fake_target,
                client_class_num,
                iterations=100,
                device='cuda',
            )

        # Test the global model
        if args['use_server_model']:
            for k in args['meta_config']['test_k']:
                accs = []
                logger.info(f'>> Server Model {k}-shot Meta-Testing')
                for _ in tqdm(range(args['meta_config']['num_test_tasks'])):
                    acc = meta_test_net(
                        args=args,
                        net=server,
                        x_test=x_test,
                        y_test=y_test,
                        transform=server_test_transform,
                        ft_approach=args['meta_config']['test_ft_approach'],
                        test_k=k,
                        device=device,
                    )
                    accs.append(acc[args['meta_config']['test_ft_steps']])

                global_acc = np.mean(accs)

                if global_acc > best_acc_dict[k]:
                    best_acc_dict[k] = global_acc
                logger.info(
                    f'>> Server Model {k}-shot Meta-Test Accuracy: {global_acc:.4f} Best Acc: {best_acc_dict[k]:.4f}')
                if args['log_wandb']:
                    wandb.log({f'Global K={k} Test Accuracy': global_acc})

        # Meta-training on each client's end
        clients_meta_train(
            args,
            clients,
            client_optimizers,
            x_train_clients,
            y_train_clients,
            x_test,
            y_test,
            client_train_transform,
            client_test_transform,
            device=device)

    if args['save_clients']:
        meta_learning_folder = args['checkpoint_dir'] + 'meta_learning/'
        mkdirs(meta_learning_folder)
        for client_id, client in clients.items():
            logger.info(f'>> Saving warmup checkpoint for client {client_id}')
            torch.save(client.state_dict(), meta_learning_folder + f'{client_id}.pth')
        shutil.copy(os.path.join(args['log_dir'], log_path), meta_learning_folder)

if __name__ == '__main__':
    args = init_args()
    run_experiment(args)