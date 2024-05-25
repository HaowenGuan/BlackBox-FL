from models import resnet
# from models import simple_models
# from models import vgg
from models import generators
from models.model import *


def get_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet.resnet18(**kwargs)
    else:
        raise ValueError('Wrong model name.')


def get_generator(model_name, **kwargs):
    if model_name == 'CGeneratorA':
        return generators.CGeneratorA(**kwargs)
    else:
        print(model_name)
        raise ValueError('Wrong model name.')


def init_nets(net_configs, num_clients, args):
    """
    Initialize the networks for each client
    """
    nets = {net_i: None for net_i in range(num_clients)}
    server_classes = args.num_classes
    if args.mode == 'few-shot':
        client_classes = args.N * 4

    if args.mode == 'few-shot' and args.method == 'new':
        if args.dataset == '20newsgroup':
            ebd = WORDEBD(args.finetune_ebd)
        for net_i in range(num_clients):
            if args.dataset == 'FC100' or args.dataset == 'miniImageNet':
                net = ImageModel(args.model, args.out_dim, client_classes, server_classes, net_configs, args)
            else:
                net = LSTMAtt(WORDEBD(args.finetune_ebd), args.out_dim, client_classes, server_classes, args)
            net.to(args.device)
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type