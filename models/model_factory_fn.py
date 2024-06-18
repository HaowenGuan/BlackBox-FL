from models import resnet
# from models import simple_models
# from models import vgg
from models import generators
from models.model import *
import copy


def get_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet.resnext18(**kwargs)
    else:
        raise ValueError('Wrong model name.')


def get_generator(model_name, **kwargs):
    if model_name == 'CGeneratorA':
        return generators.CGeneratorA(**kwargs)
    else:
        print(model_name)
        raise ValueError('Wrong model name.')


def init_client_nets(num_clients, args):
    """
    Initialize the networks for each client
    """
    nets = {net_i: None for net_i in range(num_clients)}
    net_config = args['net_config']
    client_classes = args['meta_config']['train_client_class']
    server_classes = args['num_classes']

    if args['mode'] == 'few-shot':
        if args['dataset'] == 'FC100' or args['dataset'] == 'miniImageNet':
            model = ClientModel(net_config['encoder'], net_config['total_class'], 768)
        else:
            model = LSTMAtt(WORDEBD(args['finetune_ebd']), net_config['out_dim'], client_classes, server_classes, args)

        for net_i in range(num_clients):
            net = copy.deepcopy(model)
            net.to(args['device'])
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_server_net(args):
    """
    Initialize the networks for each client
    """
    net_config = args['net_config']
    client_classes = args['meta_config']['train_client_class']
    server_classes = args['num_classes']

    if args['dataset'] == 'FC100' or args['dataset'] == 'miniImageNet':
        model = ServerModel(**net_config)
    else:
        model = LSTMAtt(WORDEBD(args['finetune_ebd']), net_config['out_dim'], client_classes, server_classes, args)
    model.to(args['device'])

    model_meta_data = []
    layer_type = []
    for (k, v) in model.state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return model, model_meta_data, layer_type