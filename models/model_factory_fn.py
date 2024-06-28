from models import resnet
# from models import simple_models
# from models import vgg
from models import generators
from models.model import *
import copy
import torchsummary


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
            model.to(args['device'])
            size = get_model_size(model)
            print(f'Client model {net_config["encoder"]} size: {size:.3f}MB')
            # torchsummary.summary(model, input_size=(3, 32, 32))
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
        model.to(args['device'])
        size = get_model_size(model)
        # torchsummary.summary(model, input_size=(3, 224, 224))
        print(f'Server model {net_config["encoder"]} size: {size:.3f}MB')
    else:
        model = LSTMAtt(WORDEBD(args['finetune_ebd']), net_config['out_dim'], client_classes, server_classes, args)
    model.to(args['device'])

    model_meta_data = []
    layer_type = []
    for (k, v) in model.state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return model, model_meta_data, layer_type


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb