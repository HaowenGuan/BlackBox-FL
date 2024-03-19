from models import resnet
# from models import simple_models
# from models import vgg
from models import generators


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
