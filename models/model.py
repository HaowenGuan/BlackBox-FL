import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from models.resnetcifar import ResNet18_cifar10, ResNet50_cifar10
from torch.distributions import Bernoulli
from torchtext.vocab import GloVe
from embedding.meta import RNN
from torch.hub import load_state_dict_from_url
from models.feature_extractor.vision_transformer import VisionTransformer
from functools import partial
import timm
from lora import LoRA_ViT_timm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm + 1e-9)
    return out


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).to(x.device)
            # print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            # print (block_mask.size())
            # print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x


    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().to(mask.device)
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long().to(mask.device), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask
# import pytorch_lightning as pl

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            #self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet12(keep_prob=1.0, avg_pool=False, drop_rate=0.0,**kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool,drop_rate=drop_rate, **kwargs)
    return model


class MLP_header(nn.Module):
    def __init__(self, ):
        super(MLP_header, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        # projection
        # self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class FcNet(nn.Module):
    """
    Fully connected network for MNIST classification
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(self.output_dim)

        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i + 1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )

        self.__init_net_weights__()

    def __init_net_weights__(self):

        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply ReLU on the final layer
            if i < (len(self.layers) - 1):
                x = nn.ReLU(x)

            # if i < (len(self.layers) - 1):  # No dropout on output layer
            #     x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x


class FCBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGConvBlocks(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_classes=10):
        super(VGGConvBlocks, self).__init__()
        self.features = features
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FCBlockVGG(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlockVGG, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.input_dim = input_dim
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # print(x.shape,'--------')
        x = x.view(-1, self.input_dim)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.relu(out)
        # out = self.pool(out)
        # out = self.conv2(out)
        # out = self.relu(out)
        # out = self.pool(out)
        # out = out.view(-1, 16 * 5 * 5)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# a simple perceptron model for generated 3D data
class PerceptronModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(PerceptronModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


class SimpleCNNMNIST_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return x, 0, y


class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############## LeNet for MNIST ###################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x


### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self, output_dim=10):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNNCeleba(nn.Module):
    def __init__(self):
        super(ModerateCNNCeleba, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 4096)
        x = self.fc_layer(x)
        return x


class ModerateCNNMNIST(nn.Module):
    def __init__(self):
        super(ModerateCNNMNIST, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x


class ModelFedCon(nn.Module):

    def __init__(self, encoder, out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        if encoder == "resnet50-cifar10" or encoder == "resnet50-cifar100" or encoder == "resnet50-smallkernel" or encoder == "resnet50":
            temp_model = ResNet50_cifar10()
            self.features = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "resnet18-cifar10" or encoder == "resnet18":
            temp_model = ResNet18_cifar10()
            self.features = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "mlp":
            self.features = MLP_header()
            num_feature = 512
        elif encoder == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_feature = 84
        elif encoder == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_feature = 84

        # summary(self.features.to('cuda:0'), (3,32,32))
        # print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_feature, num_feature)
        self.l2 = nn.Linear(num_feature, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_base_model(self, model_name):
        try:
            model = self.model_dict[model_name]
            # print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        # print("h before:", h)
        # print("h size:", h.size())
        h = h.squeeze()
        # print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y


class ModelFedCon_noheader(nn.Module):

    def __init__(self, encoder, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()

        if encoder == "resnet50":
            temp_model = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "resnet18":
            temp_model = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "resnet50-cifar10" or encoder == "resnet50-cifar100" or encoder == "resnet50-smallkernel":
            temp_model = ResNet50_cifar10()
            self.features = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "resnet18-cifar10":
            temp_model = ResNet18_cifar10()
            self.features = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "mlp":
            self.features = MLP_header()
            num_feature = 512
        elif encoder == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_feature = 84
        elif encoder == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_feature = 84

        # summary(self.features.to('cuda:0'), (3,32,32))
        # print("features:", self.features)
        # projection MLP
        # self.l1 = nn.Linear(num_feature, num_feature)
        # self.l2 = nn.Linear(num_feature, out_dim)

        # last layer
        self.l3 = nn.Linear(num_feature, n_classes)

    def _get_base_model(self, model_name):
        try:
            model = self.model_dict[model_name]
            # print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        # print("h before:", h)
        # print("h size:", h.size())
        h = h.squeeze()
        # print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        y = self.l3(h)
        return h, h, y


class ServerModel(nn.Module):

    def __init__(self, encoder, client_class, total_class, pretrained):
        """
        Initialize the image model
        Args:
            encoder: str, encoder name
            client_class: int, number of classes
        """
        super(ServerModel, self).__init__()
        self.transform = None

        if encoder == "resnet50-cifar10" or encoder == "resnet50-cifar100" or encoder == "resnet50-smallkernel" or encoder == "resnet50":
            temp_model = ResNet50_cifar10()
            self.encoder = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "resnet18-cifar10" or encoder == "resnet18":
            print("Using resnet18")
            temp_model = ResNet18_cifar10()
            if pretrained:
                state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
                del state_dict['conv1.weight']
                logs = temp_model.load_state_dict(state_dict, strict=False)
                print(logs)
            self.encoder = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "mlp":
            self.encoder = MLP_header()
            num_feature = 512
        elif encoder == 'simple-cnn':
            self.encoder = SimpleCNN_header(input_dim=(16 * 5 * 5),
                                             hidden_dims=[120, 84], output_dim=client_class)
            num_feature = 84
        elif encoder == 'simple-cnn-mnist':
            self.encoder = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=client_class)
            num_feature = 84
        elif encoder == 'resnet12':
            self.encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2)
            #num_feature=2560
            num_feature=640
        elif encoder == 'vit_base_patch16_clip_224.openai':
            self.transform = transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)
            clip = timm.create_model(encoder,
                                          pretrained=pretrained,
                                          img_size=224,
                                          num_classes=0).eval()
            # We freeze the weights of the ViT inside this model
            # do not set all parameters to require_grad when using server model
            self.encoder = LoRA_ViT_timm(vit_model=clip, r=4, alpha=4, num_classes=0)
            num_feature = 768
        elif encoder == 'clip_vit_tiny':
            self.encoder = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            num_feature = 192
        else:
            raise "Unknown encoder"

        # trainable parameter for cosine similarity
        self.tau = nn.Parameter(torch.tensor(1.0))

        # projection MLP
        self.all_classify = nn.Linear(num_feature, total_class)


    def forward(self, x_input):
        """
        :param x_input: input image
        """
        if self.transform:
            x_input = self.transform(x_input)
        ebd = self.encoder(x_input)
        # remove all dimensions with size 1
        b, c = ebd.size(0), ebd.size(1)
        h = ebd.squeeze().view(b, c)
        y = self.all_classify(h)
        return h, y


class ClientModel(nn.Module):

    def __init__(self, encoder, num_classes, server_feature):
        """
        Initialize the image model
        Args:
            encoder: str, encoder name
            num_classes: int, number of classes
        """
        super(ClientModel, self).__init__()

        if encoder == "resnet18":
            print("Using resnet18")
            temp_model = ResNet18_cifar10()
            self.encoder = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif 'clip' in encoder:
            self.encoder = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            num_feature = 192
        else:
            raise "Unknown encoder"

        # trainable parameter for cosine similarity
        self.tau = nn.Parameter(torch.tensor(1.0))

        # Docking layer
        self.docking = nn.Linear(num_feature, server_feature)

        # projection MLP
        self.output = nn.Linear(num_feature, num_classes)


    def forward(self, x_input, use_docking=True):
        """
        :param x_input: input image
        :param use_docking: if True, use docking layer to project the feature, else use the original feature
        """
        ebd = self.encoder(x_input)
        # remove all dimensions with size 1
        b, c = ebd.size(0), ebd.size(1)
        h = ebd.squeeze().view(b, c)
        y = self.output(h)
        if use_docking:
            h = self.docking(h)
        return h, y


class WORDEBD(nn.Module):
    '''
        An embedding layer that maps the token id into its corresponding word
        embeddings. The word embeddings are kept as fixed once initialized.
    '''

    def __init__(self, finetune_ebd):
        super(WORDEBD, self).__init__()
        #vectors = Vectors('wiki.en.vec', cache='./')
        vectors = GloVe(name='42B', dim=300)

        self.vocab_size, self.embedding_dim = vectors.vectors.size()
        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight.data = vectors.vectors

        self.finetune_ebd = finetune_ebd

        if self.finetune_ebd:
            self.embedding_layer.weight.requires_grad = True
        else:
            self.embedding_layer.weight.requires_grad = False

    def forward(self, data, weights=None):
        '''
            @param text: batch_size * max_text_len
            @return output: batch_size * max_text_len * embedding_dim
        '''
        if (weights is None) or (self.finetune_ebd == False):
            return self.embedding_layer(data)

        else:
            return F.embedding(data['text'],
                               weights['ebd.embedding_layer.weight'])


class LSTMAtt(nn.Module):

    def __init__(self, ebd, out_dim, n_classes, total_classes, args=None):
        super(LSTMAtt, self).__init__()

        # ebd = WORDEBD(args['finetune_ebd'])

        self.args = args
        if args['dataset']=='20newsgroup':
            self.max_text_len=500
        elif args['dataset']=='fewrel':
            self.max_text_len=38
        elif args['dataset']=='huffpost':
            self.max_text_len=44

        self.ebd = ebd
        # self.aux = get_embedding(args)

        self.input_dim = self.ebd.embedding_dim  # + self.aux.embedding_dim

        # Default settings in induction encoder
        u = args['induct_rnn_dim']
        da = args['induct_att_dim']

        self.rnn = RNN(self.input_dim, u, 1, True, 0.5)

        # Attention
        self.head = nn.Parameter(torch.Tensor(da, 1).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(u * 2, da)

        self.ebd_dim = u * 2

        self.l1 = nn.Linear(self.ebd_dim, self.ebd_dim)
        self.l2 = nn.Linear(self.ebd_dim, out_dim)

        # last layer for few
        self.few_classify = nn.Linear(out_dim, n_classes)

        self.all_classify = nn.Linear(out_dim, total_classes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.ebd_dim, nhead=4)
        self.transformer= nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

        print(self.state_dict().keys())



    def _attention(self, x, text_len):
        '''
            text:     batch, max_text_len, input_dim
            text_len: batch, max_text_len
        '''
        batch_size, max_text_len, _ = x.size()

        proj_x = torch.tanh(self.proj(x.view(batch_size * max_text_len, -1)))
        att = torch.mm(proj_x, self.head)
        att = att.view(batch_size, max_text_len, 1)  # unnormalized

        # create mask
        idxes = torch.arange(max_text_len, out=torch.cuda.LongTensor(max_text_len,
                                                                     device=text_len.device)).unsqueeze(0)
        mask = (idxes < text_len.unsqueeze(1)).bool()
        att[~mask] = float('-inf')

        # apply softmax
        att = F.softmax(att, dim=1).squeeze(2)  # batch, max_text_len

        return att

    def forward(self, data, all_classify=False):
        """
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml
            @return output: batch_size * embedding_dim
        """

        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        ebd = self.ebd(data[:, :self.max_text_len])


        # add augmented embedding if applicable
        # aux = self.aux(data)

        # ebd = torch.cat([ebd, aux], dim=2)

        # result: batch_size, max_text_len, embedding_dim

        # apply rnn
        ebd = self.rnn(ebd, data[:, self.max_text_len])
        # result: batch_size, max_text_len, 2*rnn_dim

        #ebd=F.dropout(ebd,p=0.8, training=self.training)

        # apply attention
        alpha = self._attention(ebd, data[:, self.max_text_len])

        # aggregate
        ebd = torch.sum(ebd * alpha.unsqueeze(-1), dim=1)
        #ebd = ebd.mean(1)

        #x=F.dropout(ebd, p=0.5,training=self.training)


        #x = self.l1(ebd)
        #x = F.relu(x)
        #x = self.l2(x)


        if not all_classify:
            x=self.transformer(ebd)
            y = self.few_classify(x)
        else:
            x = self.l1(ebd)
            x = F.relu(x)
            x = self.l2(x)
            y = self.all_classify(x)
        return ebd, x, y


