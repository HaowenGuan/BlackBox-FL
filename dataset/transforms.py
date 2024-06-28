import random
import timm
import torch
import torch.nn.functional as F

from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def get_fc100_transform():
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_mini_image_transform():
    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_pix, std=std_pix)
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_vit_224_size_transform(timm_model):
    # data_config = timm.data.resolve_model_data_config(timm_model)
    # transform  = timm.data.create_transform(**data_config, is_training=False)
    # train_transform = test_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     *transform.transforms[2:]
    # ])
    train_transform = test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_vit_original_size_transform():
    train_transform = test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_dino_transform():
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ],
                               p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # first global crop
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.14, 1.), interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        GaussianBlur(1.0),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.14, 1.), interpolation=Image.BICUBIC),
        normalize,
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_resnet_transform():
    transform = transforms.Compose([
        transforms.RandomChoice(
            [transforms.Resize(256),
             transforms.Resize(480)]),
        transforms.RandomCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return {
        'train_transform': transform,
        'test_transform': transform,
    }


def get_empty_transform():
    return {
        'train_transform': transforms.Compose([]),
        'test_transform': transforms.Compose([]),
    }


class GaussianBlur(object):
    """
  Apply Gaussian Blur to the PIL image.
  """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)))