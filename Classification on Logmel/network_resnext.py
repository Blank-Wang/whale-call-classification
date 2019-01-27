import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels


def resnext101_32x4d_(pretrained='imagenet', **kwargs):
    model = pretrainedmodels.resnext101_32x4d(pretrained=pretrained)
    model.avg_pool = nn.AvgPool2d((4, 5), stride=(1, 1))
    model.last_linear = nn.Linear(512 * 4, 16)
    return model


def resnext101_64x4d_(pretrained='imagenet', **kwargs):
    model = pretrainedmodels.resnext101_64x4d(pretrained=pretrained)
    model.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
    model.last_linear = nn.Linear(512 * 4, 16)
    return model

