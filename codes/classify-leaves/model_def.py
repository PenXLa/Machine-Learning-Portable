import torch as pt
from torch import nn
import torchvision
from lib.utils import *
def Net():
    model = torchvision.models.resnet18(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 176)
    return model
