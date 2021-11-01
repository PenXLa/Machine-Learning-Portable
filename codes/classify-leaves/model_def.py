import torch as pt
from torch import nn
import torchvision
from lib.utils import *


def Net(pthfile=None, device='cuda'):
    if pthfile is None:
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 176)
    else:
        model = torchvision.models.resnet18()
        model.load_state_dict(pt.load(pthfile, map_location=device))
    return model
