import torch as pt
from torch import nn
import torchvision
from lib.utils import *


def Net(pthfile=None, device='cuda'):
    model = torchvision.models.resnet18(pretrained=pthfile is None)
    model.fc = nn.Linear(model.fc.in_features, 176)
    if pthfile is not None:
        model.load_state_dict(pt.load(pthfile, map_location=device))
    return model
