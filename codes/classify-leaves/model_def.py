import torch as pt
from torch import nn
import torchvision


# 未训练的网络
def RawNet():
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 176)
    return model


# 加载预训练，未微调的网络
def RareNet():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 176)
    return model


# 加载训练好的网络
def WellDoneNet(pthfile, device='cuda'):
    model = RawNet()
    model.load_state_dict(pt.load(pthfile, map_location=device))
    return model


