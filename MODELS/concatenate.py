import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from MODELS.model_resnet import *
from MODELS.model_mlp import *


class TwoInputNet(nn.Module):
    def __init__(self, num_classes, depth, att_type=None):
        super(TwoInputNet, self).__init__()
        self.cnn = ResidualNet('ImageNet', depth, num_classes, att_type)
        self.mlp = MLP(num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, input_1, input_2):   # output = model(images, features)..
        cnn_out = self.relu(self.cnn(input_1))
        mlp_out = self.relu(self.mlp(input_2))
        combined = torch.cat((0.6 * cnn_out, 0.4 * mlp_out), 1)

        x = self.fc(combined)
        return x

