import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import init


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(11, 32, bias=True)
        self.bn1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(32, 64, bias=True)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32, bias=True)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, num_classes, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        init.kaiming_normal(self.fc1.weight)
        init.kaiming_normal(self.fc2.weight)
        init.kaiming_normal(self.fc3.weight)
        init.kaiming_normal(self.fc4.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

