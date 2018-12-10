import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    """Neural Network"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 2)

    def forward(self, x):
        # attempt at softmax regression
        x = F.softmax(self.fc1(x))
        return x

class GaussianSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self):
        super(GaussianSVM, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        h = self.fc(x)
        return h
