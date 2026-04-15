import torch
import torch.nn as nn

class DDoSModel(nn.Module):
    def __init__(self):
        super(DDoSModel, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x