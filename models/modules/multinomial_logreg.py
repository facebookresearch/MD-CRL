import torch
from torch import nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        return self.linear(x)