import torch
from torch import nn


class SimpleEstimator(nn.Module):
    def __init__(self, neurons=6):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, neurons),
            nn.Tanh(),
            nn.Linear(neurons, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_stack(x)
    

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.radii = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.radii, 1)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / self.radii.unsqueeze(0)
        return torch.exp(-distances.pow(2))
    

class RBFNEstimator(nn.Module):
    def __init__(self, neurons=6):
        super().__init__()
        self.rbf = RBFLayer(2, neurons)
        self.linear = nn.Linear(neurons, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        return self.sigmoid(x)
    
