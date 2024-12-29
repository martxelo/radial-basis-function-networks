from torch import nn


class SimpleModel(nn.Module):
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
