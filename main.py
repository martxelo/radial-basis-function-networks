import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from rbfn.utils import NumpyDataset
from rbfn.models import SimpleModel


# raw data
x, y = make_moons(n_samples=1024, noise=0.2, random_state=42)
x = x.astype(np.float32)
y = np.eye(2)[y].astype(np.float32)

# datasets
dataset = NumpyDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

# model
model = SimpleModel(neurons=6)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

device = "cuda" if torch.cuda.is_available() else "cpu"

# train
model.train()
for epoch in range(50):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        
    print(f"{total_loss=}")
