import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

from rbfn.estimators import RBFNEstimator
from rbfn.utils import get_dataloader


class Model():

    def __init__(self, n_samples=1024, epochs=100, neurons=6):

        self.n_samples = n_samples
        self.epochs = epochs
        self.dataloader, self.x, self.y = get_dataloader(n_samples=n_samples)
        self.estimator = RBFNEstimator(neurons=neurons)
        self.x_bg = self.get_background_coordinates()
        self.centres = []
        self.radii = []
        self.pred_bg = []

    def get_background_coordinates(self):

        xmin, ymin = self.x.min(axis=0)
        xmax, ymax = self.x.max(axis=0)
        x = np.linspace(xmin, xmax, 50)
        y = np.linspace(ymin, ymax, 50)

        xx, yy = np.meshgrid(x, y)

        return np.c_[xx.reshape(-1), yy.reshape(-1)]

    def train(self):

        # loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.estimator.parameters())

        # device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # show total number of parameters
        # print("total params =",sum(p.numel() for p in self.estimator.parameters() if p.requires_grad))

        # train
        self.estimator.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in self.dataloader:
                x, y = x.to(device), y.to(device)

                pred = self.estimator(x)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            # add new parameters to the list
            self.radii.append(self.estimator.rbf.radii.detach().numpy())
            self.centres.append(self.estimator.rbf.centres.detach().numpy())

            # add predictions for the background
            pred_bg = self.estimator(torch.from_numpy(self.x_bg).to(device)).detach().numpy()
            self.pred_bg.append(pred_bg)

            # score
            pred = self.estimator(torch.from_numpy(self.x).to(device)).detach().numpy()
            score = f1_score(np.argmax(self.y, axis=1), np.argmax(pred, axis=1))

            # show some metrics
            if epoch % 10 == 0:
                print(f"{epoch=} {total_loss=:.4f} {score=:.4f}")


