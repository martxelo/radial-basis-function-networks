import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons


class NumpyDataset(Dataset):
    def __init__(self, x_numpy, y_numpy):
        self.x = torch.from_numpy(x_numpy)
        self.y = torch.from_numpy(y_numpy)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

def get_dataloader(n_samples=1024):

    # raw data
    x, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    x = x.astype(np.float32)
    y = np.eye(2)[y].astype(np.float32)

    # datasets
    dataset = NumpyDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)

    return dataloader, x, y