import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, x_numpy, y_numpy):
        self.x = torch.from_numpy(x_numpy)
        self.y = torch.from_numpy(y_numpy)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]