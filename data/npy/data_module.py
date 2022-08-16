import numpy as np
import torch
from torch.utils.data import DataLoader


class NPYDataModule:
    def __init__(self, image_npy_file, label_npy_file):
        super().__init__()
        self.images = torch.from_numpy(np.load(image_npy_file))
        self.labels = torch.from_numpy(np.load(label_npy_file))
        self.datasets = torch.utils.data.TensorDataset(self.images, self.labels)

    def get_data_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return torch.utils.data.DataLoader(self.datasets, batch_size=batch_size, shuffle=shuffle, num_workers=16)
