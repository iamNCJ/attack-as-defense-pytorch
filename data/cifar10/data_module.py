import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class CIFAR10DataModule:
    def __init__(self, root_dir):
        super().__init__()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.datasets = datasets.CIFAR10(root=root_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    def get_data_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return torch.utils.data.DataLoader(self.datasets, batch_size=batch_size, shuffle=shuffle)
