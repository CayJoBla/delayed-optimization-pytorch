from torchvision import datasets, transforms

from .dataset import Dataset

class CIFAR10(Dataset):
    def __init__(self, train_size=40000):
        super().__init__()
        self.name = "cifar10"
        self.train_data = datasets.CIFAR10(
            root=self.root_data_dir,
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.test_data = datasets.CIFAR10(
            root=self.root_data_dir, 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.train_size = train_size


class CIFAR100(Dataset):
    def __init__(self, train_size=40000):
        super().__init__()
        self.name = "cifar100"
        self.train_data = datasets.CIFAR100(
            root=self.root_data_dir,
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.test_data = datasets.CIFAR100(
            root=self.root_data_dir, 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.train_size = train_size