from torchvision import datasets, transforms

from .dataset import Dataset

class MNIST(Dataset):
    def __init__(self, train_size=50000):
        super().__init__()
        self.name = "mnist"
        self.train_data = datasets.MNIST(
            root=self.root_data_dir,
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.test_data = datasets.MNIST(
            root=self.root_data_dir, 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.train_size = train_size