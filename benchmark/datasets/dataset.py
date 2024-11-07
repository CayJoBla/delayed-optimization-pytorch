from torch.utils import data
import numpy as np

class Dataset:
    def __init__(self):
        self.root_data_dir = "data"

    def initialize_dataloaders(self, batch_size):
        self.train_loader, self.val_loader = self._create_train_val_dataloaders(
            self.train_data, batch_size
        )
        self.test_loader = self._create_test_dataloader(self.test_data, batch_size)
    
    def _create_train_val_dataloaders(self, dataset, batch_size):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        train_sampler = data.sampler.SubsetRandomSampler(indices[:self.train_size])
        train_loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            drop_last=True
        )
        val_sampler = data.sampler.SubsetRandomSampler(indices[self.train_size:])
        val_loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=val_sampler, 
            drop_last=False
        )
        return train_loader, val_loader

    def _create_test_dataloader(self, dataset, batch_size):
        return data.DataLoader(dataset, batch_size=batch_size)
