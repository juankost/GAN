
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor, ToTensor, Normalize

class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir, num_workers=5, batch_size=16, augmentations=None):
        super().__init__()

        self.transform = transforms.Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        mnist_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=self.transform)
        self.mnist_test = mnist_testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


class CIFARDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size=64, num_workers=8):

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Download the dataset and compute the averages and std
        # Get the means for the CIFAR 10 dataset
        cifar_trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=ToTensor())
        data = cifar_trainset.data / 255  # data is numpy array
        mean = data.mean(axis=(0, 1, 2))
        std = data.std(axis=(0, 1, 2))
        del cifar_trainset

        # Define the data transformation for the images
        self.transform = transforms.Compose(
            [
                ToTensor(),
                Normalize(mean, std),
            ]
        )
        self.train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=self.transform,
                                              target_transform=None, download=True)
        self.val_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=self.transform,
                                              target_transform=None, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)




if __name__== "__main__":

    test = CIFARDataModule(data_dir="/Users/juankostelec/Google_drive/Projects/data", batch_size=1, num_workers=1)
    train_data = test.train_dataloader()
    for batch in train_data:
        print(batch[0][0,0])
        break