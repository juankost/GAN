
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir, num_workers=5, batch_size=16, augmentations=None):
        super().__init__()


        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
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
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

