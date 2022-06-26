import argparse
from pytorch_lightning import Trainer

from data import MNISTDataModule, CIFARDataModule
from experiment import GAN
from utils import DATA_DIR, AVAIL_GPUS



if __name__ == "__main__":
    # args = argparse.ArgumentParser()

    dm = MNISTDataModule(data_dir=DATA_DIR, batch_size=128)
    # dm = CIFARDataModule(data_dir=DATA_DIR, batch_size=128)
    model = GAN(*dm.size())
    trainer = Trainer(gpus=int(AVAIL_GPUS), max_epochs=50, progress_bar_refresh_rate=5)
    trainer.fit(model, dm)


# TODO: Better logging of the losses during training 40 min
# TODO: Adapt the exact loss how it's computed - remember the trick to stabilize the training 45 min
# TODO: Understand how exactly does pytorch lightning deal with multiple optimizers 30 min

# TODO: Adapt the model architecture for stronger discriminator/generator and see the impact 40 min --> Convolutional model based on DC GAN!!
# TODO: Implement a conditional GAN 1h