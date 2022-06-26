import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import MNISTDataModule, CIFARDataModule
from experiment import GAN
from utils import DATA_DIR, AVAIL_GPUS, CHECKPOINTS_DIR, LOG_DIR



if __name__ == "__main__":
    # Choose the dataset used
    dm = MNISTDataModule(data_dir=DATA_DIR, batch_size=128)
    # dm = CIFARDataModule(data_dir=DATA_DIR, batch_size=128)

    # Load the model
    model = GAN(*dm.size())

    # Checkpointing
    regular_checkpoint_callback = ModelCheckpoint(
        save_top_k=20,
        monitor="epoch",
        mode="max",
        dirpath=CHECKPOINTS_DIR,
        filename="regular-{epoch:02d}-{global_step}",
    )

    # Logger
    logger = TensorBoardLogger(save_dir=LOG_DIR, name="GAN")
    trainer = Trainer(logger=logger,
                      callbacks=[regular_checkpoint_callback],
                      gpus=int(AVAIL_GPUS),
                      max_epochs=500,
                      log_every_n_steps=1)
    trainer.fit(model, dm)

# TODO: Debug GAN --> Generator is basically not training: look at loss function used, data augmentaiton, model specs,
# TODO: Implement a conditional GAN 1h
# TODO: Implement DC GAN  --> Train on the CIFAR 10 dataset!

# TODO: Missing rescaling step in the training, i.e. the generator produces images [-1, 1], while the real images are [0,1]