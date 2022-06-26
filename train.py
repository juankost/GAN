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
        monitor="global_step",
        mode="max",
        dirpath=CHECKPOINTS_DIR,
        filename="regular-{epoch:02d}-{global_step}",
    )

    # Logger
    logger = TensorBoardLogger(save_dir=LOG_DIR, name="GAN_logs")
    trainer = Trainer(logger=logger,
                      callbacks=[regular_checkpoint_callback],
                      gpus=int(AVAIL_GPUS),
                      max_epochs=500,
                      log_every_n_steps=1)
    trainer.fit(model, dm)

# TODO: Understand the performance of the generator/discriminator from the loss values/functions
# TODO: Improve quality on the MNIST dataset
# TODO: Adapt the model architecture for stronger discriminator/generator and see the impact 40 min --> Convolutional model based on DC GAN!!
# TODO: Implement a conditional GAN 1h