from pytorch_lightning import Trainer

from GANs.data import MNISTDataModule
from GANs.experiment import GAN




dm = MNISTDataModule(data_dir="/content", batch_size=128)
model = GAN(*dm.size())
trainer = Trainer(gpus=1, max_epochs=50, progress_bar_refresh_rate=5)
trainer.fit(model, dm)