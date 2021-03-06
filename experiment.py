from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch
import torchvision

from model import Discriminator, Generator


class GAN(LightningModule):

    def __init__(self, channels, width, height, latent_dim=100, lr=0.0002,
                 b1=0.5, b2=0.999, batch_size=16):
        super().__init__()
        self.save_hyperparameters()

        # Networks
        data_shape = (channels, width, height)
        # self.generator = Generator(latent_dim, data_shape)
        # self.discriminator = Discriminator(data_shape)
        self.generator = Generator(100, 32, 784)
        self.discriminator = Discriminator(784, 32, 1)

        self.validation_z = torch.randn(16, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

        self.criterion = nn.BCEWithLogitsLoss()
    def sample_generator(self, batch_size=1):
        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim).float()
        imgs = self.generator(z)
        return imgs

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        # return F.binary_cross_entropy(y_hat, y)
        return self.criterion(y_hat, y)

    def training_step(self, batch, batch_ids, optimizer_idx):

        imgs, _ = batch

        # Rescale the images to be [-1, 1] instead of [0, 1]
        imgs = imgs * 2 - 1

        # Sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Train generator
        if optimizer_idx == 0:

            # Generate img
            self.generated_imgs = self.generator(z)

            # Log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, self.trainer.global_step)

            # Ground truth result (all fake images)
            valid = torch.ones(imgs.size(0), 1)  # Valid = 1 because the generator is trying to fool the discriminator
            valid = valid.type_as(imgs)

            # Adversarial loss
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log("g_loss", g_loss, prog_bar=True)
            return output

        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(self.discriminator(self(z)), fake)

            # Discriminator loss
            d_loss = (fake_loss + real_loss) / 2.0
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log("d_loss", d_loss, prog_bar=True)
            self.log("d_real_sample_loss", real_loss, prog_bar=False)
            self.log("d_fake_sample_loss", fake_loss, prog_bar=False)
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_training_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # Log sampled image
        sample_imgs = self.generator(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)