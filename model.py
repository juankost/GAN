import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule


class Generator(LightningModule):

    def __init__(self, latent_dim, img_shape):
        super().__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), *self.img_shape)
        return out



class Discriminator(LightningModule):

    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.model(img.view(img.shape[0], -1))
        return out