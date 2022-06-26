import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class Generator(LightningModule):

    def __init__(self, latent_dim, img_shape, hidden_size=32, dropout=0.3):
        super().__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, hidden_size, normalize=False),
            *block(hidden_size, hidden_size*2, normalize=False),
            *block(hidden_size*2, hidden_size*4, normalize=False),
            nn.Linear(hidden_size*4, np.prod(img_shape), normalize=False),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), *self.img_shape)
        return out


class Discriminator(LightningModule):

    def __init__(self, img_shape, hidden_size=32, output_size=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*4, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, img):
        out = self.model(img.view(img.shape[0], -1))
        return out