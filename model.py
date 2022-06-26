import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

"""
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
            nn.Linear(hidden_size*4, np.prod(img_shape)),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), *self.img_shape)
        return out


class Discriminator(LightningModule):

    def __init__(self, img_shape, hidden_size=32, output_size=1, dropout=0.3):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *block(np.prod(img_shape), hidden_size*4, normalize=False),
            *block(hidden_size*4, hidden_size*2, normalize=False),
            *block(hidden_size*2, hidden_size, normalize=False),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, img):
        out = self.model(img.view(img.shape[0], -1))
        return out
        
"""

import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)

        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim, output_size)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # flatten image
        x = x.view(-1, 28 * 28)
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer
        out = self.fc4(x)

        return out


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)

        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim * 4, output_size)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2)  # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer with tanh applied
        out = F.tanh(self.fc4(x))

        return out