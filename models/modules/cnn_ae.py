import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import numpy as np


class Encoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in self.hparams.encoder_layers.items()]
        )        

    def forward(self, x):
        
        # input `x` or `image` has shape: [batch_size, num_channels, width, height].
        # the output is of dimensions [batch_size, latent_dim, 1, 1]
        x = self.layers(x)
        # if x is 2D, we need to add the extra dimensions to make it [batch_size, latent_dim, 1, 1]
        if len(x.shape) == 2:
            x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        return x


class Decoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.width = self.hparams.width
        self.height = self.hparams.height
        self.num_channels = self.hparams.num_channels
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in self.hparams.decoder_layers.items()]
        )
                
    def forward(self, x):

        # `x` has shape: [batch_size, latent_dim].
        # self.layers(x) has shape: [batch_size, width*height*num_channels]
        return torch.reshape(self.layers(x), (-1, self.num_channels, self.width, self.height))
        

class CNNAE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder_cnn = hydra.utils.instantiate(self.hparams.encoder_cnn)
        self.decoder_cnn = hydra.utils.instantiate(self.hparams.decoder_cnn)


    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].
        z = self.encoder_cnn(image)
        recons = self.decoder_cnn(z)
        return torch.reshape(z, (z.shape[0], -1)), recons

