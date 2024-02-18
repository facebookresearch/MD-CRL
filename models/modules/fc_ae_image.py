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
        
        # input `x` or `image` has shape: [batch_size, num_channels, width, height]. num_channels=1 for mnist
        return self.layers(x)
    

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
        

class FCAE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder_fc = hydra.utils.instantiate(self.hparams.encoder_fc)
        self.decoder_fc = hydra.utils.instantiate(self.hparams.decoder_fc)
        
    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].
        z = self.encoder_fc(image)
        recons = self.decoder_fc(z)

        return z, recons
