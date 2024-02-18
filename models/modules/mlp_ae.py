import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in kwargs['encoder_layers'].items()]
        )

    def forward(self, x):

        # input `x` has shape: [batch_size, x_dim]
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in kwargs['decoder_layers'].items()]
        )
                
    def forward(self, z):
        
        # `z` has shape: [batch_size, z_dim].

        # [batch_size, x_dim]
        return self.layers(z)
        

class FCAE(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()

        self.encoder = hydra.utils.instantiate(kwargs['encoder'])
        self.decoder = hydra.utils.instantiate(kwargs['decoder'])

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # `x` has shape: [batch_size, x_dim]
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat
