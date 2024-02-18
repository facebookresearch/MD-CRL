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
        print(kwargs['decoder_layers'].items())
        for layer_key, layer in kwargs['decoder_layers'].items():
            self.layers = layer
                
    def forward(self, z):
        
        # `z` has shape: [batch_size, z_dim].

        # [batch_size, x_dim]
        return self.layers(z)
        

class FCAEPoly(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()

        self.encoder = hydra.utils.instantiate(kwargs['encoder'])
        self.decoder = hydra.utils.instantiate(kwargs['decoder'])
        print(f"encoder: {self.encoder}")
        print(f"decoder: {self.decoder}")
        
    def forward(self, x):
        # `x` has shape: [batch_size, x_dim]
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat
