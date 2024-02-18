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
        # the output is of dimensions [batch_size, latent_dim]

        # compute the forward pass of the encoder using explicit use of all layers, and 
        # do not use self.layers(x) as it won't allow for keeping the intermediate
        # feature maps. Do this over a for loop, and keep the outputs of layers
        # corresponding to BatchNorm layers in a separate list.
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                skip_connections.append(x)
        return x, skip_connections[:-1]
        
        # return self.layers(x)

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
                
    def forward(self, x, skip_connections):
        
        # `x` has shape: [batch_size, latent_dim].

        # compute the forward pass of the decoder using explicit use of all layers, and 
        # do not use self.layers(x) as it won't allow for the addition of skip connections
        # Do this over a for loop, and use skip connections if the layer is a transposed convolution
        # layer. Note that the skip connections are in reverse order, i.e. the first skip connection
        # is the last one in the list
        skip_connect_count = 2
        for layer in self.layers:
            if isinstance(layer, nn.ConvTranspose2d) and skip_connect_count == 0:
                try:
                    skip = skip_connections.pop()
                    # print(f"layer: {layer}, \t x.shape:{x.shape}, \t layer(x).shape:{layer(x).shape}, \t skip_connections.shape:{skip.shape}")
                except:
                    skip = 0.0
                x = layer(x)
                x = x + skip
                # skip_connect_count += 1
            else:
                skip_connect_count -= 1
                x = layer(x)
        return x
        
        # # self.layers(x) has shape: [batch_size, width*height*num_channels]
        # return torch.reshape(self.layers(x), (-1, self.num_channels, self.width, self.height))
        

class CNNAERes(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder_cnn = hydra.utils.instantiate(self.hparams.encoder_cnn)
        self.decoder_cnn = hydra.utils.instantiate(self.hparams.decoder_cnn)


    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        z, skip_connections = self.encoder_cnn(image)
        recons = self.decoder_cnn(z, skip_connections)
        return torch.reshape(z, (z.shape[0], -1)), recons
        
        # # `image` has shape: [batch_size, num_channels, width, height].
        # z = self.encoder_cnn(image)
        # recons = self.decoder_cnn(z)
        # return torch.reshape(z, (z.shape[0], -1)), recons
